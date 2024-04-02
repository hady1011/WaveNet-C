import argparse
import os
import shutil
import time
import math
import random

import torch
import torch.distributed as dist
import torchvision.models as models

import numpy as np
from utils.dist_utils import dist_print, DistSummaryWriter
import datetime
from utils.utils import CosineAnnealingLR, CrossEntropyLabelSmooth
from model.WaveNet import wn_resnet_18, wn_resnet_34, wn_resnet_50, wn_resnet_101, wn_resnet_152

from model.fcanet import fcanet34, fcanet50, fcanet101, fcanet152
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


def parse():
#    model_names = sorted(name for name in models.__dict__
#                     if name.islower() and not name.startswith("__")
#                     and callable(models.__dict__[name])) + ['fcanet34', 'fcanet50', 'fcanet101', 'fcanet152']
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ['wn_resnet_18', 'wn_resnet_34', 'wn_resnet_50', 'wn_resnet_101', 'wn_resnet_152']

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*', default=['/home/hs028/Documents/DATA/ImgNet/train/'],
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='wn_resnet_34',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--evaluate_model', type=str, default=None, help='the model for evaluation')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--dali_cpu', action='store_true',default = False,
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str, default=None)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    parser.add_argument('--work_dir', type=str, default = './log')
    parser.add_argument('--note', type=str, default='')
    args = parser.parse_args()
    return args


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 shard_id, num_shards, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir,
                                    shard_id=args.local_rank,
                                    num_shards=args.world_size,
                                    random_shuffle=True,
                                    pad_last_batch=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device,
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        dist_print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 size, shard_id, num_shards):
        super(HybridValPipe, self).__init__(batch_size,
                                           num_threads,
                                            device_id,
                                            seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir,
                                    shard_id=args.local_rank,
                                    num_shards=args.world_size,
                                    random_shuffle=False,
                                    pad_last_batch=True)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.res = ops.Resize(device="cpu",
                              resize_shorter=size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True


def main():
    time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    global best_prec1, args
    best_prec1 = 0
    args = parse()
    
    print(args)
    if not len(args.data):
        raise Exception("error: No data set provided")
        
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # make apex optional
    if args.opt_level is not None or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    if args.opt_level is None and args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

    dist_print("opt_level = {}".format(args.opt_level))
    dist_print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    dist_print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    dist_print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    torch.backends.cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        # torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
        setup_seed(0)


    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    args.work_dir = os.path.join(args.work_dir, time_stamp + args.arch + args.note)
    if not args.evaluate:
        if args.local_rank == 0:
            os.makedirs(args.work_dir)
        logger = DistSummaryWriter(args.work_dir)

    # create model
    if args.pretrained:
        dist_print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == 'fcanet34':
            model = fcanet34(pretrained=True)
        elif args.arch == 'fcanet50':
            model = fcanet50(pretrained=True)
        elif args.arch == 'fcanet101':
            model = fcanet101(pretrained=True)
        elif args.arch == 'fcanet152':
            model = fcanet152(pretrained=True)
        else:
            model = models.__dict__[args.arch](pretrained=True)
    else:
        dist_print("=> creating model '{}'".format(args.arch))
        if args.arch == 'fcanet34':
            model = fcanet34()
        elif args.arch == 'fcanet50':
            model = fcanet50()
            print('i created the model')
        elif args.arch == 'fcanet101':
            model = fcanet101()
        elif args.arch == 'fcanet152':
            model = fcanet152()
        elif args.arch == 'wn_resnet_18':
            model = wn_resnet_18()
        elif args.arch == 'wn_resnet_34':
            model = wn_resnet_34()
        elif args.arch == 'wn_resnet_50':
            model = wn_resnet_50()
        elif args.arch == 'wn_resnet_101':
            model = wn_resnet_101()
        elif args.arch == 'wn_resnet_152':
            model = wn_resnet_152()
        else:
            model = models.__dict__[args.arch]()

    if args.sync_bn:
        dist_print("using apex synced BN")
        model = parallel.convert_syncbn_model(model)
    
    # if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
    #     if args.channels_last:
    #         memory_format = torch.channels_last
    #     else:
    #         memory_format = torch.contiguous_format
    #     model = model.cuda().to(memory_format=memory_format)
    # else:
    #    model = model.cuda()
    print('about to cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
   # print('i am here 1') 
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#trainable Params = ',pytorch_total_params)
    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size*args.world_size)/256.
    print(args.lr)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #print('i am here') 
    
    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.opt_level is not None:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        if args.opt_level is not None:
            model = DDP(model, delay_allreduce=True)
        else:
            model = DDP(model, device_ids=[args.local_rank], output_device = args.local_rank)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                dist_print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))

                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                dist_print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                dist_print("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    if args.evaluate:
        assert args.evaluate_model is not None
        dist_print("=> loading checkpoint '{}' for eval".format(args.evaluate_model))
        checkpoint = torch.load(args.evaluate_model, map_location = lambda storage, loc: storage.cuda(args.gpu))
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
        else:
            state_dict_with_module = {}
            for k,v in checkpoint.items():
                state_dict_with_module['module.'+k] = v
            model.load_state_dict(state_dict_with_module)


    # Data loading code
    if len(args.data) == 1:
        
        traindir = '/home/hs028/Documents/DATA/ImgNet/train/'
        valdir = '/home/hs028/Documents/DATA/ImgNet/val/'
        print('traindir',traindir)
        print('valdir',valdir)
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 256
        val_size = 256
    print('about to create Train pipeline')
    print(args.batch_size,args.workers,args.local_rank,
                           traindir,
                           crop_size,
                           args.dali_cpu,
                           args.local_rank,
                           args.world_size)    
    pipe = HybridTrainPipe(batch_size=args.batch_size,
                           num_threads=args.workers,
                           device_id=args.local_rank,
                           data_dir=traindir,
                           crop=crop_size,
                           dali_cpu=args.dali_cpu,
                           shard_id=args.local_rank,
                           num_shards=args.world_size)
    print('about to build pipeline') 
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=False)

    pipe = HybridValPipe(batch_size=args.batch_size,
                         num_threads=args.workers,
                         device_id=args.local_rank,
                         data_dir=valdir,
                         crop=crop_size,
                         size=val_size,
                         shard_id=args.local_rank,
                         num_shards=args.world_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=False)


    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = CrossEntropyLabelSmooth().cuda()

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    len_epoch = int(math.ceil(train_loader._size / args.batch_size))
    print('len_epoch',len_epoch)
    T_max = 95 * len_epoch
    print('T_max',T_max)
    warmup_iters = 5 * len_epoch
    print('warmup',warmup_iters)
    scheduler = CosineAnnealingLR(optimizer, T_max, warmup='linear', warmup_iters=warmup_iters)
    #scheduler = StepLR(optimizer, step_size=30,gamma=0.1) 
    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('learning_rate',optimizer)
        print(epoch)
                
        avg_train_time = train(train_loader, model, criterion, optimizer, epoch, logger, scheduler)
        ###
        #avg_train_time = 0.4
        ###
        #scheduler.step()
        #optimizer.step()
        total_time.update(avg_train_time)
        torch.cuda.empty_cache()
        # evaluate on validation set
        [prec1, prec5] = validate(val_loader, model, criterion)
        ###
        #prec1 = 0.1
        #prec5 = 0.2
        ###
        logger.add_scalar('Val/prec1', prec1, global_step=epoch)
        logger.add_scalar('Val/prec5', prec5, global_step=epoch)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, work_dir = args.work_dir)
            if epoch == args.epochs - 1:
                print(args.total_batch_size / total_time.avg)
                print(best_prec1)
                #dist_print('##Best Top-1 {0}\n##Perf  {2}'.format(best_prec1, args.total_batch_size / total_time.avg))
                with open(os.path.join(args.work_dir, 'res.txt'), 'w') as f:
                    f.write('arhc: {0} \n best_prec1 {1}'.format(args.arch+args.note, best_prec1))

        train_loader.reset()
        val_loader.reset()
        

def train(train_loader, model, criterion, optimizer, epoch, logger, scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
   
    # switch to train mode
    model.train()
    end = time.time()
    
    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

        if args.prof >= 0 and i == args.prof:
            dist_print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        #scheduler.step()
        
        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.
            
            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.5f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
                
                logger.add_scalar('Train/loss', losses.val, global_step = epoch * train_loader_len + i)
                logger.add_scalar('Train/top1', top1.val, global_step = epoch * train_loader_len + i)
                logger.add_scalar('Train/top5', top5.val, global_step = epoch * train_loader_len + i)
                logger.add_scalar('Meta/lr', optimizer.param_groups[0]['lr'], global_step=epoch * train_loader_len + i)
        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
        
    scheduler.step()
    return batch_time.avg

@torch.no_grad()
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    dist_print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, work_dir = './', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(work_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(work_dir, filename), os.path.join(work_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 0.5
##changed this to 0.5 instead of 1
    lr = args.lr*(0.1**factor)


    # """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # scheduler.step()
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        x = correct[:k].contiguous().view(-1)
        correct_k = x.float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
