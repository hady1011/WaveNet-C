import torch.nn as nn
import torch
import math
import model.WaveletAttention_WAV as WAV
import model.FWT_MODULE as FWT


__all__ = ['WNNet', 'wn_resnet_18', 'wn_resnet_34', 'wn_resnet_50', 'wn_resnet_101',
           'wn_resnet_152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self._process: nn.Module = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Chunk_Size = 16
        
        self.planes = planes
        
        self.AttentionalWAV = WAV.WaveletAttention_WAV(planes, 'Basic')
        
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self._process(x)
        splits = torch.split(out, self.Chunk_Size, dim=1)
        attention = self.AttentionalWAV([FWT.FWT_MODULE.build(i) for i in range(len(splits))], splits)
        
        attention += residual
        activated = torch.relu(attention)
        
        return activated


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.AttentionalWAV = WAV.WaveletAttention_WAV(planes,'Bottle')
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.AttentionalWAV(out)
    
        out += residual
        out = self.relu(out)

        return out


class WNNet(nn.Module):
#make 1000
    def __init__(self, block, layers, num_classes=1000):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inplanes = 64
        super(WNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #print('Block_Exp',block.expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.to(self._device)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print('x_layer4',x.shape)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print('x',x.shape)
        x = self.fc(x)

        return x


def wn_resnet_18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WNNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def wn_resnet_34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WNNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def wn_resnet_50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WNNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def wn_resnet_101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WNNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def wn_resnet_152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WNNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model