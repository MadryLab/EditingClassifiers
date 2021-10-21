import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
import torch as ch
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
           'wide_resnet50_3', 'wide_resnet50_4', 'wide_resnet50_5', 
           'wide_resnet50_6', ]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

##

class Twin(nn.Module):
    def forward(self,  x):
        return x, x.clone()


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x[0]), x[1]


class Residual(nn.Module):
    def __init__(self, downsample):
        super().__init__()
        self.downsample = downsample

    def forward(self,  x):
        identity = x[1]
        if self.downsample is not None:
            identity = self.downsample(x[1])
        return x[0] + identity
    
class finalBlock(nn.Sequential):
    expansion = 4
    __constants__ = ['downsample']
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(finalBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
    
        super().__init__(OrderedDict([
            ('conv3', Wrapper(conv1x1(width, planes * self.expansion))),
            ('bn3', Wrapper(norm_layer(planes * self.expansion))),
            ('residual', Residual(downsample)),
            ('relu', nn.ReLU(inplace=False))
        ]))

class finalBlock18(nn.Sequential):
    expansion = 4
    __constants__ = ['downsample']
    
    def __init__(self, planes, downsample=None, norm_layer=None):
        super(finalBlock18, self).__init__()
    
        super().__init__(OrderedDict([
            ('conv2',  Wrapper(conv3x3(planes, planes))),
            ('bn2', Wrapper(norm_layer(planes))),
            ('residual', Residual(downsample)),
            ('relu', nn.ReLU()),
        ]))
        
class SeqBasicBlock(nn.Sequential):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SeqBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        super().__init__(OrderedDict([
            ('twin', Twin()),
            ('conv1', Wrapper(conv3x3(inplanes, planes, stride))),
            ('bn1', Wrapper(norm_layer(planes))),
            ('relu1', Wrapper(nn.ReLU())),
            ('final', finalBlock18(planes, 
                                   downsample=downsample,
                                    norm_layer=norm_layer))
        ]))    
    
class SeqBottleneck(nn.Sequential):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SeqBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
    
        super().__init__(OrderedDict([
            ('twin', Twin()),
            ('conv1', Wrapper(conv1x1(inplanes, width))),
            ('bn1', Wrapper(norm_layer(width))),
            ('relu1', Wrapper(nn.ReLU())),
            ('conv2',  Wrapper(conv3x3(width, width, stride, groups, dilation))),
            ('bn2', Wrapper(norm_layer(width))),
            ('relu2', Wrapper(nn.ReLU())),
            ('final', finalBlock(inplanes, planes, stride=stride,
                                downsample=downsample, groups=groups, 
                                base_width=base_width, dilation=dilation,
                                norm_layer=norm_layer)),
            #('conv3', Wrapper(conv1x1(width, planes * self.expansion))),
            #('bn3', Wrapper(norm_layer(planes * self.expansion))),
            #('residual', Residual(downsample)),
            #('relu', nn.ReLU(inplace=False))
        ]))    
        
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return self.relu(out)

class InputNormalize(ch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        self.new_std = new_std[..., None, None]
        self.new_mean = new_mean[..., None, None]

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean.to(x.device))/self.new_std.to(x.device)
        return x_normalized
    
class InitBlock(nn.Sequential):

    def __init__(self, inplanes, norm_layer, mean, std):
        super(InitBlock, self).__init__()
        super().__init__(OrderedDict([
            ('normalize', InputNormalize(mean, std)),
            ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm_layer(inplanes)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        
class FinalBlock(nn.Sequential):

    def __init__(self, block, num_classes):
        super(FinalBlock, self).__init__()
        super().__init__(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', Flatten()),
            ('fc', nn.Linear(512 * block.expansion, num_classes)),
        ]))

class Flatten(nn.Module):
    def forward(self, x):
        x = torch.flatten(x, 1)
        return x
        
class ResNet(nn.Sequential):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, mean=ch.tensor([0.4850, 0.4560, 0.4060]), 
                 std=ch.tensor([0.2290, 0.2240, 0.2250])):
        #super(ResNetSeq, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
                
        sequence = [('layer0', InitBlock(self.inplanes, norm_layer, mean, std))]
        layer1 = self._make_layer(block, 64, layers[0])
        layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.sequence_dict = {'layer0': 'layer0'}
        layer_num = 1
        for opi, op_layer in enumerate([layer1, layer2, layer3, layer4]):
            for li, l in enumerate(op_layer):
                self.sequence_dict[f'layer{opi + 1}.{li}'] = f'layer{layer_num}'
                sequence.append((f'layer{layer_num}', l))
                layer_num += 1
        sequence.append((f'layer{layer_num}', FinalBlock(block, num_classes)))
        self.sequence_dict.update({'layer5': f'layer{layer_num}'})
        
        super().__init__(OrderedDict(sequence))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, SeqBottleneck):
                    nn.init.constant_(m.bn3.module.weight, 0)
                elif isinstance(m, SeqBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return layers 
    
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', SeqBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', SeqBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_og(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50_og', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)