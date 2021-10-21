import torch.nn as nn
import torch
import torch as ch
from torchvision.models.utils import load_state_dict_from_url
from collections import OrderedDict

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

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
    
class Flatten(nn.Module):
    def forward(self, x):
        x = torch.flatten(x, 1)
        return x
    
class VGG(nn.Sequential):

    def __init__(self, features, num_classes=1000, init_weights=True, 
                 mean=ch.tensor([0.4850, 0.4560, 0.4060]), 
                 std=ch.tensor([0.2290, 0.2240, 0.2250])):

        features = features
        sequence, self.sequence_dict = [('normalize', InputNormalize(mean, std))], {}
        layer_num = -1
        for fi, ff in enumerate(features):
            fn, f = ff[0], ff[1]
            if isinstance(f, ch.nn.Conv2d):
                layer_num += 1
                if layer_num > 0:
                    sequence[-1] = (sequence[-1][0], 
                                    ch.nn.Sequential(OrderedDict(sequence[-1][1])))
                sequence.append((f'layer{layer_num}', [(fn, f)]))
            else:
                sequence[-1][1].append((fn, f))
            self.sequence_dict[f'features.{fi}'] = f'layer{layer_num}.{fn}'
        sequence[-1] = (sequence[-1][0], 
                        ch.nn.Sequential(OrderedDict(sequence[-1][1])))
                
        avgpool = nn.AdaptiveAvgPool2d((7, 7))
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        sequence.extend([#('features', features),
                    ('avgpool', avgpool),
                    ('flatten', Flatten()),
                    ('classifier', classifier)
                      ])
        
        super().__init__(OrderedDict(sequence))
        
        if init_weights:
            self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
class VGG_OG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.last_relu = nn.ReLU()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        feats = self.features(x)
        pooled = self.avgpool(feats)
        x = pooled.view(pooled.size(0), -1)
        x_latent = self.classifier[:4](x)
        x_relu = self.last_relu_fake(x_latent) if fake_relu \
                    else self.last_relu(x_latent)
        x_out = self.classifier[-2:](x_relu)
        return x_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [('conv', conv2d), 
                           ('bn', nn.BatchNorm2d(v)),
                           ('relu', nn.ReLU())]
            else:
                layers += [('conv', conv2d), ('relu', nn.ReLU())]
            in_channels = v
    return layers


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)