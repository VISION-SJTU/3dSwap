import torch.nn as nn
from .stylegan2.model import EqualLinear
import torch
from .discriminator import MultiscaleDiscriminator


def define_mlp(layers_num):
    layers = [EqualLinear(1024, 512)]
    for _ in range(layers_num - 1):
        layers.append(EqualLinear(512, 512))
    mlp = nn.Sequential(*layers)
    return mlp.cuda()

def define_D(input_nc, n_layers=3, norm_layer=torch.nn.BatchNorm2d):
    netD = MultiscaleDiscriminator(input_nc, n_layers=n_layers, norm_layer=norm_layer, use_sigmoid=False)
    netD.cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netD.apply(weights_init)
    return netD