import torch.nn as nn
from .stylegan2.model import EqualLinear


def define_mlp(layers_num):
    layers = [EqualLinear(1024, 512)]
    for _ in range(layers_num - 1):
        layers.append(EqualLinear(512, 512))
    mlp = nn.Sequential(*layers)
    return mlp.cuda()
