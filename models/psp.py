"""
This file defines the core research contribution
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')
import math
import cv2

import torch
from torch import nn
from models.encoders import psp_encoders
from .networks import define_D, define_mlp
import torch.nn.functional as F

import dnnlib
from utils import legacy

from training.triplane import TriPlaneGenerator
from torch_utils import misc
from tqdm import tqdm

from lpips import LPIPS
from models.id_loss import IDLoss
from models.w_norm import WNormLoss


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts, train_faceswap=True):
        super(pSp, self).__init__()
        self.opts = opts

        self.encoder = self.set_encoder().to(self.opts.device)

        if train_faceswap:
            encoder_ckpt = torch.load('checkpoints/encoder.pt')
            self.encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)

            for i in range(5):
                mlp = define_mlp(4)
                setattr(self, f'MLP{i}', mlp.train())

        with dnnlib.util.open_url('checkpoints/ffhq512-128.pkl') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.opts.device)
            G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(self.opts.device)
            misc.copy_params_and_buffers(G, G_new, require_all=True)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G_new.rendering_kwargs = G.rendering_kwargs
            self.decoder = G_new

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def detector(self, frame):
        rects = self.align.getAllFaceBoundingBoxes(frame)
        landmarks = {}
        if len(rects) > 0:
            bb = self.align.findLandmarks(frame, rects[0])
            for i in range(68):
                landmarks[i] = bb[i]
        return landmarks

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se')
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def Inversion(self, x, cp, rand_cp, w_avg):
        cp = cp.squeeze(dim=1)
        rand_cp = rand_cp.squeeze(dim=1).repeat(x.shape[0], 1)
        x = F.interpolate(x, size=[256, 256], mode='bilinear', align_corners=True)
        wx = self.encoder(x) + w_avg
        x_prime = self.decoder.synthesis(wx, cp)['image']
        x_hat = self.decoder.synthesis(wx, rand_cp)['image']
        wx_hat = self.encoder(F.interpolate(x_hat, size=[256, 256], mode='bilinear', align_corners=True)) + w_avg
        x_hat_prime = self.decoder.synthesis(wx_hat, cp)['image']

        return x_prime, x_hat, x_hat_prime, wx, wx_hat

    def ger_average_color(self, mask, arms):
        color = torch.zeros(arms.shape).cuda()
        mask = mask.repeat([arms.shape[0], 1, 1, 1])
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 10:
                color[i, 0, :, :] = 0
                color[i, 1, :, :] = 0
                color[i, 2, :, :] = 0

            else:
                color[i, 0, :, :] = arms[i, 0, :, :].sum() / count
                color[i, 1, :, :] = arms[i, 1, :, :].sum() / count
                color[i, 2, :, :] = arms[i, 2, :, :].sum() / count
        return color

    def my_acti(self, w):
        return 1 / (1 + torch.exp(-100 * (w - 0.5)))

    def FaceSwap(self, x, y, x_cp, y_cp, w_avg):
        with torch.no_grad():
            x_cp = x_cp.squeeze(1)
            y_cp = y_cp.squeeze(1)
            x = F.interpolate(x, size=[256, 256], mode='bilinear', align_corners=True)
            y = F.interpolate(y, size=[256, 256], mode='bilinear', align_corners=True)
            x_ws = self.encoder(x) + w_avg
            y_ws = self.encoder(y) + w_avg
            x_rec = self.decoder.synthesis(x_ws, x_cp)['image']
            y_rec = self.decoder.synthesis(y_ws, y_cp)['image']

        x_codes, y_codes = [], []

        start_index = 5
        index_length = 5

        for i in range(start_index, start_index + index_length):
            x_codes.append(x_ws[:, i: i + 1])
            y_codes.append(y_ws[:, i: i + 1])

        yhat_codes = []
        yhat_codes.append(y_ws[:, :start_index])
        for i in range(start_index, start_index + index_length):
            i = i - start_index
            MLP = getattr(self, f'MLP{i}')
            rho = MLP(torch.cat([x_codes[i], y_codes[i]], dim=2))
            rho = (rho - rho.min()) / (rho.max() - rho.min())
            rho = self.my_acti(rho)
            yhat_codes.append(y_codes[i] * rho + x_codes[i] * (1 - rho))
        yhat_codes.append(y_ws[:, start_index + index_length:])

        ws = torch.cat(yhat_codes, dim=1)
        y_hat = self.decoder.synthesis(ws, y_cp)['image']
        y_rand = self.decoder.synthesis(ws, x_cp)['image']

        return x_rec, y_rec, y_hat, y_rand

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg_2d = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg_2d = self.latent_avg_2d.repeat(repeat, 1)
        else:
            self.latent_avg = None
