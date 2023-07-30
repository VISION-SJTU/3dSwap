import torch
from torch import nn
from models.encoders.model_irse import Backbone
import sys

sys.path.append(".")
sys.path.append("..")


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.4, mode='ir_se')
        self.facenet.load_state_dict(torch.load('models/model_ir_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

    def extract_feats(self, x):
        factor = int(x.shape[-1] / 256)
        x = x[:, :, 35 * factor:223 * factor, 32 * factor:220 * factor]

        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_hat_feats = self.extract_feats(y_hat)
        x_feats = x_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(x_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count
