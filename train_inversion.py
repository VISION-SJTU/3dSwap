import os
import json
import sys
import pprint
import torch
import torch.distributed as dist
from models.psp import pSp
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from options.train_options import TrainOptions
from torch.nn.parallel import DistributedDataParallel as DDP
from training.ranger import Ranger
import numpy as np
from utils.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
import torch.nn.functional as F
import torch.nn as nn
from lpips import LPIPS
from models.id_loss import IDLoss
from models.w_norm import WNormLoss
import PIL.Image as Image

sys.path.append(".")
sys.path.append("..")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


class Criterion(nn.Module):
    def __init__(self, opts):
        super(Criterion, self).__init__()
        self.mse_loss = nn.MSELoss().to(opts.device).eval()
        self.lpips_loss = LPIPS(net='alex').to(opts.device).eval()
        self.id_loss = IDLoss().to(opts.device).eval()
        self.w_norm_loss = WNormLoss(start_from_latent_avg=opts.start_from_latent_avg)

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

    def calc_loss(self, x, x_prime, x_hat, x_hat_prime, wx, wx_hat, w_avg, pseudo_factor=0.8):
        loss_dict = {}

        loss_w_norm = 0.0
        loss_w_norm += self.w_norm_loss(wx, w_avg)
        loss_w_norm += self.w_norm_loss(wx_hat, w_avg) * pseudo_factor
        loss_w_norm /= (1 + pseudo_factor)
        loss_dict['loss_w_norm'] = float(loss_w_norm)

        loss_w_plus = (1 - torch.cosine_similarity(wx, wx_hat, dim=2).mean())
        loss_dict['loss_w_plus'] = float(loss_w_plus)

        loss_id = self.id_loss.forward(x, x_prime)
        loss_dict['loss_id'] = float(loss_id)

        loss_id_rand = self.id_loss.forward(x, x_hat)
        loss_dict['loss_id_rand'] = float(loss_id_rand)

        loss_l1 = F.l1_loss(x, x_prime) + (F.l1_loss(x, x_hat_prime) + F.l1_loss(x_prime, x_hat_prime)) * pseudo_factor
        loss_dict['loss_l1'] = float(loss_l1)

        loss_lpips = self.lpips_loss(x, x_prime) + self.lpips_loss(x, x_hat_prime) * pseudo_factor
        loss_dict['loss_lpips'] = float(loss_lpips)

        loss = loss_w_norm * 0.005 + loss_w_plus + loss_id + loss_id_rand + loss_l1 + loss_lpips
        loss_dict['loss'] = float(loss)

        return loss, loss_dict


def make_experiment_dir(opts):
    opts.exp_dir = os.path.join('experiments', opts.exp_dir)
    os.makedirs(opts.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.exp_dir, 'sample'), exist_ok=True)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)


def configure_optimizers(model, opts):
    params = model.encoder.parameters()
    optimizer = Ranger(params, lr=opts.learning_rate)
    return optimizer


def configure_dataset(opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                  target_root=dataset_args['train_target_root'],
                                  opts=opts)
    return train_dataset


def gen_avg_latent(opts, net):
    intrinsics = FOV_to_intrinsics(18.837, device=opts.device)
    cam_pivot = torch.tensor(net.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=opts.device)
    cam_radius = net.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=opts.device)
    constant_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    num_samples = 10000
    z_samples = np.random.RandomState(123).randn(num_samples, 512)
    w_samples = net.decoder.mapping(torch.from_numpy(z_samples).to(opts.device),
                                         constant_params.repeat([num_samples, 1]), truncation_psi=0.7,
                                         truncation_cutoff=14)
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_avg = np.repeat(w_avg, 14, axis=1)
    w_avg = torch.tensor(w_avg).to(opts.device)

    return w_avg


def print_metrics(metrics_dict, prefix, global_step):
    print(f'Metrics for {prefix}, step {global_step}')
    for key, value in metrics_dict.items():
        print(f'\t{key} = ', value)


def __get_save_dict(net):
    save_dict = {
        'state_dict': net.state_dict(),
    }
    return save_dict


def checkpoint_me(loss_dict, global_step, net, opts):
    save_name = f'iteration_{global_step}.pt'
    save_dict = __get_save_dict(net)
    checkpoint_path = os.path.join(opts.checkpoint_dir, save_name)
    torch.save(save_dict, checkpoint_path)
    with open(os.path.join(opts.checkpoint_dir, 'timestamp.txt'), 'a') as f:
        f.write(f'Step - {global_step}, \n{loss_dict}\n')


def store(imgs, opts, name='test'):
    img = torch.cat(imgs, 3)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(opts.exp_dir, 'sample', name + '.png'))


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    opts = TrainOptions().parse()
    make_experiment_dir(opts)
    device_id = rank % torch.cuda.device_count()
    opts.device = f'cuda:{device_id}'
    opts.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    model = pSp(opts, train_faceswap=False).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    criterion = Criterion(opts).to(device_id)

    optimizer = configure_optimizers(ddp_model.module, opts)
    train_set = configure_dataset(opts)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=opts.batch_size, sampler=train_sampler)

    w_avg = gen_avg_latent(opts, ddp_model.module)

    ddp_model.module.train()
    global_step = 0
    while global_step < opts.max_steps:
        for batch_idx, batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            x, y, cp, _, _ = batch
            x, y = x.to(opts.device).float(), y.to(opts.device).float()
            cp = cp.to(opts.device)

            angle_x = torch.rand(1) - 1 / 2
            angle_y = torch.rand(1) - 1 / 2
            cam_pivot = torch.tensor(ddp_model.module.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=opts.device)
            cam_radius = ddp_model.module.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
            intrinsics = FOV_to_intrinsics(18.837, device=opts.device)

            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y.to(opts.device),
                                                      np.pi / 2 + angle_x.to(opts.device), cam_pivot, radius=cam_radius,
                                                      device=opts.device)
            rand_cp = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(opts.device)
            rand_cp = rand_cp.unsqueeze(dim=0).repeat([opts.batch_size, 1, 1])

            x_prime, x_hat, x_hat_prime, wx, wx_hat = ddp_model.module.Inversion(x, cp, rand_cp, w_avg)
            loss, loss_dict = criterion.calc_loss(x, x_prime, x_hat, x_hat_prime, wx, wx_hat, w_avg)

            loss.backward()
            optimizer.step()

            if rank == 0:
                if global_step % opts.board_interval == 0:
                    print_metrics(loss_dict, prefix='train', global_step=global_step)

                if global_step % opts.save_interval == 0 or global_step == opts.max_steps:
                    checkpoint_me(loss_dict, global_step, ddp_model.module, opts)

                if global_step % 500 == 0:
                    store([x, x_prime, x_hat, x_hat_prime], opts, str(global_step))

                global_step += torch.cuda.device_count() * opts.batch_size


if __name__ == '__main__':
    main()
