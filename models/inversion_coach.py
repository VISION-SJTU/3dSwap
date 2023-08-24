from models.encoders.psp_encoders import GradualStyleEncoder
import torch
import dnnlib
from utils import legacy
from utils.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
import numpy as np
import PIL.Image
import torch.nn.functional as F
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from torch_utils import misc
from lpips import LPIPS
import torchvision.transforms as transforms
import sys
import os.path as osp
import mrcfile, os
from models.faceswap_coach import get_keys, load_image, load_parameter, configure_optimizers, create_samples

sys.path.append(".")
sys.path.append("..")

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class InversionCoach:
    def __init__(self):
        self.device = torch.device('cuda')
        self.l2 = torch.nn.MSELoss(reduction='mean')
        self.lpips = LPIPS(net='alex').to(self.device).eval()
        self.encoder = self.load_encoder()
        self.decoder = self.load_decoder()
        self.w_avg = self.gen_w_avg()

    def load_encoder(self):
        encoder = GradualStyleEncoder(50, 'ir_se')
        encoder_ckpt = torch.load('checkpoints/encoder.pt')
        encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
        return encoder

    def load_decoder(self):
        network_pkl = 'checkpoints/ffhq512-128.pkl'
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(self.device)
            misc.copy_params_and_buffers(G, G_new, require_all=True)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G_new.rendering_kwargs = G.rendering_kwargs
            G = G_new.requires_grad_(True)
        return G

    def gen_w_avg(self):
        intrinsics = FOV_to_intrinsics(18.837, device=self.device)
        cam_pivot = torch.tensor(self.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
        cam_radius = self.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=self.device)
        constant_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        num_samples = 10000
        z_samples = np.random.RandomState(123).randn(num_samples, 512)
        w_samples = self.decoder.mapping(torch.from_numpy(z_samples).to(self.device), constant_params.repeat([num_samples, 1]), truncation_psi=0.7, truncation_cutoff=14)
        w_samples = w_samples[:, :1, :].cpu().detach().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
        w_avg = np.repeat(w_avg, 14, axis=1)
        w_avg = torch.tensor(w_avg).to(self.device)
        return w_avg

    def calc_loss(self, generated_images, real_images):
        loss_l2 = self.l2(generated_images, real_images)
        loss_lpips = self.lpips(generated_images, real_images)
        loss_lpips = torch.squeeze(loss_lpips)

        loss = loss_l2 + loss_lpips

        return loss, loss_l2, loss_lpips

    def gen_shape(self, ws, name, shape_res=512, shape_format='.ply'):
        max_batch = 1000000

        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                                           cube_length=self.decoder.rendering_kwargs['box_warp'] * 1)
        samples = samples.to(self.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=self.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=self.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total=samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.decoder.sample_ws(samples[:, head:head + max_batch],
                                                   transformed_ray_directions_expanded[:, :samples.shape[1] - head], ws,
                                                   noise_mode='const')['sigma']
                    sigmas[:, head:head + max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        pad = int(30 * shape_res / 256)
        pad_value = -1000
        sigmas[:pad] = pad_value
        sigmas[-pad:] = pad_value
        sigmas[:, :pad] = pad_value
        sigmas[:, -pad:] = pad_value
        sigmas[:, :, :pad] = pad_value
        sigmas[:, :, -pad:] = pad_value

        if shape_format == '.ply':
            from utils.shape_utils import convert_sdf_samples_to_ply
            convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                       os.path.join('output/inversion/', name + '.ply'), level=10)
        elif shape_format == '.mrc':
            with mrcfile.new_mmap(os.path.join('output/inversion/', name + '.mrc'), overwrite=True, shape=sigmas.shape,
                                  mrc_mode=2) as mrc:
                mrc.data[:] = sigmas

    def run(self, args):
        name = str(args.index)
        image = load_image(osp.join(args.dataroot, 'final_crops', name + '.jpg'), self.device)
        camera_pose = load_parameter(osp.join(args.dataroot, 'camera_pose', name + '.npy'), self.device)

        img = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs = [img]

        x = F.interpolate(image, size=[256, 256], mode='bilinear', align_corners=True)
        with torch.no_grad():
            w_pivot = self.encoder(x.cpu()).to(self.device) + self.w_avg

        real_images = image.to(self.device)
        LPIPS_value_threshold = 0.06

        self.decoder.train()
        optimizer = configure_optimizers(self.decoder, args.lr)

        for _ in tqdm(range(args.epoch)):
            generated_images = self.decoder.synthesis(w_pivot, camera_pose)['image']
            loss, loss_l2, loss_lpips = self.calc_loss(generated_images, real_images)
            optimizer.zero_grad()
            if loss_lpips <= LPIPS_value_threshold:
                break
            loss.backward()
            optimizer.step()

        img = self.decoder.synthesis(w_pivot, camera_pose)['image']
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)
        angle_p = -0.2
        intrinsics = FOV_to_intrinsics(18.837, device=self.device)
        for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
            cam_pivot = torch.tensor(self.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
            cam_radius = self.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot,
                                                      radius=cam_radius, device=self.device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            img = self.decoder.synthesis(w_pivot, camera_params)['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)

        img = torch.cat(imgs, dim=2)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'output/inversion/' + name + '.png')

        self.gen_shape(w_pivot, name)
