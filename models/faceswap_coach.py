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
from models.networks import define_mlp
from PIL import Image
import torchvision.transforms as transforms
from models.id_loss import IDLoss
import sys
import os.path as osp
import mrcfile, os

sys.path.append(".")
sys.path.append("..")

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = TRANSFORM(image).unsqueeze(0)
    return image.to(device)


def load_parameter(param_path, device):
    parameter = torch.zeros([1, 25], device=device)
    parameter_np = np.load(param_path)
    for i in range(parameter_np.__len__()):
        parameter[0, i] += parameter_np[i]
    return parameter


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def acti(w):
    return 1 / (1 + torch.exp(-10 * (w - 0.5)))


def gen_mask():
    mask = torch.zeros([1, 512, 512])
    num = 75
    for i in range(35 * 2 + num, 223 * 2 - num):
        for j in range(32 * 2 + num, 220 * 2 - num):
            mask[0][i][j] += 1
    return mask


def configure_optimizers(networks, lr=3e-4):
    params = list(networks.backbone.parameters()) + list(networks.renderer.parameters()) + list(networks.decoder.parameters())
    optimizer = torch.optim.Adam([{'params': params}], lr=lr)
    return optimizer


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples.unsqueeze(0), voxel_origin, voxel_size


class FaceSwapCoach:
    def __init__(self):
        self.device = torch.device('cuda')
        self.l2 = torch.nn.MSELoss(reduction='mean')
        self.lpips = LPIPS(net='alex').to(self.device).eval()
        self.id_loss = IDLoss().to(self.device).eval()
        self.encoder = self.load_encoder()
        self.MLPs = self.load_mlps()
        self.decoder = self.load_decoder()
        self.w_avg = self.gen_w_avg()

    def load_encoder(self):
        encoder = GradualStyleEncoder(50, 'ir_se')
        encoder_ckpt = torch.load('checkpoints/encoder.pt')
        encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
        return encoder

    def load_mlps(self):
        MLPs = []
        mlp_ckpt = torch.load('checkpoints/mlp.pt')
        for i in range(5):
            mlp = define_mlp(4)
            mlp.load_state_dict(get_keys(mlp_ckpt, f'MLP{i}'), strict=True)
            MLPs.append(mlp)
        return MLPs

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

    def inversion(self, x, y):
        with torch.no_grad():
            x_ws = self.encoder(x.cpu()).to(self.device) + self.w_avg
            y_ws = self.encoder(y.cpu()).to(self.device) + self.w_avg
        return x_ws, y_ws

    def latent_interpolation(self, x_ws, y_ws):
        start_layer = 5
        leng = 5

        x_codes, y_codes = [], []
        for i in range(start_layer, start_layer + leng):
            x_codes.append(x_ws[:, i: i + 1])
            y_codes.append(y_ws[:, i: i + 1])

        yhat_codes = [y_ws[:, :start_layer]]
        for i in range(start_layer, start_layer + leng):
            i = i - start_layer
            MLP = self.MLPs[i]
            rho = acti(MLP(torch.cat([x_codes[i], y_codes[i]], dim=2)))
            yhat_codes.append(y_codes[i] * rho + x_codes[i] * (1 - rho))

        yhat_codes.append(y_ws[:, start_layer + leng:])
        ws = torch.cat(yhat_codes, dim=1)

        return ws

    def synthesis_inversion(self, x_ws, y_ws, in_cp, out_cp):
        x_rec = self.decoder.synthesis(x_ws, in_cp)['image']
        y_rec = self.decoder.synthesis(y_ws, out_cp)['image']
        return x_rec, y_rec

    def synthesis_faceswap(self, ws, in_cp, out_cp):
        y_hat_out, y_hat_in = self.decoder.synthesis(ws, out_cp)['image'], self.decoder.synthesis(ws, in_cp)['image']
        return y_hat_out, y_hat_in

    def cal_loss_rec(self, x, y, x_rec, y_rec):
        x_rec = F.interpolate(x_rec, size=[256, 256], mode='bilinear', align_corners=True)
        y_rec = F.interpolate(y_rec, size=[256, 256], mode='bilinear', align_corners=True)

        y_factor = 1.0

        loss_l2 = self.l2(x, x_rec) + self.l2(y, y_rec) * y_factor
        loss_lpips = self.lpips(x, x_rec) + self.lpips(y, y_rec) * y_factor
        loss_lpips = torch.squeeze(loss_lpips)
        loss_id = self.id_loss.forward(x, x_rec) + self.id_loss.forward(y, y_rec) * y_factor

        loss = loss_l2 * 1.0 + loss_lpips * 1.0 + loss_id * 5.0
        return loss

    def cal_loss_fs(self, x, y, y_hat_out, y_hat_in, mask):
        y_hat_out = F.interpolate(y_hat_out, size=[256, 256], mode='bilinear', align_corners=True)
        y_hat_in = F.interpolate(y_hat_in, size=[256, 256], mode='bilinear', align_corners=True)

        loss_l2 = self.l2(y_hat_out * (1 - mask), y * (1 - mask))
        loss_lpips = torch.squeeze(self.lpips(y_hat_out * (1 - mask), y * (1 - mask)))
        loss_id = self.id_loss.forward(x, y_hat_out) + self.id_loss.forward(y_hat_out, y_hat_in)

        loss = loss_l2 * 10.0 + loss_lpips * 10.0 + loss_id * 0.75

        return loss

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
            from shape_utils import convert_sdf_samples_to_ply
            convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                       os.path.join('output/faceswap/', name + '.ply'), level=10)
        elif shape_format == '.mrc':
            with mrcfile.new_mmap(os.path.join('output/faceswap/', name + '.mrc'), overwrite=True, shape=sigmas.shape,
                                  mrc_mode=2) as mrc:
                mrc.data[:] = sigmas

    def run(self, args):
        in_name, out_name = str(args.from_index), str(args.to_index)
        name = in_name + '_' + out_name

        in_image = load_image(osp.join(args.dataroot, 'final_crops', in_name + '.jpg'), self.device)
        out_image = load_image(osp.join(args.dataroot, 'final_crops', out_name + '.jpg'), self.device)

        in_cp = load_parameter(osp.join(args.dataroot, 'camera_pose', in_name + '.npy'), self.device)
        out_cp = load_parameter(osp.join(args.dataroot, 'camera_pose', out_name + '.npy'), self.device)

        in_img = (in_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out_img = (out_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs = [in_img, out_img]

        mask = F.interpolate(gen_mask().unsqueeze(0), size=[256, 256], mode='bilinear', align_corners=True).cuda()

        x = F.interpolate(in_image, size=[256, 256], mode='bilinear', align_corners=True)
        y = F.interpolate(out_image, size=[256, 256], mode='bilinear', align_corners=True)
        x_ws, y_ws = self.inversion(x, y)

        self.decoder.train()
        optimizer = configure_optimizers(self.decoder, args.lr)

        for _ in tqdm(range(args.epoch)):
            x_rec, y_rec = self.synthesis_inversion(x_ws, y_ws, in_cp, out_cp)
            loss = self.cal_loss_rec(x, y, x_rec, y_rec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ws = None
        optimizer = configure_optimizers(self.decoder, args.lr / 10)

        for _ in tqdm(range(args.epoch)):
            ws = self.latent_interpolation(x_ws, y_ws)
            y_hat_out, y_hat_in = self.synthesis_faceswap(ws, in_cp, out_cp)
            loss = self.cal_loss_fs(x, y, y_hat_out, y_hat_in, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        yhat_out = self.decoder.synthesis(ws, out_cp)['image']
        yhat_in = self.decoder.synthesis(ws, in_cp)['image']

        yhat_out = (yhat_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        yhat_in = (yhat_in.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        imgs.append(yhat_out)
        imgs.append(yhat_in)

        img = torch.cat(imgs, dim=2)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'output/faceswap/' + name + '.png')
        self.gen_shape(ws, name)
