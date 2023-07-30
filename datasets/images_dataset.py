import os.path
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import random
import numpy as np
import torch
import torchvision.transforms as transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)


def gen_mask(p):
    if p is None:
        mask_rect = torch.zeros([1, 512, 512])
        num = 75
        for i in range(50, 100):
            for j in range(32 * 2 + num, 220 * 2 - num):
                mask_rect[0][i][j] += 1
        return mask_rect
    index = [1]
    mask = torch.zeros([1, 512, 512])
    for i in index:
        mask += p == i
    mask_rect = torch.zeros([1, 512, 512])
    num = 75
    for i in range(35 * 2 + num, 223 * 2 - num):
        for j in range(32 * 2 + num, 220 * 2 - num):
            mask_rect[0][i][j] += 1

    return mask * mask_rect


def load_parameter(param_path):
    parameter = torch.zeros([1, 25])
    parameter_np = np.load(param_path)
    for i in range(parameter_np.__len__()):
        parameter[0, i] += parameter_np[i]
    return parameter


class ImagesDataset(Dataset):

    def __init__(self, source_root, target_root, opts):
        self.camera_pose_root = source_root[:-11] + 'camera_pose'

        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.target_num = len(self.source_paths)
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        to_index = (random.randint(0, self.target_num)) % self.target_num

        from_im = Image.open(self.source_paths[index]).convert('RGB')
        from_im = TRANSFORM(from_im)

        to_im = Image.open(self.source_paths[to_index]).convert('RGB')
        to_im = TRANSFORM(to_im)

        from_camera_parameter = load_parameter(
            os.path.join(self.camera_pose_root, self.source_paths[index].split('/')[-1].split('.')[0] + '.npy'))
        to_camera_parameter = load_parameter(
            os.path.join(self.camera_pose_root, self.source_paths[to_index].split('/')[-1].split('.')[0] + '.npy'))

        try:
            to_label_path = os.path.join('datasets/EG3D/labels',
                                         self.source_paths[to_index].split('/')[-1].split('.')[0] + '.png')
            to_label = Image.open(to_label_path).convert('L')
            to_label_tensor = TRANSFORM(to_label) * 255.0
            to_face_mask = gen_mask(to_label_tensor)
        except:
            to_face_mask = gen_mask(None)

        return from_im, to_im, from_camera_parameter, to_camera_parameter, to_face_mask
