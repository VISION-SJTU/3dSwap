import os
from models.faceswap_coach import FaceSwapCoach
import argparse

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run_3dSwap(args):
    coach = FaceSwapCoach()
    coach.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_index', type=int, default=0)
    parser.add_argument('--to_index', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dataroot', type=str, default='datasets/CelebA-HD')
    args = parser.parse_args()

    run_3dSwap(args)
