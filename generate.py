import argparse
import os
import numpy as np
from tqdm import tqdm
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from dataset import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
# python implementations/context_encoder/generate.py <input_dir> <out_dir> <pth_path>
parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str, default='../../input1', help="root dir for  generating initial data randomly")
parser.add_argument("out_dir", type=str, default='out', help="out dir to save the generated image seq")
parser.add_argument("ckpt", type=str, default='../../ckpt.pth', help="checkpoint of generator")
parser.add_argument("--height", type=int, default=1, help="number of generate images")
parser.add_argument("--ckpt_dir", type=str, help="root dir of ckpts for testing which ckpt is the best")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

seq_len = 6
generator = Generator(output_dim=1, input_dim=1, hidden_dim=[8, 12, 8], kernel_size=(3, 3), num_layers=3)
weights_dict = torch.load(opt.ckpt, map_location='cpu')
generator.load_state_dict(weights_dict)

if cuda:
    generator.cuda()

# Dataset loader
transformsList = transforms.Compose( [
    transforms.Resize(600),
    transforms.Normalize((0.5,) * seq_len, (0.5,) * seq_len)  # (x-mean) / std
])

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
initial_data_path = os.path.join(opt.root_dir, os.listdir(opt.root_dir)[random.randint(0, len(os.listdir(opt.root_dir)))])
imgs = torch.load(initial_data_path)
imgs = imgs.unsqueeze(1)[:-1, :, :, :].unsqueeze(0)
imgs = Variable(imgs.type(Tensor))

# 保存初始的5张图像
for seq_i in range(seq_len - 1):
    save_image(imgs[:25, seq_i, :, :, :], "%s/initial_%d.png" % (opt.out_dir, seq_i), nrow=5, normalize=True)

# 顺序生成
with  torch.no_grad():
    # 在所有的ckpt中测试，仅生成一张图像，用于找到效果最佳的ckpt
    if opt.ckpt_dir:
        ckpts = glob.glob(os.path.join(opt.ckpt_dir, '*.pth'))

        for ckpt in tqdm(ckpts):
            name = ckpt.split('/')[-1].split('.')[0]
            if 'adv' in name:
                continue
            weights_dict = torch.load(ckpt, map_location='cpu')
            generator.load_state_dict(weights_dict)
            gen_img = generator(imgs)
            save_image(gen_img[:25, 0, :, :, :], "%s/%s.png" % (opt.out_dir, name), nrow=5, normalize=True)
    # 使用指定的ckpt生成序列图像
    else:
        for i in tqdm(range(opt.height)):
            gen_img = generator(imgs)
            save_image(gen_img[:25, 0, :, :, :], "%s/gen_%d.png" % (opt.out_dir, i), nrow=5, normalize=True)
            imgs = torch.cat([imgs[:, :-1, ...], gen_img], dim = 1)