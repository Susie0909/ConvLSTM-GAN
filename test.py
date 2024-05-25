import torch
import torch.nn as nn

def downsample(in_feat, out_feat, normalize=True, dropout=True):
    layers = [nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2))
    if dropout:
        layers.append(nn.Dropout2d(0.5))
    return layers

def upsample(in_feat, out_feat, normalize=True, dropout=True):
    layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=2, padding=1, output_padding=1)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_feat, 0.8))
    if dropout:
        layers.append(nn.Dropout2d(0.5))
    layers.append(nn.ReLU())
    return layers

model = nn.Sequential(
    *downsample(16, 16, normalize=False), # ->300
    *downsample(16, 32), # ->150
    *downsample(32, 32), # ->75
    nn.Conv2d(32, 64, 1), # -> 75
    *upsample(64, 32), # -> 8
    *upsample(32, 16), # -> 16
    *upsample(16, 8), # -> 32
    nn.Conv2d(8, 1, 3, 1, 1), # -> 128
    nn.Tanh()
)


x = torch.rand(1, 16, 600, 600)
breakpoint()
