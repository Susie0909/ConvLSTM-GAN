"""
edited by @Susie0909 2024.05.19
"""

import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import *
from models import *
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="/Users/shuyi/dataset/rockCT_tensor", help="root dir of img dataset")
parser.add_argument("--out_dir", type=str, default="/Users/shuyi/dataset/rockCT_train_out", help="out dir to save the generated image")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")

parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")

parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
# patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch_h, patch_w = int(opt.img_size / 2 ** 4), int(opt.img_size / 2 ** 4)
patch = (1, patch_h, patch_w)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()


# Initialize generator and discriminator
seq_len = 6
generator = Generator(output_dim=1, input_dim=1, hidden_dim=[16, 16, 16], kernel_size=(3, 3), num_layers=3)
discriminator = Discriminator(output_dim=1, input_dim=1, hidden_dim=[16, 16, 16, 8], kernel_size=(3, 3), num_layers=4)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
    transforms.Resize(opt.img_size),
    transforms.Normalize((0.5,) * seq_len, (0.5,) * seq_len)  # (x-mean) / std
]

dataloader = DataLoader(
    ImageDataset(opt.root_dir, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_sample(batches_done):
    samples = next(iter(dataloader))
    samples = Variable(samples.type(Tensor))
    
    # Generate inpainted image
    gen_img = generator(samples[:, :-1, ...])
    gen_imgs = torch.cat([samples[:, :-1, ...], gen_img], dim = 1)

    # Save sample
    seq_len = samples.size()[1]
    for seq_i in range(seq_len):
        save_image(samples[:25, seq_i, :, :], "%s/%d_ori_%d.png" % (opt.out_dir, batches_done, seq_i), nrow=5, normalize=True)
    save_image(gen_imgs[:25, -1, :, :], "%s/%d_gen_%d.png" % (opt.out_dir, batches_done, seq_len - 1), nrow=5, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        
        # gen_imgs = generator(imgs)
        gen_img = generator(imgs[:, :-1, ...]) # [bs, 1, h, w]
        gen_imgs = torch.cat([imgs[:, :-1, ...], gen_img], dim=1)
        # Adversarial loss
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)
        
        if batches_done % (10 * opt.sample_interval) == 0:
            torch.save(generator.state_dict(), "%s/%d_gen.pth" % (opt.out_dir, batches_done))
            torch.save(discriminator.state_dict(), "%s/%d_adv.pth" % (opt.out_dir, batches_done))

torch.save(generator.state_dict(), "%s/%d.pth" % (opt.out_dir, batches_done))
