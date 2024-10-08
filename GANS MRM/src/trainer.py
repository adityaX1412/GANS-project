import wandb
import argparse
import numpy as np
import math
import torch
import numpy as np
from torch.autograd import Variable
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from utils.dataloader import CelebADataset
from utils.gans import Generator, Discriminator

cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--wd", type=float, default=0.003, help="weight decay for the optimizer")
opt = parser.parse_args(args=[])

dataset_name = "tpremoli/CelebA-attrs"
dataset = load_dataset(dataset_name)

transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.CenterCrop(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

celebA_dataset = CelebADataset(dataset['train'], transform=transform)
dataloader = DataLoader(celebA_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
adversarial_loss = torch.nn.BCELoss()

def context_loss(output, target):
    if output.is_cuda and not target.is_cuda:
        target = target.cuda()
    elif not output.is_cuda and target.is_cuda:
        output = output.cuda()
    return nn.MSELoss()(output, target)

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

wandb.login(key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775")
# Initialize wandb
wandb.init(project="DCGAN-CelebA-Inpainting", config=opt.__dict__)

# Training loop
for epoch in range(opt.n_epochs):
    for i, (masked_imgs, real_imgs) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(masked_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(masked_imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        masked_imgs = Variable(masked_imgs.type(Tensor))
        real_imgs = Variable(real_imgs.type(Tensor))

        # Train Generator
        optimizer_G.zero_grad()

        # Sample noise as generator input (optional, based on specific architecture)
        z = Variable(Tensor(np.random.normal(0, 1, (masked_imgs.size(0), opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid) + context_loss(gen_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Calculate PSNR
        psnr_value = calculate_psnr(gen_imgs, real_imgs)

        # Log losses and PSNR
        wandb.log({"G_loss": g_loss.item(), "D_loss": d_loss.item(), "PSNR": psnr_value.item(), "epoch": epoch})

        # Every few batches, save and log images
        if i % opt.sample_interval == 0:
            # Save generated images
            sample_images = gen_imgs.data[:16]
            wandb.log({"generated_images": [wandb.Image(sample_images, caption=f"Epoch {epoch}, Batch {i}")]})

            # Log model weights and biases
            wandb.log({
                # Generator Weights and Biases
                "Generator Weights (l1.0)": wandb.Histogram(generator.state_dict()['l1.0.weight'].cpu().numpy()),
                "Generator Biases (l1.0)": wandb.Histogram(generator.state_dict()['l1.0.bias'].cpu().numpy()),
                "Generator Weights (conv_blocks.0)": wandb.Histogram(generator.state_dict()['conv_blocks.0.weight'].cpu().numpy()),
                "Generator Biases (conv_blocks.0)": wandb.Histogram(generator.state_dict()['conv_blocks.0.bias'].cpu().numpy()),

                # Discriminator Weights and Biases
                "Discriminator Weights (model.0)": wandb.Histogram(discriminator.state_dict()['model.0.weight'].cpu().numpy()),
                "Discriminator Biases (model.0)": wandb.Histogram(discriminator.state_dict()['model.0.bias'].cpu().numpy()),
                "Discriminator Weights (model.3)": wandb.Histogram(discriminator.state_dict()['model.3.weight'].cpu().numpy()),
                "Discriminator Biases (model.3)": wandb.Histogram(discriminator.state_dict()['model.3.bias'].cpu().numpy()),
            })

# Finish wandb run
wandb.finish()

