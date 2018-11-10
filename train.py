import argparse
import os
import numpy as np
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from models import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default='cityscapes', help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Loss function
criterion_pixelwise = torch.nn.L1Loss()

# Initialize encdec
encdec = encdec()

# Use CUDA if available
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
if use_cuda:
    encdec = encdec.cuda()
    criterion_pixelwise.cuda()

encdec.apply(weights_init)

# Optimizers
optimizer = torch.optim.Adam(encdec.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

dataloader = DataLoader(ImageDataset('datasets/%s' % opt.dataset_name, transforms_=transforms_),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = DataLoader(ImageDataset('datasets/%s' % opt.dataset_name, transforms_=transforms_, mode='val'),
                            batch_size=10, shuffle=True, num_workers=1)

# Tensor type
Tensor = torch.FloatTensor
if use_cuda:
    Tensor = torch.cuda.FloatTensor

# Save result images during training
def sample_images(batches_done):
    imgs = next(iter(val_dataloader))
    input = Variable(imgs['A'].type(Tensor))
    gtruth = Variable(imgs['B'].type(Tensor))
    result = encdec(input)
    img_sample = torch.cat((input.data, result.data, gtruth.data), -2)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)

# Training
prev_time = time.time()
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Input and ground truth
        input = Variable(batch['A'].type(Tensor))
        gtruth = Variable(batch['B'].type(Tensor))

        optimizer.zero_grad()

        # Loss
        result = encdec(input)
        loss = criterion_pixelwise(result, gtruth)
        loss.backward()

        optimizer.step()

        # Estimate remaining time
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print logs
        sys.stdout.write('\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s' %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss.item(),
                                                        time_left))

        # Save result images
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model
        torch.save(encdec.state_dict(), 'saved_models/%s/encdec_%d.pth' % (opt.dataset_name, epoch))
