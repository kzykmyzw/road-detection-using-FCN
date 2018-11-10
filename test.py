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

import cv2

from models import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='cityscapes', help='name of the dataset')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--model', type=str, default='', help='path to the model')
opt = parser.parse_args()
print(opt)

os.makedirs('results/%s' % opt.dataset_name, exist_ok=True)

# Load model
encdec = encdec()

# Use CUDA if available
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
if use_cuda:
    encdec = encdec.cuda()

encdec.load_state_dict(torch.load(opt.model))

# Configure dataloaders
transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

val_dataloader = DataLoader(ImageDataset('datasets/%s' % opt.dataset_name, transforms_=transforms_, mode='val'),
                            batch_size=1, shuffle=False, num_workers=1)

# Tensor type
Tensor = torch.FloatTensor
if use_cuda:
    Tensor = torch.cuda.FloatTensor

# Test
average_inference_time = 0
for i, batch in enumerate(val_dataloader):

    # Input and ground truth
    input = Variable(batch['A'].type(Tensor))
    gtruth = Variable(batch['B'].type(Tensor))

    # Inference
    prev_time = time.time()
    result = encdec(input)
    average_inference_time += time.time() - prev_time

    # Print logs
    sys.stdout.write('\r[File %d/%d] Average Inference Time: %f msec' %
                     (i, len(val_dataloader),
                      average_inference_time / (i + 1) * 1000))

    # Save result image
    save_image(result, 'results/%s/result_%s.%s' % (opt.dataset_name, '{0:04d}'.format(i), 'png'), normalize=True)

    # Save synthesized image
    original = cv2.imread('datasets/%s/%s/%s.%s' % (opt.dataset_name, 'val', '{0:04d}'.format(i), 'png'))
    input = cv2.resize(original[:, :original.shape[1]//2, :], (opt.img_width, opt.img_height))
    result = cv2.imread('results/%s/result_%s.%s' % (opt.dataset_name, '{0:04d}'.format(i), 'png'))
    mask = np.where(result[:, :, 0] > 128, 1, 0).astype(np.float)
    input[:, :, 0] = 0.1 * mask * input[:, :, 0] + 0.5 * (1 - mask) * input[:, :, 0]
    input[:, :, 1] = 1.0 * mask * input[:, :, 1] + 0.5 * (1 - mask) * input[:, :, 1]
    input[:, :, 2] = 1.0 * mask * input[:, :, 2] + 0.5 * (1 - mask) * input[:, :, 2]
    cv2.imwrite('results/%s/synthesized_%s.%s' % (opt.dataset_name, '{0:04d}'.format(i), 'png'), input)