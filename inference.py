# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:46:39 2022

@author: Phenil Buch
"""

from tqdm import tqdm
import torchvision
from torchvision.utils import save_image
from PIL import Image
import cv2
import numpy as np
from dataset import OCTQualityDataset
from torch.utils.data import DataLoader
import torch
from config import test_folder, lr, n_epochs, decay_epoch, test_size, testBatchSize, device
from itertools import chain
from torchvision import transforms
from torch.autograd import Variable
from model import Generator
from model import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
#from utils import weights_init_normal
from initialize_weights import weight_init
import gc # garbage collector
from GPUtil import showUtilization as gpu_usage #need to do pip install for this package

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')


torch.cuda.empty_cache()

PATH = './weights/checkpoint.pth'

netG_HN2LN = Generator(1,1)

netG_HN2LN.to(device)

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_HN2LN.parameters(), lr=lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if device=="cuda" else torch.Tensor
#Tensor = torch.Tensor
input_HN = Tensor(testBatchSize, 1, test_size, test_size) #input_channels
input_LN = Tensor(testBatchSize, 1, test_size, test_size) #output_channels

# Dataset loader
transforms_ = transforms.Compose([
                transforms.RandomCrop(test_size),
                transforms.ToTensor(),
                #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                #The above line is used in case of RGB images
                #ours is greyscale
                transforms.Normalize(0.5, 0.5)
                #
                #
                #Not Normalizing for Inference???
                #
                #
                #
                ])

#dataloader = DataLoader(OCTQualityDataset(parent_folder=test_folder, transformation=transforms_), batch_size=batchSize, shuffle=True)
dataloader = DataLoader(OCTQualityDataset(parent_folder=test_folder, transformation=transforms_), batch_size=testBatchSize, shuffle=False)

checkpoint = torch.load(PATH)
    
EPOCH = checkpoint['epoch']
print("Checkpoint completed Epochs: ", EPOCH)
netG_HN2LN.load_state_dict(checkpoint['netG_HN2LN_state_dict'])
optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
logger = Logger(n_epochs-EPOCH, len(dataloader))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, EPOCH, decay_epoch).step) #EPOCH = Starting Epoch


netG_HN2LN.eval()


with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Set model input
        real_HN = Variable(input_HN.copy_(batch['HN']))
        real_LN = Variable(input_LN.copy_(batch['LN']))
        real_HN.to(device)
        real_LN.to(device)
        fake_LN = netG_HN2LN(real_HN)
        for no_imgs in range(len(fake_LN)):
            #print("no_imgs ", no_imgs)
            #print("i", i)
            save_image(fake_LN[no_imgs], f"./images_for_fid_eval/{i+(no_imgs*i)}.jpg".format(i+(no_imgs*i)))
            save_image(real_LN[no_imgs], f"./actual_test_images_for_ssim_psnr/{i+(no_imgs*i)}.jpg".format(i+(no_imgs*i)))

gc.collect()

# Save models checkpoints
# There already needs to be a weights directory created to do this
