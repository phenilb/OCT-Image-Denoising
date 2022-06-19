# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:50:12 2022

@author: Phenil Buch
"""

# =============================================================================
# from PIL import Image
# import cv2
# import numpy as np
# import torchvision
# from config import testBatchSize, test_size

# Imports For Saving Sample Images

# =============================================================================

from dataset import OCTQualityDataset
from torch.utils.data import DataLoader
import torch
from config import parent_folder, lr, n_epochs, decay_epoch, device, size, batchSize
from torchvision import transforms
from torch.autograd import Variable
from model import Generator
from model import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from initialize_weights import weight_init
import gc # garbage collector
from GPUtil import showUtilization as gpu_usage #need to do pip install for this package


from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

#print("GPU usage before cuda empty cache")
#gpu_usage()
#torch.cuda.empty_cache()
#print("GPU usage after cuda empty cache")
gpu_usage()

writer = SummaryWriter('./runs/cyclegan_med_denoise_experiment_1')

PATH = './weights/checkpoint.pth'
PATH_PREV = './previous_weights/'

netG_HN2LN = Generator(1,1) #input_channels, output_channels
netD_LN = Discriminator(1) #output_channels

netG_HN2LN.to(device)
netD_LN.to(device)

netG_HN2LN.apply(weight_init)
netD_LN.apply(weight_init)


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_HN2LN.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_LN = torch.optim.Adam(netD_LN.parameters(), lr=lr, betas=(0.5, 0.999))


Tensor = torch.cuda.FloatTensor if device=="cuda" else torch.Tensor
#Tensor = torch.Tensor
input_HN = Tensor(batchSize, 1, size, size) #input_channels
input_LN = Tensor(batchSize, 1, size, size) #output_channels
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False).to(device)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False).to(device)

fake_LN_buffer = ReplayBuffer()
same_LN_buffer = ReplayBuffer()

# Dataset loader
transforms_ = transforms.Compose([
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                #The above line is used in case of RGB images
                #ours is greyscale
                transforms.Normalize(0.5, 0.5)
                ])
dataloader = DataLoader(OCTQualityDataset(parent_folder=parent_folder, transformation=transforms_), batch_size=batchSize, shuffle=True)


# To Save Sample Images
# =============================================================================
# for i, batch in enumerate(dataloader):
#     real_HN = Variable(input_HN.copy_(batch['HN'][0]))
#     real_LN = Variable(input_LN.copy_(batch['LN'][0]))
#     hn_grid = torchvision.utils.make_grid(real_HN)
#     ln_grid = torchvision.utils.make_grid(real_LN)
#     hn_img = torchvision.transforms.ToPILImage()(hn_grid)
#     ln_img = torchvision.transforms.ToPILImage()(ln_grid)
#     display_image = Image.new('RGB', (hn_img.width, hn_img.height + ln_img.height))
#     display_image.paste(hn_img, (0, 0))
#     display_image.paste(ln_img, (0, hn_img.height))
#     open_cv_image = np.array(display_image)  
#     open_cv_image = open_cv_image[:, :, ::-1].copy()
#     imageText = open_cv_image.copy()
#     imageText = cv2.cvtColor(imageText, cv2.COLOR_BGR2RGB)
#     text = "High Noise"
#     text2 = "Low Noise"
#     fontScale = 2.3
#     fontFace = cv2.FONT_HERSHEY_PLAIN
#     fontColor = (255, 255, 255)
#     fontThickness = 3
#     cv2.putText(imageText, text, (20, 40), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)
#     cv2.putText(imageText, text2, (20, hn_img.height + 40), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)
#     final_image = Image.fromarray(imageText)
#     #final_image.show()
#     final_image.save(f"HighNoiseLowNoiseDatasetSample_{i}.jpg".format(i))
# =============================================================================


try:
    checkpoint = torch.load(PATH)
    
    EPOCH = checkpoint['epoch']
    print("Checkpoint completed Epochs: ", EPOCH)
    netG_HN2LN.load_state_dict(checkpoint['netG_HN2LN_state_dict'])
    netD_LN.load_state_dict(checkpoint['netD_LN_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D_LN.load_state_dict(checkpoint['optimizer_D_LN_state_dict'])
    # Loss plot
    logger = Logger(n_epochs-EPOCH, len(dataloader))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, EPOCH, decay_epoch).step) #EPOCH = Starting Epoch
    lr_scheduler_D_LN = torch.optim.lr_scheduler.LambdaLR(optimizer_D_LN, lr_lambda=LambdaLR(n_epochs, EPOCH, decay_epoch).step)
except:
    EPOCH = 0
    print("Checkpoint completed Epochs: ", EPOCH)
    # Loss plot
    logger = Logger(n_epochs, len(dataloader))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, 1, decay_epoch).step) #Starting Epoch
    lr_scheduler_D_LN = torch.optim.lr_scheduler.LambdaLR(optimizer_D_LN, lr_lambda=LambdaLR(n_epochs, 1, decay_epoch).step)

disc_iters = 1

###################################

###### Training ######
for epoch in range(EPOCH+1, n_epochs+1):
    print("Current Epoch: ", epoch)
    for i, batch in enumerate(dataloader):
        # Set model input
        real_HN = Variable(input_HN.copy_(batch['HN']))
        real_LN = Variable(input_LN.copy_(batch['LN']))
        real_HN.to(device)
        real_LN.to(device)

        ###################################

        ###### Generator ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed

        same_LN = netG_HN2LN(real_LN)
        fake_LN = netG_HN2LN(real_HN)

        loss_identity_LN = criterion_identity(same_LN, real_LN)*5.0
        
        # GAN loss
        pred_fake = netD_LN(fake_LN)
        loss_GAN_HN2LN_fake = criterion_GAN(pred_fake, target_real)
        pred_fake_same = netD_LN(same_LN)
        loss_GAN_HN2LN_same = criterion_GAN(pred_fake_same, target_real)
        loss_GAN_HN2LN = (loss_GAN_HN2LN_same + loss_GAN_HN2LN_fake)*0.5
    
        # Total loss
        loss_G = loss_identity_LN + loss_GAN_HN2LN
        loss_G.backward()
        
        optimizer_G.step()
        
        ###### Discriminator ######
        
        optimizer_D_LN.zero_grad()
        
        for _ in range(disc_iters):
            # Real loss
            pred_real = netD_LN(real_LN)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_LN = fake_LN_buffer.push_and_pop(fake_LN)
            pred_fake = netD_LN(fake_LN.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Fake (Same) loss
            same_LN = same_LN_buffer.push_and_pop(same_LN)
            pred_fake_same = netD_LN(same_LN.detach())
            loss_D_fake_same = criterion_GAN(pred_fake_same, target_fake)        

            # Total loss
            D_total_fake_loss = (loss_D_fake + loss_D_fake_same)*0.5
            loss_D_LN = loss_D_real + D_total_fake_loss
            loss_D_LN.backward()

            optimizer_D_LN.step()
        ###################################

        
        ####################################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': loss_identity_LN, 'loss_G_GAN': loss_GAN_HN2LN, 'loss_D': loss_D_LN}, 
                    images={'real_A': real_HN, 'real_B': real_LN, 'fake_B': fake_LN})
        
        writer.add_scalars('Training Loss',
                            {'loss_G': loss_G, 'loss_G_identity': loss_identity_LN, 'loss_G_GAN': loss_GAN_HN2LN, 'loss_D': loss_D_LN},
                            (epoch-1) * len(dataloader) + i)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_LN.step()
    gc.collect()

    # Save models checkpoints
    # There already needs to be a weights directory created to do this
    
    torch.save({
            'epoch': epoch,
            'netG_HN2LN_state_dict': netG_HN2LN.state_dict(),
            'netD_LN_state_dict': netD_LN.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_LN_state_dict': optimizer_D_LN.state_dict(),
            }, PATH)
    PATH_PREV_W = PATH_PREV + 'checkpoint_epoch_' + str(epoch) + '.pth'
    torch.save({
        'epoch': epoch,
        'netG_HN2LN_state_dict': netG_HN2LN.state_dict(),
        'netD_LN_state_dict': netD_LN.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_LN_state_dict': optimizer_D_LN.state_dict(),
        }, PATH_PREV_W)
    
writer.flush()
    