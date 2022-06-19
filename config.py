# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:50:49 2022

@author: Phenil Buch
"""

#from model import Generator, Discriminator
#from torch.nn.functional import relu
#from criterion import ClassificationLoss
#from torch.optim import Adam
import torch

seed = 0
#optimizer = Adam
#loss = ClassificationLoss
logdir = f"./training_log/{seed}"
parent_folder = 'oct_quality_original'
test_folder = 'oct_quality_test'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
lr = 0.0005
n_epochs = 50
decay_epoch = 45 # which epoch to start linearly decaying the learning rate to 0
size = 512 # image size # Prev 384
test_size = 512 # image size
batchSize = 2
testBatchSize = 1

# generator = {
#     'scale_factor': 3,
#     'channel_factor': 16,
#     'activation': relu,
#     'kernel_size': (3, 3),
#     'n_residual': (6, 3),
#     'input_channels': 1,
#     'skip_conn': 'concat'
# }

# discriminator = {
#     'n_layers': 7,
#     'kernel_size': (3, 3),
#     'activation': relu,
#     'channel_factor': 16,
#     'max_channels': 1024,
#     'input_channels': 1,
#     'n_residual': (1, 2),
#     'affine': False
# }

# model = {
#     'discriminator': (Discriminator, discriminator),
#     'generator': (Generator, generator),
#     'input_size': (1, 512, 512),
#     'pool_size': 32,
#     'pool_write_probability': 1
# }