# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:59:21 2022

@author: Phenil Buch
"""

from torch.utils.data import Dataset
from os.path import join
import os
from imageio import imread
from PIL import Image
import torch as pt

class OCTQualityDataset(Dataset):

    def __init__(self, parent_folder, fraction=0.8, transformation=lambda x: x):

        self.transformation = transformation

        # get lists of filenames
        self.hn_files = self.gather_filenames(join(parent_folder, 'high-noise'))
        self.ln_files = self.gather_filenames(join(parent_folder, 'low-noise'))
        
        # keep fraction
        #last_index = int(len(self.hn_files) * abs(fraction))
        #self.hn_files = self.hn_files[:last_index] if fraction > 0 else self.hn_files[-last_index:]
        #last_index = int(len(self.ln_files) * abs(fraction))
        #self.ln_files = self.ln_files[:last_index] if fraction > 0 else self.ln_files[-last_index:]

    @staticmethod
    def gather_filenames(folder):

        # walk through directories and collect filenames with path (in numerical order to preserve pairing)
        filenames = []
        for root, dirs, files in os.walk(folder):
            dirs.sort(key=int)
            if not files: continue
            files.sort(key=lambda x: int(x.split('.')[0]))
            filenames += [f'{root}/{f}' for f in files]
        
        return filenames

    def prepare_image(self, filename):

        # load and convert to normalized float32 tensor
        #image = imread(filename) #Using PIL now instead of imread
        image = Image.open(filename)
        image = self.transformation(image)
        #image = pt.from_numpy(image).float() #image is already a tensor because of transforms
        #image = image / image.max() #no need because of transforms
        
        return image

    def __len__(self):
        return min(len(self.hn_files), len(self.ln_files))

    def __getitem__(self, item):

        hn = self.prepare_image(self.hn_files[item])
        ln = self.prepare_image(self.ln_files[item])
        #images = (hn, ln)
        #labels = (1, 2)

        return {'HN': hn, 'LN': ln}
        #return images, labels