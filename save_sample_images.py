# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:23:36 2022

@author: Phenil Buch
"""

from PIL import Image
import cv2
import numpy as np
import torchvision


class SaveSampleImages():
    def __init__(self, dataloader, save_or_not=False):
        dataiter = iter(dataloader)
        for i in range(20):
            images, labels = dataiter.next()
            #images is a tuple of (hn_image, ln_image) which is a thing you must remember
            #print(labels[0]) # this is 1 for all high noise images
            #print(labels[1]) # this is 2 for all low noise images
            hn_grid = torchvision.utils.make_grid(images['HN'])
            ln_grid = torchvision.utils.make_grid(images['LN'])
            hn_img = torchvision.transforms.ToPILImage()(hn_grid)
            ln_img = torchvision.transforms.ToPILImage()(ln_grid)
            display_image = Image.new('RGB', (hn_img.width, hn_img.height + ln_img.height))
            display_image.paste(hn_img, (0, 0))
            display_image.paste(ln_img, (0, hn_img.height))
            open_cv_image = np.array(display_image)  
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            imageText = open_cv_image.copy()
            imageText = cv2.cvtColor(imageText, cv2.COLOR_BGR2RGB)
            text = "High Noise"
            text2 = "Low Noise"
            fontScale = 2.3
            fontFace = cv2.FONT_HERSHEY_PLAIN
            fontColor = (255, 255, 255)
            fontThickness = 3
            cv2.putText(imageText, text, (20, 40), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(imageText, text2, (20, hn_img.height + 40), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)
            final_image = Image.fromarray(imageText)
            if save_or_not:
                final_image.show()
                final_image.save(f"HighNoiseLowNoiseDatasetSample{i}.jpg".format(i))
