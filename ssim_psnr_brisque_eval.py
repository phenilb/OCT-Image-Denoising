# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:49:22 2022

@author: Phenil Buch
"""

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import imquality.brisque as brisque
import PIL.Image

import cv2

import os
from os import listdir

from tqdm import tqdm

from statistics import mean, pstdev

# get the path/directory

folder_dir = "./images_for_fid_eval/"
folder_original = "./actual_test_images_for_ssim_psnr/"
# for im in os.listdir(folder_dir):
#     image = cv2.imread(os.path.join(folder_dir, im))
#     for orig_image in os.listdir(folder_original):
#         orig_image = cv2.imread(os.path.join(folder_original, orig_image))
#         s = ssim(image, orig_image, multichannel=True)
#         p = psnr(image, orig_image)
#         print("SSIM: ", s)
#         print("PSNR: ", p)

#=============================================================================

ssim_list = []
psnr_list = []
brisque_list_fake = []
brisque_list_real = []

number_of_images = len([file for file in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file))])
print("number_of_images: ", number_of_images)

for im, orig_im in tqdm(zip(os.listdir(folder_dir), os.listdir(folder_original)), total=number_of_images):
	image = cv2.imread(os.path.join(folder_dir, im), 0)
	orig_image = cv2.imread(os.path.join(folder_original, orig_im), 0)
	#cv2.imwrite("Fake_LN.png",image)
	#cv2.imwrite("Real_LN.png",orig_image)
	s = ssim(image, orig_image)
	p = psnr(image, orig_image)
	b = brisque.score(image)
	b_r = brisque.score(orig_image)
	print("SSIM: ", s)
	print("PSNR: ", p)
	print("BRISQUE FAKE: ", b)
	print("BRISQUE REAL: ", b_r)
	ssim_list.append(s)
	psnr_list.append(p)
	brisque_list_fake.append(b)
	brisque_list_real.append(b_r)

mean_ssim = mean(ssim_list)
mean_psnr = mean(psnr_list)
mean_brisque_fake = mean(brisque_list_fake)
mean_brisque_real = mean(brisque_list_real)
stddev_ssim = pstdev(ssim_list)
stddev_psnr = pstdev(psnr_list)
stddev_brisque_fake = pstdev(brisque_list_fake)
stddev_brisque_real = pstdev(brisque_list_real)

print("\n")
print("\n")
print("MEAN_SSIM: ", mean_ssim)
print("MEAN_PSNR: ", mean_psnr)
print("MEAN_BRISQUE_REAL: ", mean_brisque_real)
print("MEAN_BRISQUE_FAKE: ", mean_brisque_fake)
print("STDDEV_SSIM: ", stddev_ssim)
print("STDDEV_PSNR: ", stddev_psnr)
print("STDDEV_BRISQUE_REAL: ", stddev_brisque_real)
print("STDDEV_BRISQUE_FAKE: ", stddev_brisque_fake)


# converted = 'converted.png'
# orig = 'orig.png'
# ppb1_path = 'ppb3.jpg'
# ppb2_path = 'ppb2.jpg'


# cvtd = cv2.imread(converted, 0)
# orig = cv2.imread(orig, 0)
# print("Image Shape: ")
# print(cvtd.shape)
# print("SSIM: ")
# print(ssim(cvtd,orig))
# print("PSNR: ")
# print(psnr(cvtd,orig))
# #cvtd = cv2.imread(converted, 1)
# #cvtd = cvtd[:, :, ::-1]
# print("BRISQUE: ")
# print(brisque.score(cvtd))
# ppb1 = cv2.imread(ppb1_path, 0)
# ppb2 = cv2.imread(ppb2_path, 0)
# print("BRISQUE PPB1: ")
# print(brisque.score(ppb1))
# print("BRISQUE PPB2: ")
# print(brisque.score(ppb2))