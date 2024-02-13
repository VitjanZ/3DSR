import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np


def generate_perlin_noise(resize_shape, perlin_scale=6, min_perlin_scale=0):
    rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((resize_shape[0], resize_shape[1]), (perlin_scalex, perlin_scaley))
    perlin_noise = rot(image=perlin_noise)
    beta = 0.4
    threshold = torch.rand(1).numpy()[0] * beta + beta
    perlin_thr = np.where(np.abs(perlin_noise) > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    norm_perlin = np.where(np.abs(perlin_noise) > threshold, perlin_noise, np.zeros_like(perlin_noise))
    return norm_perlin, perlin_thr, perlin_noise, threshold

class DADADataset(Dataset):

    def __init__(self, imagenet_path, resize_shape=(256,256), depth_only=False, bs=8):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.depth_only = depth_only
        self.inet_images = [None]
        if not self.depth_only:
            self.inet_images = sorted(glob.glob(imagenet_path+"*.JPEG"))

        self.resize_shape=resize_shape
        self.bs = bs

    def __len__(self):
        return self.bs*1000


    def transform_image(self, rgb_img_path):
        # Generates a perlin noise map
        perlin_norm, perlin_thr, perlin_noise, p_thr = generate_perlin_noise(self.resize_shape)
        beta = torch.rand(1).item()
        beta2 = torch.rand(1).item()
        pmin = np.min(perlin_noise)
        pmax = np.max(perlin_noise)
        # Scales the noise from 0 to 1
        perlin_noise = (perlin_noise - pmin) / (pmax-pmin) # from 0 to 1
        image = perlin_noise
        image = beta * perlin_noise # Scales the depth from 0 to beta
        image = image + (beta2*(1-beta)) # Translates from beta2*(1-beta) to 1.0
        image = np.clip(image, 0.0, 1.0)
        image = np.expand_dims(image,2)
        image = np.transpose(image, (2, 0, 1))

        rgb_image = None
        if not self.depth_only:
            rgb_image = cv2.imread(rgb_img_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (self.resize_shape[1], self.resize_shape[0])).astype(np.float32) / 255.0
            rgb_image = np.transpose(rgb_image, (2, 0, 1))
        return image, rgb_image

    def __getitem__(self, idx):
        idx_inet = torch.randint(0, len(self.inet_images), (1,)).item()
        rgb_img_path = self.inet_images[idx_inet]
        image, rgb_image = self.transform_image(rgb_img_path)
        sample = {'image': image}
        if not self.depth_only:
            sample['rgb_image']=rgb_image
        return sample
