import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import tifffile as tif
from geo_utils import *

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None,img_min=0.0, img_max=1.0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.images = sorted(glob.glob(root_dir+"/*.tiff"))
        self.rgb_images = sorted(glob.glob(root_dir+"/../rgb/*.png"))

        self.resize_shape=resize_shape
        self.im_min = img_min
        self.im_max = img_max

    def __len__(self):
        return len(self.images)


    def transform_image(self, image_path, rgb_img_path, mask_path):
        rgb_image = cv2.imread(rgb_img_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        image = tif.imread(image_path).astype(np.float32)

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            #h, w
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]), 0, 0, interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.resize(rgb_image, (self.resize_shape[1], self.resize_shape[0])).astype(np.float32) / 255.0
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image_t = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32) / 255.0

        image = image_t[:, :, 2]

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)

        image = image * plane_mask[:,:,0] # brisi background

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0-zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        image = image * 0.8 + 0.1
        image = image * (1.0 - zero_mask) # 0 are missing pixels, the rest are in [0.1,0.9]
        image = fill_depth_map(image) # fill missing pixels with mean of local valid values

        image = np.expand_dims(image,2)


        image = np.transpose(image, (2, 0, 1))
        rgb_image = np.transpose(rgb_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        plane_mask = np.transpose(plane_mask, (2, 0, 1))
        return image, rgb_image, mask, plane_mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        rgb_img_path = self.rgb_images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = dir_path.split("/")[-2]
        if base_dir == 'good':
            image, rgb_image, mask, plane_mask = self.transform_image(img_path,rgb_img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../gt/')
            mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, rgb_image, mask, plane_mask = self.transform_image(img_path,rgb_img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)


        sample = {'image': image, 'fg_mask':plane_mask, 'rgb_image':rgb_image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample

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

class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, d_path="/data/mvtec3d/*/train/good/", resize_shape=None, mixup=False):
        self.mixup = mixup

        self.images = sorted(glob.glob(d_path+"xyz/*.tiff"))
        self.rgb_images = sorted(glob.glob(d_path+"rgb/*.png"))
        self.global_min = 1000000
        self.global_max = 0.0
        for image_path in self.images:
            image = tif.imread(image_path).astype(np.float32)
            image_t = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
            image = image_t[:, :, 2]
            zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))

            plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
            plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
            plane_mask = fill_plane_mask(plane_mask)
            image = image * plane_mask[:,:,0]
            zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))


            im_max= np.max(image)
            im_min = np.min(image * (1.0-zero_mask) + 1000 * zero_mask)
            self.global_min = min(self.global_min, im_min)
            self.global_max = max(self.global_max, im_max)
        self.global_min = self.global_min * 0.9
        self.global_max = self.global_max * 1.1

        self.resize_shape=resize_shape
        self.rot_rgb = iaa.Rotate((-15, 15),seed=1)
        self.rot_d = iaa.Rotate((-15, 15), seed=1)

    def __len__(self):
        #return len(self.images)        
        return 4000


    def transform_image(self, image_path, rgb_img_path):
        rgb_image = cv2.imread(rgb_img_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        image = tif.imread(image_path).astype(np.float32)

        if self.resize_shape != None:
            #h, w
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]), 0, 0, interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.resize(rgb_image, (self.resize_shape[1], self.resize_shape[0])).astype(np.float32) / 255.0

        image_t = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        image = image_t[:, :, 2]
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)

        image = image * plane_mask[:,:,0] # remove background

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0-zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min) # normalize image according to it's min and max values
        image = image * 0.8 + 0.1 # leave some room for anomaly generation
        image = image * (1.0 - zero_mask) # set missing pixels to 0
        image = fill_depth_map(image) # fill missing pixels with mean of local valid values

        _, perlin_thr, _, _ = generate_perlin_noise(self.resize_shape)
        perlin_thr = perlin_thr * plane_mask
        msk = (perlin_thr).astype(np.float32)
        msk[:,:,0] = msk[:,:,0] * (1.0 - zero_mask)

        image = np.expand_dims(image,2)
        rgb_image = self.rot_rgb(image=rgb_image)
        image = self.rot_d(image=image)

        image = np.transpose(image, (2, 0, 1))
        rgb_image = np.transpose(rgb_image, (2, 0, 1))
        msk = np.transpose(msk, (2, 0, 1))
        return image, rgb_image, msk

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.images), (1,)).item()
        img_path = self.images[idx]
        rgb_img_path = self.rgb_images[idx]
        image, rgb_image, anomaly_mask = self.transform_image(img_path, rgb_img_path)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            anomaly_mask = anomaly_mask * 0.0


        sample = {'image': image, 'rgb_image':rgb_image, "mask":anomaly_mask}

        return sample
