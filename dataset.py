import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import torch


import rasterio
'''with rasterio.open("data/Kivuwest.tif") as f:
    img = f.read()
    #img = f.read(window=((0, 100),(0, 100)))
print(img.shape)

with rasterio.open("data/Sentinel-2_northkivuwest.tif") as f2:
    img2 = f2.read()
    img2= np.array([img2[3],img2[2],img2[1],img2[7]])
print(img2.shape)'''

class Dataset(BaseDataset):
    
    def __init__(
            self, 
            images_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        #self.ids = os.listdir(images_dir)
        self.images_dir = images_dir
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.wsize = 256
        with rasterio.open(self.images_dir) as f:
            imgallchannels = f.read()
            # Only select the R,G,B,NIR channels
            image = np.array([imgallchannels[3],imgallchannels[2],imgallchannels[1],imgallchannels[7]])
            self.image = image
            shapearr = image.shape
            self.num_smpls = shapearr[0]
            self.sh_x = shapearr[1]
            self.sh_y = shapearr[2]

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls*self.sh_x/self.wsize*self.sh_y/self.wsize
        self.num_windows = int(self.num_windows)
        
    
    def __getitem__(self, index):
        
        # determine where to crop a window from all images (no overlap)
        m = index*self.wsize%self.sh_x # iterate from left to right
        # increase row by windows size everytime m increases
        n = (int(np.floor(index * self.wsize/self.sh_x))*self.wsize)%self.sh_x
        #determine which batch to use
        b = (index*self.wsize*self.wsize // (self.sh_x*self.sh_y))%self.num_smpls
        # crop all data at the previously determined position
        img_sample = self.image[b, n:n+self.wsize, m:m+self.wsize]

        pad_x,pad_y = 0,0
        if self.sh_x<self.wsize:
            pad_x = self.wsize - self.sh_x
        if self.sh_y<self.wsize:
            pad_y = self.wsize - self.sh_y

        x_sample = np.pad(self.image,[[0,pad_x],[0,pad_y],[0,0]])
        
        return torch.tensor(np.asarray(x_sample))
        
    def __len__(self):
        return self.num_windows


