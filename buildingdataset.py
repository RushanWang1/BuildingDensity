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

with rasterio.open("data/x_test/Sentinel-2_northkivuwest.tif") as f2:
    img2 = f2.read()
    img2= np.array([img2[3],img2[2],img2[1],img2[7]])
print(img2.shape)'''

class BuildingDataset(BaseDataset):
    
    def __init__(
            self, 
            images_dir, 
            gt_dir,#Ground truth
    ):
        #self.ids = os.listdir(images_dir)
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.wsize = 256
        with rasterio.open(self.images_dir) as f:
            imgallchannels = f.read()
            # Only select the R,G,B,NIR channels
            image = np.array([imgallchannels[3],imgallchannels[2],imgallchannels[1],imgallchannels[7]])
            # image = np.array([imgallchannels[3],imgallchannels[2],imgallchannels[1]])
            self.image = image
            shapearr = image.shape
            self.num_smpls = shapearr[0] # number of channels
            self.sh_x = shapearr[1] 
            self.sh_y = shapearr[2]

        with rasterio.open(self.gt_dir) as f:
            gt = f.read()
            self.gt = gt

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls*(self.sh_y - self.wsize +1)*(self.sh_x - self.wsize +1)
        self.num_windows = int(self.num_windows)
        
    
    def __getitem__(self, index):
        
        # determine where to crop a window from all images 
        m = index%self.sh_x
        n = (int(np.floor(index/self.sh_x)))%self.sh_x
        print(index, n,m)
        # n = int(np.floor((index+self.wsize-1)/self.sh_y))
        img_sample = self.image[:,n:n+self.wsize, m:m+self.wsize]
        gt_sample = self.gt[:,n:n+self.wsize, m:m+self.wsize]
        print(img_sample.shape, gt_sample.shape)
        return torch.tensor(img_sample), torch.tensor(gt_sample)
        
    def __len__(self):
        return self.num_windows


