#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:59:11 2022

@author: ruswang
"""
# from osgeo import gdal
from ctypes import util
from sched import scheduler
import cv2
import matplotlib.pyplot as plt
from dataset import Dataset 
import os

import rasterio
'''with rasterio.open("data/Kivuwest.tif") as f:
    # img = f.read()
    img = f.read(window=((0, 100),(0, 100)))

print(img.shape)'''

import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from typing import List
from catalyst import utils
from catalyst.contrib.optimizers import RAdam, Lookahead
from catalyst.dl import SupervisedRunner
from pathlib import Path
'''

"""
Creat class to read images and applies augmentation
"""
class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir,
        transforms=None
    ) -> None:
        self.images_dir = images_dir
        self.ids = os.listdir(images_dir)
        self.transforms = transforms
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images_fps[idx]
        image = cv2.imread(image_path)
        
        result = {"image": image}
        
        if self.transforms is not None:
            result = self.transforms(**result)
        
        result["filename"] = image_path.name

        return result

"""
Augmentation
"""
def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
      albu.RandomRotate90(),
      albu.Cutout(),
      albu.RandomBrightnessContrast(
          brightness_limit=0.2, contrast_limit=0.2, p=0.3
      ),
      albu.GridDistortion(p=0.3),
      albu.HueSaturationValue(p=0.3)
    ]

    return result
  
def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
      albu.SmallestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
      albu.LongestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
      albu.OneOf([
          random_crop,
          rescale,
          random_crop_big
      ], p=1)
    ]

    return result

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensorV2()]
  
def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result

train_transforms = compose([
    resize_transforms(), 
    hard_transforms(), 
    post_transforms()
])
valid_transforms = compose([pre_transforms(), post_transforms()])

"""
Loaders
"""
import collections
from sklearn.model_selection import train_test_split

def get_loaders(
    images: List[Path],
    random_state: int,
    valid_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transforms_fn = None,
    valid_transforms_fn = None,
) -> dict:

    indices = np.arange(len(images))

    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(
      indices, test_size=valid_size, random_state=random_state, shuffle=True
    )

    np_images = np.array(images)

    # Creates our train dataset
    train_dataset = SegmentationDataset(
      images = np_images[train_indices].tolist(),
      transforms = train_transforms_fn
    )

    # Creates our valid dataset
    valid_dataset = SegmentationDataset(
      images = np_images[valid_indices].tolist(),
      transforms = valid_transforms_fn
    )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=True,
    )

    valid_loader = DataLoader(
      valid_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders

batch_size = 32
SEED = 42

print(f"batch_size: {batch_size}")
train_image_path = Path("data/x_train")
ALL_IMAGES = sorted(train_image_path.glob("*.tif"))

loaders = get_loaders(
    images=ALL_IMAGES,
    random_state=SEED,
    train_transforms_fn=train_transforms,
    valid_transforms_fn=valid_transforms,
    batch_size=batch_size
)



ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'identity'
DEVICE = 'cuda'

#create regression model with pretrained encoder
model = smp.FPN(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    in_channels = 4,
    classes = 2,
    activation = ACTIVATION,
)

learning_rate = 0.001
encoder_learning_rate = 0.0005

layerwise_params = {"encoder*": dict(lr = encoder_learning_rate, weight_decay = 0.00003)}
model_params = util.process_model_params(model, layerwise_params = layerwise_params)

base_optimizer = RAdam(model_params, lr = learning_rate, weight_decay=0.0003)
optimizer = Lookahead(base_optimizer)
scheduler = torch.optim.lr_scheduler.ReduceLROnPLateau(optimizer, factor=0.25, patience = 2)

runner = SupervisedRunner(device=torch.device, input_key="image")
runner.train(
    model = model,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders = loaders,
    logdir="./logs",
    num_epochs=3,
    verbose=True,
)
'''


ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'identity'
DEVICE = 'cuda'

#create regression model with pretrained encoder
model = smp.FPN(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    in_channels = 4,
    classes = 2,
    activation = ACTIVATION,
)

# Loading training and validation data
# x_train_dir = "data/x_train/Sentinel-2_northkivunorth.tif"
# y_train_dir = "data/y_train/Kivunorth.tif"
# x_valid_dir = "data/x_valid/Sentinel-2_northkivusouth1.tif"
# y_valid_dir = "data/y_valid/Kivusouth1.tif"
# x_test_dir = "data/x_test/Sentinel-2_northkivuwest.tif"
# y_test_dir = "data/y_test/Kivuwest.tif"
x_train_dir = "data/x_train"
y_train_dir = "data/y_train"
x_valid_dir = "data/x_valid"
y_valid_dir = "data/y_valid"
x_test_dir = "data/x_test"
y_test_dir = "data/y_test"

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    #augmentation=get_training_augmentation(), 
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    #augmentation=get_validation_augmentation(),
)


train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 8)
valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers = 4)

def loss_fn(pred, y):
    err = pred-y

'''
Define training loop
'''
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred,y)
        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




#Loss Function
## TODO: change loss function
loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='none')
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

#create epoch runners
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE, 
    verbose=True,
)

#train model for 40 epochs
max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
        
### Test best saved model
# load best saved checkpoint
best_model = torch.load('./best_model.pth')



# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
)

test_dataloader = DataLoader(test_dataset)


