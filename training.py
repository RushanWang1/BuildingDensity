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
from buildingdataset import BuildingDataset 
from unet import UNet
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

ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'identity'
DEVICE = 'cuda'

#create regression model with pretrained encoder
model = smp.FPN(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    in_channels = 4,
    classes = 1,
    activation = ACTIVATION,
)

# Loading training and validation data
x_train_dir = "data/x_train/Sentinel-2_northkivunorth.tif"
y_train_dir = "data/y_train/Kivunorth.tif"
x_valid_dir = "data/x_valid/Sentinel-2_northkivusouth1.tif"
y_valid_dir = "data/y_valid/Kivusouth1.tif"
x_test_dir = "data/x_test/Sentinel-2_northkivuwest.tif"
y_test_dir = "data/y_test/Kivuwest.tif"
# x_train_dir = "data/x_train"
# y_train_dir = "data/y_train"
# x_valid_dir = "data/x_valid"
# y_valid_dir = "data/y_valid"
# x_test_dir = "data/x_test"
# y_test_dir = "data/y_test"

train_dataset = BuildingDataset(
    x_train_dir,
    y_train_dir,
    #augmentation=get_training_augmentation(), 
)

valid_dataset = BuildingDataset(
    x_valid_dir, 
    y_valid_dir, 
    #augmentation=get_validation_augmentation(),
)

test_dataset = BuildingDataset(
    x_test_dir, 
    y_test_dir, 
    #augmentation=get_validation_augmentation(),
)


train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = False, num_workers = 12)
valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 4)
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# TODO: Loss function
loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='none')
loss_mean = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

def loss_fn(pred, y):
    err = pred-y
    loss1 = err[0].mean()
    mask = y[0]>0
    loss2 = err[1][mask].mean()
    loss = loss1+loss2
    return loss



'''
Define training loop
'''
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch = 0
    for X, y in dataloader:
        # Compute prediction and loss
        pred = model(X)
        # pred = UNet(X)
        loss = loss_fn(pred,y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        batch += 1

"""
Define test loop
"""
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")

# #Loss Function
# ## TODO: change loss function
# loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='none')
# metrics = [
#     smp.utils.metrics.IoU(threshold=0.5),
# ]

# optimizer = torch.optim.Adam([ 
#     dict(params=model.parameters(), lr=0.0001),
# ])

# #create epoch runners
# train_epoch = smp.utils.train.TrainEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     optimizer=optimizer,
#     device=DEVICE,
#     verbose=True,
# )

# valid_epoch = smp.utils.train.ValidEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     device=DEVICE, 
#     verbose=True,
# )

# #train model for 40 epochs
# max_score = 0

# for i in range(0, 40):
    
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
    
#     # do something (save model, change lr, etc.)
#     if max_score < valid_logs['iou_score']:
#         max_score = valid_logs['iou_score']
#         torch.save(model, './best_model.pth')
#         print('Model saved!')
        
#     if i == 25:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')
        
# ### Test best saved model
# # load best saved checkpoint
# best_model = torch.load('./best_model.pth')



# # create test dataset
# test_dataset = Dataset(
#     x_test_dir, 
#     y_test_dir, 
# )

# test_dataloader = DataLoader(test_dataset)


