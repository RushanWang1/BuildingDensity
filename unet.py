import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

WINDOW_SIZE = 256

class conv_block(nn.Module):
    """
    Convolution Block 
    
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode = 'bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

    
class UNet(nn.Module):
    def __init__(self,in_channels = 4, out_channels = 1):
        super().__init__()

        self.codename = 'UNet'

        filters = [32,64,128,256]
        self.Maxpool = nn.MaxPool2d( 2, stride = 2)
        
        self.Conv1 = conv_block(in_channels,filters[0])
        self.Conv2 = conv_block(filters[0],filters[1])
        self.Conv3 = conv_block(filters[1],filters[2])
        self.Conv4 = conv_block(filters[2],filters[3])
        
        self.Up4 = up_conv(filters[3],filters[2])
        self.Up_conv4 = conv_block(filters[3],filters[2])
        
        self.Up3 = up_conv(filters[2],filters[1])
        self.Up_conv3 = conv_block(filters[2],filters[1])
        
        self.Up2 = up_conv(filters[1],filters[0])
        self.Up_conv2 = conv_block(filters[1],filters[0])
        
        
        self.Up_conv1 = conv_block(filters[0],out_channels)
        

    def forward(self, x):
        b1 = self.Conv1(x) #[size,size]
        
        b2 = self.Maxpool(b1) #[size/2,size/2]
        b2 = self.Conv2(b2)
        
        b3 = self.Maxpool(b2) #[size/4,size/4]
        b3 = self.Conv3(b3)
        
        b4 = self.Maxpool(b3) #[size/8,size/8]
        b4 = self.Conv4(b4)
        
        c4 = self.Up4(b4)   #[size/4,size/4]
        c4 = torch.cat((b3,c4),dim = 1)
        c4 = self.Up_conv4(c4)
        
        c3 = self.Up3(c4)  #[size/2,size/2]
        c3 = torch.cat((b2,c3),dim = 1)
        c3 = self.Up_conv3(c3)
        
        c2 = self.Up2(c3)   #[size,size]
        c2 = torch.cat((b1,c2),dim = 1)
        c2 = self.Up_conv2(c2)
        
        c1 = self.Up_conv1(c2)
        
        return c1