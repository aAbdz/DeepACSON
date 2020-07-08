#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aliabd
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class resBlock(nn.Module):   
    def __init__(self, in_channels):
        super().__init__()        
        self.res_blk = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
            )
    
    def forward(self, x):
        return x + self.res_blk(x)    

   
    
class basicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.basic_block(x)


class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = basicBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = basicBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        diffZ = torch.tensor([x2.size(2) - x1.size(2)])        
        diffY = torch.tensor([x2.size(3) - x1.size(3)])
        diffX = torch.tensor([x2.size(4) - x1.size(4)])
        

        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()     
        self.maxpool_conv = nn.Sequential(
            resBlock(in_channels),
            basicBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(2)                       
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

    
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv_first = basicBlock(n_channels, 64, kernel_size=3, stride=1, padding=1)
                
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        
        self.up1 = up(1024+512, 512)
        self.up2 = up(512+256, 256)
        self.up3 = up(256+128, 128)
        self.up4 = up(128+64, 64)
        
        self.conv_last = nn.Conv3d(64, n_classes, kernel_size=1)
        self.softmax = nn.Sigmoid()
        
        
    def forward(self, x):
        x1 = self.conv_first(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.conv_last(x)
        out = self.softmax(out)
        return out

    
class FCN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
    
        self.conv1 = basicBlock(n_channels, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = basicBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = basicBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = basicBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv3d(512, n_classes, kernel_size=1)
        self.softmax = nn.Sigmoid()
        
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv_last(x4)
        out = self.softmax(out)
        return out    
    
    
    
    

    
    
    
    
    