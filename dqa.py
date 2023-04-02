import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
import cv2
import numpy
import numpy as np
from einops.layers.torch import Rearrange
  
class DQA(nn.Module):
    def __init__(self,img_size=384, patch_size=4, in_chans=3,  embed_dim=80):
        super(DQA, self).__init__()
        self.patches = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h = img_size//4, w = img_size//4),
            nn.LayerNorm(embed_dim)
        )
        
 
        
    def forward(self, rgb,depth):
        B = rgb.shape[0]
        rgb = self.patches(rgb)
        B,L,C = rgb.shape
        B1 = depth.shape[0]
        depth = self.patches(depth)
        B1,L1,C1 = depth.shape
        print('rgb shape patches',rgb.shape)
        print('depth shape patches',depth.shape)
        return rgb,depth
      
      
def build_model():
 
        return DQA()
