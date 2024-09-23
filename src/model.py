import pandas as pd
import numpy as np
import torch
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
from modules import *

from einops import rearrange
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt

    
class UnetContructor(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding, num_heads=None, t_emb_dim=None, embed=False):
        super(UnetContructor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed = embed
        self.num_heads = num_heads
        self.embedding_dim = t_emb_dim
        self.time_embedding = time_embedding
        self.module_list = []
        self.embed_time = self._build_time_embedding(self.time_embedding)
        self.sequence = None
    
    def down_sample(dim, out_dim = None):
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            ConvolutionalNeuralNetwork_2D(dim * 4, out_dim if out_dim else dim, (1, 1))
            # Reference code: nn.Conv2d(dim * 4, default(dim_out, dim), 1)
        )

    def up_sample(dim, out_dim = None):
        return nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            ConvolutionalNeuralNetwork_2D(dim, out_dim if out_dim else dim, (3, 3), padding_type='same')
            # Reference code: nn.Conv2d(dim, out_dim if out_dim else dim, 3, padding = 1)
        )
    
    def apply_resnet(self):
        # Build the first Resnet block by default:
        self.module_list.append(
            ResNetBlock() # TO DO: Fill out the inputs here
        )

    def apply_self_attention_norm(self, batch, embed_dim, num_heads):
        self.module_list.append(
            nn.GroupNorm(8, self.out_channels),
            _AttentionArithmetic(batch, embed_dim, num_heads),
        )

    def apply_down_sample(self, out_channels):
        self.module_list.append(
            self.down_sample()
        )

    def apply_down_sample(self, out_channels):
        self.module_list.append(
            self.up_sample()
        )

    def forward(self, x):
        # Unpacking the build layers based on what was build
        self.sequence = nn.Sequential(*self.module_list)
    
