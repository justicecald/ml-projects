import pandas as pd
import numpy as np
import torch
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
from modules import *
import matplotlib.pyplot as plt

class SwitchSequential(nn.Sequential):
    def forward(self, x, time):
        for layer in self:
            if isinstance(layer, ResNetBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UnetContructor(nn.Module):
    def __init__(self):
        super(UnetContructor, self).__init__()

        self.encoder = nn.Sequential([
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ConvolutionalNeuralNetwork_2D(3, 64, (3,3), padding_type='same')),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(64, 64), SelfAttentionBlock(32, 4)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(64, 64), SelfAttentionBlock(32, 4)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(64, 128), SelfAttentionBlock(64, 4)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(128, 128), SelfAttentionBlock(64, 4)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(128, 256), SelfAttentionBlock(64, 4)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(256, 256), SelfAttentionBlock(64, 4)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(256, 512), SelfAttentionBlock(128, 4)),
            
            # <ADD DIM BREAKDOWN> 
            SwitchSequential(ResNetBlock(512, 512), SelfAttentionBlock(128, 4)),

            # <ADD DIM BREAKDOWN> 
            SwitchSequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1))
        ])

        # TODO
        self.bottleneck = SwitchSequential(
            # <ADD DIM BREAKDOWN>
            ResNetBlock(512, 512), 
            
            # <ADD DIM BREAKDOWN>
            SelfAttentionBlock(128, 4), 
            
            # <ADD DIM BREAKDOWN>
            ResNetBlock(512, 512), 
        )

        self.decoder = nn.Sequential([
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(1024, 512)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(1024, 512)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(1024, 512), self.up_sample(512)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(1024, 512), SelfAttentionBlock(8, 160)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(1024, 512), SelfAttentionBlock(8, 160)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(768, 512), SelfAttentionBlock(8, 160), self.up_sample(512)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(768, 256), SelfAttentionBlock(8, 80)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(512, 256), SelfAttentionBlock(8, 80)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(384, 256), SelfAttentionBlock(8, 80), self.up_sample(256)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(384, 128), SelfAttentionBlock(8, 40)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(256, 128), SelfAttentionBlock(8, 40)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(192, 128), SelfAttentionBlock(8, 40), self.up_sample(128)),

            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(192, 64), SelfAttentionBlock(8, 40)),
            
            # <ADD DIM BREAKDOWN>
            SwitchSequential(ResNetBlock(128, 64), SelfAttentionBlock(8, 40))
        ])

    def up_sample(dim, out_dim = None):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvolutionalNeuralNetwork_2D(dim, out_dim if out_dim else dim, (3, 3), padding_type='same')
        )

    def apply_output_layer(self):
        self.module_list.append(
            UnetOutput()
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, time)
            skip_connections.append(x)

        x = self.bottleneck(x, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, time)
        
        return x

class UnetOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetOutput, self).__init__()
        self.group_norm = nn.GroupNorm(8, in_channels)
        self.conv = ConvolutionalNeuralNetwork_2D(in_channels, out_channels, (3,3), padding_type='same')

    def forward(self, x):
        x = self.group_norm(x)
        x = nn.functional.silu(x)
        x = self.conv(x)

        return x
    
