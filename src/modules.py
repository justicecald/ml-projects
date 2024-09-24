import pandas as pd
import numpy as np
import torch
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
from linear import LinearNeuralNetwork

from einops import rearrange
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt

def time_position_embedding(time_steps, time_embed_dim):
    denom_fact = 10000 ** (torch.arange(start=0, end=(time_embed_dim//2)) / (time_embed_dim // 2))
    time_embedding = time_steps[:, None].repeat(time_embed_dim//2, 1) / denom_fact

    print(f"Denominator Shape: {denom_fact.shape}\n")
    print(f"Time Embedding Shape: {time_embedding.shape}")

    output = torch.zeros(len(time_steps), time_embed_dim)
    output[:, ::2] = torch.sin(time_embedding)
    output[:, 1::2] = torch.cos(time_embedding)

    return output

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, time_emb_dim = None, groups = 8, embed_time: bool = False):
        super(ResNetBlock, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.t_embed_dim = time_emb_dim
        self.num_groups = groups

        self.embed_time = self.apply_time_embedding() if embed_time else None

        self.block_one = self.build_sub_block_one(self.in_dim, self.out_dim, groups=self.num_groups)
        self.block_two = self.build_sub_block_two(self.out_dim, self.out_dim, groups=self.num_groups)
        self.residual_conv = ConvolutionalNeuralNetwork_2D(self.in_dim, self.out_dim, (1,1)) if self.in_dim != self.out_dim else nn.Identity()


    def apply_time_embedding(self):
        return nn.Sequential(
            nn.SiLU(),
            LinearNeuralNetwork(in_features=self.t_embed_dim, out_features=self.out_dim)
        )

    def build_sub_block_one(self, in_dim, out_dim, groups = None):
        return nn.Sequential(
            ConvolutionalNeuralNetwork_2D(in_dim, out_dim, (3,3), padding_type='same'),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU()
        )
    
    def build_sub_block_two(self, in_dim, out_dim, groups = None):
        return nn.Sequential(
            ConvolutionalNeuralNetwork_2D(in_dim, out_dim, (3,3), padding_type='same'),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU()
        )
    
    def forward(self, x, time_emb: torch.Tensor = None):
        
        residue = x

        x = self.block_one(x)

        time = self.embed_time(time_emb)

        input_with_time = x + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.block_two(input_with_time)

        return merged + self.residual_conv(residue)

class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.in_dim = input_dim
        self.embedding_dim = embed_dim
        self.channels = self.in_dim * self.num_heads

        self.attention_block_one = nn.Sequential(
            nn.GroupNorm(8, self.channels, eps=1e-6),
            ConvolutionalNeuralNetwork_2D(self.channels, self.channels, kernel_size=(1,1))
        )

        self.attention_block_two = nn.Sequential(
            nn.LayerNorm(self.channels),
            _AttentionArithmetic(self.channels, self.num_heads)
        )

        self.attention_block_last = ConvolutionalNeuralNetwork_2D(self.channels, self.channels, kernel_size=(1,1))

    def forward(self, x: torch.Tensor):
        residue_end = x
        x = self.attention_block_one(x)

        n, c, h, w = x.shape

        x = x.view((n, c, h*w))
        x = x.transpose(-1, -2)

        residue_after_attention = x

        x = self.attention_block_two(x)
        x += residue_after_attention

        x = x.transpose(-1, -2)
        x = x.view((n,c,h,w))
        
        return self.attention_block_last(x) + residue_end

class UnetOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetOutput, self).__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = ConvolutionalNeuralNetwork_2D(in_channels, out_channels, (3,3), padding_type='same')

    def forward(self, x):
        x = self.group_norm(x)
        x = nn.functional.silu(x)
        x = self.conv(x)

        return x

# if __name__ == "__main__":
#     pass
