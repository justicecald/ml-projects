import pandas as pd
import numpy as np
import torch
import math
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
from linear import LinearNeuralNetwork

import matplotlib.pyplot as plt

torch.set_default_device("mps")

def time_position_embedding(ts_arr, time_embed_dim, device):
    ts_arr = torch.Tensor(ts_arr).to(device=device)
    half_dim = time_embed_dim // 2

    denom_array = torch.arange(start=0, end=half_dim)

    embeddings = torch.exp(denom_array * -(math.log(10000) / half_dim - 1))
    time_embedding = ts_arr[:, None] * embeddings[None, :]

    output = torch.zeros(time_embedding.shape[0], time_embed_dim)
    output[:, ::2] = torch.sin(time_embedding)
    output[:, 1::2] = torch.cos(time_embedding)

    return output

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, groups = 8):
        super(ResNetBlock, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.t_embed_dim = 32
        self.num_groups = groups

        self.block_one = self.build_sub_block_one(self.in_dim, self.out_dim, groups=self.num_groups)
        self.block_two = self.build_sub_block_two(self.out_dim, self.out_dim, groups=self.num_groups)
        self.time_embedding = self.apply_time_embedding()
        self.residual_conv = ConvolutionalNeuralNetwork_2D(self.in_dim, self.out_dim, (1,1), padding_type='valid') if (self.in_dim != self.out_dim) else nn.Identity()


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

        time = self.time_embedding(time_emb)
        print(f"Time Shape: {time.shape}")

        print(f"Fusing input with time...")
        input_with_time = x + time.unsqueeze(-1).unsqueeze(-1)
        print(f"Input with time shape: {input_with_time.shape}")
        
        merged = self.block_two(input_with_time)

        output = merged + self.residual_conv(residue)

        return output
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embed_dim
        self.in_dim = in_dim
        self.channels = self.embedding_dim * self.num_heads
        self.num_groups = 2

        self.attention_block_one = nn.Sequential(
            nn.GroupNorm(self.num_groups, self.in_dim, eps=1e-6),
            ConvolutionalNeuralNetwork_2D(self.channels, self.channels, kernel_size=(1,1), padding_type='same')
        )

        self.attention_layer_norm = nn.LayerNorm(self.channels)

        self.attention_block_two = _AttentionArithmetic(self.in_dim, self.embedding_dim, self.num_heads)

        self.attention_block_last = ConvolutionalNeuralNetwork_2D(self.channels, self.channels, kernel_size=(1,1), padding_type='same')

    def forward(self, x: torch.Tensor):

        residue_end = x
        x = self.attention_block_one(x)

        n, c, h, w = x.shape

        x = x.view((n, c, h*w))
        x = x.transpose(1, 2)
        residue_after_attention = x

        x = self.attention_layer_norm(x)
        x += residue_after_attention

        x = self.attention_block_two(x, (n, c, h, w))
        
        return self.attention_block_last(x) + residue_end
