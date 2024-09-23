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

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, time_emb_dim = None, groups = 8, embed_time: bool = False):
        super(ResNetBlock, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.t_embed_dim = time_emb_dim
        self.num_groups = groups

        self.embed_time_trigger = embed_time

        self.mlp = self.apply_time_embedding() if embed_time else None

        self.block_one = self.build_sub_block(self.in_dim, self.out_dim, groups=self.num_groups)
        self.block_two = self.build_sub_block(self.out_dim, self.out_dim, groups=self.num_groups)
        self.residual_conv = ConvolutionalNeuralNetwork_2D() if self.in_dim != self.out_dim else nn.Identity()

    def apply_time_embedding(self):
        self.module_list.append(
            nn.SiLU(),
            LinearNeuralNetwork(in_features=self.t_embed_dim, out_features=0)
        )

    def fuse_time_embedding(self, time_embedding, time_step, x):
        return x + time_embedding[time_step]

    def build_sub_block(self):
        return nn.Sequential(
            ConvolutionalNeuralNetwork_2D(self.in_dim, self.out_dim, (3,3), padding_type='same'),
            nn.GroupNorm(self.num_groups, self.out_dim),
            nn.SiLU()
        )

class SelfAttentionBlock(nn.Module):
    def __init__(self, batch_dim, input_dim, embed_dim, num_heads = 1):
        super(SelfAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.in_dim = input_dim
        self.batch_dim = batch_dim
        self.embedding_dim = embed_dim

        self.attention_block = nn.Sequential(
            nn.GroupNorm(num_groups=8),
            _AttentionArithmetic(batch_dim=0, embed_dim=0, num_heads=self.num_heads)
        )

if __name__ == "__main__":
    # l = np.random.rand(3, 64, 64)
    # a = ConvolutionalNeuralNetwork_2D(3, 64, (3, 3), 1, padding_type='valid')
    # a.convlutional_activation_layer(a.weights.shape[0], a.weights.shape[0])
    # output = a(l)
    # output = output.sum()
    # output.backward()
    # print(f"dO / dW:\n{a.weights.grad}\ndO / dB:\n{a.bias.grad}")
    # time_steps = np.linspace(start=1e-4, stop=0.002, num=1000)
    # time_embedding = U_Net_Convolution.time_position_embedd(time_steps=np.array(range(0, len(time_steps))), time_embed_dim=32)
    pass
