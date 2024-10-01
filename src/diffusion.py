import pandas as pd
import numpy as np
import torch
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
from model import *
import matplotlib.pyplot as plt

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = time_position_embedding(1000, 320)
        self.unet = UnetContructor()
        self.final = UnetOutput(64, 3)
    
    def forward(self, x, context, time):

        time = self.time_embedding(time)
        
        output = self.unet(x, time)
        
        output = self.final(output)
        
        return output
    
# if __name__ == "__main__":
#     betas = np.linspace(start=1e-4, stop=0.002, num=1000)
#     time_emb = time_position_embedding(np.array(range(0, len(betas))), 320)
#     print(time_emb.shape)
#     print(time_emb[200])