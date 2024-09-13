import pandas as pd
import numpy as np
import torch
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
import matplotlib.pyplot as plt

class U_Net_Convolution(nn.Module):
    def __init__(self):
        super(U_Net_Convolution, self).__init__()

    @staticmethod
    def time_position_embedd(time_steps, time_embed_dim):
        denom_fact = 10000 ** (torch.arange(start=0, end=(time_embed_dim//2)) / (time_embed_dim // 2))
        time_embedding = time_steps[:, None].repeat(time_embed_dim//2, 1) / denom_fact

        print(f"Denominator Shape: {denom_fact.shape}\n")
        print(f"Time Embedding Shape: {time_embedding.shape}")

        output = torch.zeros(len(time_steps), time_embed_dim)
        output[:, ::2] = torch.sin(time_embedding)
        output[:, 1::2] = torch.cos(time_embedding)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.imshow(output)
        # ax1.set_aspect('auto')
        # fig.savefig('equal.png')

        return time_embedding
    
class ResNet_SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=None, t_emb_dim=None, embed=False):
        super(ResNet_SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed = embed
        self.num_heads = num_heads
        self.embedding_dim = t_emb_dim
        self.module_list = []
        self.sequence = None

        # Building the sequence
    
    def _build_resnet(self):
        # Build the first Resnet block by default:
        self.module_list.append(
            nn.GroupNorm(num_groups=8, num_channels=self.in_channels),
            nn.SiLU(),
            ConvolutionalNeuralNetwork_2D(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=1, padding_type='same')
        )
    
    def _build_time_embedding(self):
        self.module_list.append(
            nn.SiLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.out_channels)
        )

    def _build_self_attention_norm(self):
        self.module_list.append(
            nn.SiLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.out_channels)
        )

    def forward(self, x):
        # Unpacking the build layers based on what was build
        self.sequence = nn.Sequential(*self.module_list)



    

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
