import pandas as pd
import numpy as np
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
import matplotlib.pyplot as plt

class U_Net_Convolution(nn.Module):
    def __init__(self):
        super(U_Net_Convolution, self).__init__()

    @staticmethod
    def time_position_embedd(time_steps, time_embed_dim):
        denom_fact = 10000 ** (np.arange(start=0, stop=(time_embed_dim//2)) / (time_embed_dim // 2))
        time_embedding = time_steps[:, None].repeat(time_embed_dim//2, 1) / denom_fact

        print(f"Denominator Shape: {denom_fact.shape}\n")

        time_embedding = np.concatenate((np.sin(time_embedding), np.cos(time_embedding)), axis=1)
        print(f"Time Embedding Shape: {time_embedding.shape}")

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
    
    def _build_resnet(self):
        # Build the first Resnet block by default:
        self.module_list.append(
            nn.GroupNorm(num_groups=8, num_channels=self.in_channels),
            nn.ReLU(),
            ConvolutionalNeuralNetwork_2D(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=1, padding_type='same')
        )
    
    def _build_time_embedding(self):
        self.module_list.append(
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.out_channels)
        )

    def _build_self_attention_norm(self):
        self.module_list.append(
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.out_channels)
        )


    

if __name__ == "__main__":
    l = np.random.rand(3, 64, 64)
    a = ConvolutionalNeuralNetwork_2D(3, 64, (3, 3), 1, padding_type='valid')
    a.convlutional_activation_layer(a.weights.shape[0], a.weights.shape[0])
    output = a(l)
    output = output.sum()
    output.backward()
    print(f"dO / dW:\n{a.weights.grad}\ndO / dB:\n{a.bias.grad}")
    
