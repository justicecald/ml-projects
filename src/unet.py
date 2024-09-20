import pandas as pd
import numpy as np
import torch
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
import matplotlib.pyplot as plt

class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.time_embedding = self.time_position_embedding(1000, 120)
        self.unet = unet()
        self.final_layer = unet_output_layer()

    def time_position_embedding(self, time_steps, time_embed_dim):
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
    
    def forward(self, x):
        # Defining the time embedding
        time_step = torch.rand(0, self.time_embedding.shape[0])
        time = self.time_embedding[time_step]

        input_shape = x.shape

        unet_output = self.unet(x, time)

        final_layer_unet_output = self.final_layer(unet_output)

    
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding, num_heads=None, t_emb_dim=None, embed=False):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed = embed
        self.num_heads = num_heads
        self.embedding_dim = t_emb_dim
        self.time_embedding = time_embedding
        self.module_list = []
        self.embed_time = self._build_time_embedding(self.time_embedding)
        self.sequence = None

    def _build_down_block(self):
        self._build_first_resnet()
    
    def _build_first_resnet(self):
        # Build the first Resnet block by default:
        self.module_list.append(
            nn.GroupNorm(num_groups=8, num_channels=self.in_channels),
            nn.SiLU(),
            ConvolutionalNeuralNetwork_2D(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=1, padding_type='same')
        )

    def _build_second_resnet(self):
        # Build the first Resnet block by default:
        self.module_list.append(
            nn.GroupNorm(num_groups=8, num_channels=self.out_channels),
            nn.SiLU(),
            ConvolutionalNeuralNetwork_2D(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=1, padding_type='same')
        )
    
    def _build_time_embedding(self):
        self.module_list.append(
            nn.SiLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.out_channels)
        )

    def _add_time_embedding(self, time_embedding, time_step, x):
        return x + time_embedding[time_step]

    def _build_self_attention_norm(self, batch, embed_dim, num_heads):
        self.module_list.append(
            nn.GroupNorm(8, self.out_channels),
            _AttentionArithmetic(batch, embed_dim, num_heads),
        )

    def _build_down_sample(self, out_channels):
        self.module_list.append(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        )

    def _build_input_residual(self, in_channels, out_channels):
        self.module_list.append(
            ConvolutionalNeuralNetwork_2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1))
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
