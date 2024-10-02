import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import Tensor, cat, empty, autograd
import matplotlib.pyplot as plt
from collections import *
from linear import LinearNeuralNetwork

"""
Referencing:
- A guide to convolution arithmetic for deep learning (https://arxiv.org/pdf/1603.07285)
"""
torch.set_default_device("mps")

class _ConvolutionArithmetic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_type='valid'):
        super(_ConvolutionArithmetic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_type = padding_type
        self.stride = stride

        weights = torch.Tensor(out_channels, in_channels, *kernel_size).to(torch.device("mps"))
        self.weights = nn.Parameter(weights)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

        bound = 1 / np.sqrt(self.out_channels)
        bias = torch.Tensor(self.out_channels).to(torch.device("mps"))
        self.bias = nn.Parameter(bias)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def padding_2d(self, input, kernel, stride, padding_type='valid'):
        """
        Using zero padding mode, i.e., creating desired size for output
        """

        output = None

        if padding_type == 'valid':
            return input, input.shape
        elif padding_type == 'same':
            if stride == 1:
                vertical_padding_size = int(np.floor(kernel.shape[1] / 2))
                horizontal_padding_size = int(np.floor(kernel.shape[2] / 2))
                for c in range(input.shape[0]):
                    pad = (vertical_padding_size, horizontal_padding_size, vertical_padding_size, horizontal_padding_size)
                    if not hasattr(output, 'shape'):
                        output = nn.functional.pad(input, pad, "constant", 0).to(torch.device("mps"))
                    else:
                        torch.cat((output, nn.functional.pad(input, pad, "constant", 0).to(torch.device("mps"))), dim=0)

                return output, output.shape
            
            elif stride > 1:
                print("Non-unit strides not supported at this time")
                return None, None

    def channel_convolution_2D(self, input, kernel, stride=1):
        """
        Perform Convolution for all channels of the input for the given filter
        """
        if stride != 1:
            print('Non-unit stride not supported for convolution layer')
            return 

        w_o = (input.shape[2] - kernel.shape[2]) + 1
        conv_output = torch.zeros((input.shape[0], w_o, w_o))

        conv_width = input.shape[3]
        conv_height = input.shape[2]

        i = 0
        j = 0

        while (j + (kernel.shape[2])) <= conv_width:
            j_end = j + (kernel.shape[2])
            while (i + (kernel.shape[2])) <= conv_height:
                i_end = i + (kernel.shape[2])
                input_conv_tensor = input[:, :, j:j_end, i:i_end].to(torch.device("mps"))
                prod = input_conv_tensor * kernel
                conv_prod_sum = prod.sum((3, 2, 1))
                conv_output[:, i, j] = conv_prod_sum
                i += 1
            j += stride
            i = 0
        return conv_output

    def perform_convolution_2D(self, input, kernel, stride):
        # Modifying the input as needed
        input, in_shape = self.padding_2d(input, self.weights, stride=stride, padding_type=self.padding_type)
        w_o = (input.shape[3] - kernel.shape[2]) + 1
        conv_output = []

        for c in range(kernel.shape[0]):
            print(f"Applying filter {c+1}")
            conv_output.append(self.channel_convolution_2D(input, kernel[c, :, :, :], stride=stride))
        
        conv_output = torch.stack(conv_output).to(torch.device("mps"))
        print(f"Output Shape: {conv_output.shape}")
        return conv_output

    def forward(self, x):
        x = self.perform_convolution_2D(x, self.weights, stride=1)
        return x

class _AttentionArithmetic(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(_AttentionArithmetic, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = None

        # Defining the Wq, Wk, Wv matrices as a single linear layer:
        # Referencing how they are used individually as matrices (d_embed x d_embed), we will "concatenate" the matrices along dim=1
        self.attention_weights = LinearNeuralNetwork(self.embed_dim, 3 * self.embed_dim)

        # Defining Wo as a single linear layer
        self.multihead_attention_weights = LinearNeuralNetwork(self.embed_dim, self.embed_dim)

        # Dimensions of heads as defined
        self.head_dim = self.embed_dim // self.num_heads
    
    def forward(self, x: torch.Tensor, causal_mask=False):
        query: torch.Tensor
        key: torch.Tensor
        value: torch.Tensor

        input_shape = x.shape
        batch_size, sequence_length, embed_dim = input_shape

        intermediate_head_split_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)

        # Copying the input into query, key, value for the multihead input
        query, key, value = self.attention_weights(x).chunk(3, dim=-1)

        query = query.view(intermediate_head_split_shape).transpose(1, 2)
        key = key.view(intermediate_head_split_shape).transpose(1, 2)
        value = value.view(intermediate_head_split_shape).transpose(1, 2)

        # Computing the matrix multiplication for the arg of softmax (Q * K.transpose())
        softmax_arg = query @ key.transpose(-1, -2)

        if causal_mask:
            # Setting a mask of upper diagnol commponents (above the principal diagnol)
            mask = torch.ones_like(softmax_arg, dtype=torch.bool).triu(1)
            softmax_arg.masked_fill_(mask, -torch.inf)

        softmax_arg /= torch.sqrt(self.head_dim)

        softmax_output = nn.functional.softmax(softmax_arg, dim=-1)

        attention = softmax_output @ value

        attention = attention.transpose(1, 2)

        multihead_attention_input = attention.reshape(input_shape)

        output = self.multihead_attention_weights(multihead_attention_input)

        return output



    



