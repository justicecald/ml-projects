import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import Tensor, cat, empty, autograd
import matplotlib.pyplot as plt
from collections import *

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

        self.weights = np.random.rand(self.out_channels, self.in_channels, *kernel_size) * np.sqrt(2 / in_channels)
        self.weights = Tensor(self.weights)
        self.weights.requires_grad_()

        self.bias = np.zeros(self.out_channels) * np.sqrt(2 / in_channels)
        self.bias = Tensor(self.bias)
        self.bias.requires_grad_()
        
        print(f"Weight Shape: {self.weights.shape}")
    
    @property
    def state_dict(self):
        return OrderedDict({
            "weight": self.weights,
            "bias": self.bias
        })
    
    def padding_2d(self, input, kernel, stride, padding_type='valid'):
        """
        Using zero padding mode, i.e., creating desired size for output
        """
        print(f"(Padding Input): {input.shape}")

        output = []

        if padding_type == 'valid':
            return input, input.shape
        elif padding_type == 'same':
            if stride == 1:
                vertical_padding_size = int(np.floor(kernel.shape[0] / 2))
                horizontal_padding_size = int(np.floor(kernel.shape[1] / 2))
                for c in range(input.shape[0]):
                    output.append(np.pad(input[c, :, :], (vertical_padding_size, horizontal_padding_size), mode ='constant'))

                output = Tensor(np.array(output))

                print(f"Padding Dims: (vert) {vertical_padding_size} | (horiz) {horizontal_padding_size}")

                print(f"(Padding Output): {output.shape}")

                return output, output.shape
            
            elif stride > 1:
                print("Non-unit strides not supported at this time")
                return

    def channel_convolution_2D(self, input, kernel, stride=1):
        """
        Perform Convolution for all channels of the input for the given filter
        """
        if stride != 1:
            print('Non-unit stride not supported for convolution layer')
            return 

        w_o = (input.shape[1] - kernel.shape[1]) + 1
        conv_output = Tensor(np.zeros((w_o, w_o)))

        # print(f"Channel Convolutional Output Size: {conv_output.shape}")


        conv_width = input.shape[2]
        conv_height = input.shape[1]

        i = 0
        j = 0

        # print(f"Shapes: (i): {input.shape}, (k): {kernel.shape}")

        while (j + (kernel.shape[1])) <= conv_width:
            j_end = j + (kernel.shape[1])
            while (i + (kernel.shape[1])) <= conv_height:
                i_end = i + (kernel.shape[1])
                input_conv_tensor = Tensor(input[:, j:j_end, i:i_end])
                prod = input_conv_tensor * kernel
                conv_prod_sum = prod.sum()
                conv_output[i, j] = conv_prod_sum
                i += 1
            j += stride
            i = 0
        return conv_output

    def perform_convolution_2D(self, input, kernel, stride):
        # Modifying the input as needed
        input, in_shape = self.padding_2d(input, self.weights[0], stride=stride, padding_type=self.padding_type)
        w_o = (input.shape[1] - kernel.shape[1]) + 1
        conv_output = Tensor(np.zeros((kernel.shape[0], w_o, w_o)))

        for c in range(kernel.shape[0]):
            conv_output[c, :, :] = self.channel_convolution_2D(input, kernel[c, :, :, :], stride=stride)

        print(f"Output Shape: {conv_output.shape}")
        return conv_output

    def forward(self, x):
        return self.perform_convolution_2D(x, self.weights, stride=1)
    
    def backward(self, output):
        return output.backward(gradient=Tensor(np.ones(tuple([d for d in output.shape]))))

class _AttentionArithmetic(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(_AttentionArithmetic, self).__init__()
        self.embed_dim = torch.Tensor([embed_dim])
        self.num_heads = torch.Tensor([num_heads])
        self.bias = None

        # Defining the Wq, Wk, Wv matrices as a single linear layer:
        # Referencing how they are used individually as matrices (d_embed x d_embed), we will "concatenate" the matrices along dim=1
        self.attention_weights = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        # Defining Wo as a single linear layer
        self.multihead_attention_weights = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Dimensions of heads as defined
        self.head_dim = self.embed_dim // self.num_heads
    
    def forward(self, x, causal_mask=False):
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



    



