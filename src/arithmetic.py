import pandas as pd
import numpy as np
from torch import nn
from torch import Tensor, cat, empty, autograd

from collections import *

"""
Referencing:
- A guide to convolution arithmetic for deep learning (https://arxiv.org/pdf/1603.07285)
"""
class _ConvolutionArithmetic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_type='valid'):
        super(_ConvolutionArithmetic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_type = padding_type
        self.stride = stride

        self.weights = np.random.rand(self.out_channels, self.in_channels, *kernel_size) * np.sqrt(2 / in_channels)
        self.weights = Tensor(self.weights).requires_grad_()

        self.bias = np.zeros(self.out_channels) * np.sqrt(2 / in_channels)
        self.bias = Tensor(self.bias).requires_grad_()
        
        print(f"Weight Shape: {self.weights.shape}")

        print(f"Grads Required:\nWeights: {self.weights.requires_grad}\nBias: {self.bias.requires_grad}")
    
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
        w_in = input.shape[1]
        h_in = input.shape[0]
        s = stride

        print(f"(Padding Input): {input.shape}")

        output = []

        if padding_type == 'valid':
            return input
        elif padding_type == 'same':
            if stride == 1:
                vertical_padding_size = int(np.floor(kernel.shape[0] / 2))
                horizontal_padding_size = int(np.floor(kernel.shape[1] / 2))
                for c in range(input.shape[-1]):
                    output.append(np.pad(input[:, :, c], (vertical_padding_size, horizontal_padding_size), mode ='constant'))

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

        print(f"Channel Convolutional Output Size: {conv_output.shape}")


        conv_width = input.shape[2]
        conv_height = input.shape[1]

        i = 0
        j = 0

        print(f"Shapes: (i): {input.shape}, (k): {kernel.shape}")

        while (j + (kernel.shape[1])) <= conv_width:
            j_end = j + (kernel.shape[1])
            while (i + (kernel.shape[1])) <= conv_height:
                i_end = i + (kernel.shape[1])
                conv_prod = Tensor.sum(Tensor(input[:, j:j_end, i:i_end]) * kernel)
                conv_output[i][j] = Tensor.sum(conv_prod)
                i += 1
            j += stride
            i = 0
        print(f"Convolution Output Requires Grad: {conv_output.requires_grad}")
        return conv_output

    def perform_convolution_2D(self, input, kernel, stride):
        # Modifying the input as needed
        input, in_shape = self.padding_2d(input, self.weights[0], stride=stride, padding_type=self.padding_type)
        w_o = (input.shape[1] - kernel.shape[1]) + 1
        conv_output = Tensor(np.zeros((kernel.shape[0], w_o, w_o)))

        for c in range(kernel.shape[0]):
            conv_output[c, :, :] = Tensor(self.channel_convolution_2D(input, kernel[c, :, :, :], stride=stride))

        # conv_output.requires_grad_()
        print(f"Output Shape: {conv_output.shape}")
        return conv_output

    def forward(self, x):
        return self.perform_convolution_2D(x, self.weights, stride=1)

if __name__ == '__main__':
    l = np.random.rand(5, 5, 3)
    y = Tensor(np.random.rand(6, 5, 5))
    a = _ConvolutionArithmetic(3, 6, (3,3), 1, padding_type='same')
    a(l)
    



