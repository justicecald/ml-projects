import pandas as pd
import numpy as np
from torch import nn

class _ConvolutionArithmetic:
    """
    Referencing:
    - A guide to convolution arithmetic for deep learning (https://arxiv.org/pdf/1603.07285)
    """
    def __init__(self):
        pass

    @staticmethod
    def padding_2d(input, kernel, stride=1, padding_type='valid'):
        """
        Using zero padding mode, i.e., creating desired size for output
        """
        w_in = input.shape[1]
        h_in = input.shape[0]
        s = stride

        print(f"(Padding Input): {input.shape}")

        if padding_type == 'valid':
            return input
        elif padding_type == 'same':
            if stride == 1:
                vertical_padding_size = int(np.floor(kernel.shape[0] / 2))
                horizontal_padding_size = int(np.floor(kernel.shape[1] / 2))
                output = np.pad(input, (vertical_padding_size, horizontal_padding_size), mode ='constant')

                print(f"Padding Dims: (vert) {vertical_padding_size} | (horiz) {horizontal_padding_size}")

                print(f"(Padding Output): {output.shape}")

                return output
            
            elif stride > 1:
                print("Non-unit strides not supported at this time")
                return

    @staticmethod
    def channel_convolution_2D(input, kernel, stride=1, padding=0):
        if stride != 1:
            print('Non-unit stride not supported for convolution layer')
            return 

        w_o = (input.shape[0] - kernel.shape[0]) + 1
        conv_output = np.zeros((w_o, w_o))

        print(f"Convolutional Output Size: {conv_output.shape}")


        conv_width = input.shape[1]
        conv_height = input.shape[0]

        i = 0
        j = 0

        print(f"Shapes: (i): {input.shape}, (k): {kernel.shape}, (o): {conv_output.shape}")

        while (j + (kernel.shape[1])) <= conv_width:
            j_end = j + (kernel.shape[1])
            while (i + (kernel.shape[1])) <= conv_height:
                i_end = i + (kernel.shape[1])
                conv_output[i][j] = np.sum(input[j:j_end, i:i_end] * kernel)
                i += stride
            j += stride
            i = 0

        return conv_output

    def perform_convolution_2D(self, input, stride, channels):
        pass

# if __name__ == '__main__':
#     l = np.random.rand(5, 5)
#     k = np.random.rand(3, 3)

#     l_padded = _ConvolutionArithmetic.padding_2d(l, k)

#     print(_ConvolutionArithmetic.channel_convolution_2D(l_padded, k, padding=((l_padded.shape[0] - l.shape[0]) / 2)))


