import pandas as pd
import numpy as np
from torch import nn
from sampling import _ConvlutionalSampling

class _ConvolutionArithmetic:
    """
    Referencing:
    - A guide to convolution arithmetic for deep learning (https://arxiv.org/pdf/1603.07285)
    """
    def __init__(self):
        pass

    def convolutional_padding(self, input, input_size, output_size, padding_type = 'valid'):
        """
        Using zero padding mode, i.e., creating desired size for output
        """
        if padding_type == 'valid':
            pass
        elif padding_type == 'same':
            pass
        
        output = None
        return output

    def channel_convolution_2D(self, input, stride):
        pass

    def perform_convolution_2D(self, input, stride, channels):
        pass