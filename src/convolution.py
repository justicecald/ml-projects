import pandas as pd
import numpy as np
from torch import nn
from typing import Optional, List, Tuple, Union
from typing_extensions import deprecated
from arithmetic import *

class ConvolutionalNeuralNetwork_2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(ConvolutionalNeuralNetwork_2D, self).__init__()
        self.kernel_size = kernel_size
        self.channels = out_channels
        self.stride = stride
        self.padding = padding

    def convlutional_activation_layer(self):
        pass

    def Conv2D(self):
        pass

    def reLu(self, v):
        return np.maximum(v, 0.)
    
    def forward(self, x):
        pass
