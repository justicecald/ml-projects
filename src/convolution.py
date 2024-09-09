import pandas as pd
import numpy as np
from torch import nn
from typing import Optional, List, Tuple, Union
from typing_extensions import deprecated
from arithmetic import _ConvolutionArithmetic

class ConvolutionalNeuralNetwork_2D(_ConvolutionArithmetic):

    def __init__(self, kernel_size, stride, channels, padding = "valid"):
        self.kernel_size = kernel_size
        self.channels = channels
        self.stride = stride
        self.padding = padding

    def convlutional_activation_layer(self):
        pass
    
    def linear_layer(self):
        pass

    def Conv2D(self):
        pass

    def reLu(self, v):
        return np.maximum(v, 0.)
