import pandas as pd
import numpy as np
from torch import nn

from typing import Optional, List, Tuple, Union
from typing_extensions import deprecated

from arithmetic import _ConvolutionArithmetic


from arithmetic import *

class ConvolutionalNeuralNetwork_2D(_ConvolutionArithmetic):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_type='valid'):
        super(ConvolutionalNeuralNetwork_2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding_type=padding_type)