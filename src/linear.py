import pandas as pd
import numpy as np
from torch import nn, Tensor

from typing import Optional, List, Tuple, Union
from typing_extensions import deprecated
from collections import *

from arithmetic import *

class LinearNeuralNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearNeuralNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        weights = torch.Tensor(self.out_features, self.in_features)
        self.weights = nn.Parameter(weights)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)

        bound = 1 / np.sqrt(fan_in)
        bias = torch.Tensor(self.out_features)
        self.bias = nn.Parameter(bias)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return torch.mm(x, self.weights.t()) + self.bias