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
        self.weights = np.random.rand(self.out_features, self.in_features) * np.sqrt(2 / in_features)
        self.bias = np.random.rand(self.out_features) * np.sqrt(2 / in_features)

        print(f"Dimensions:\n- In Features: {self.in_features}\n- Out Features: {self.out_features}\n- Weights: {self.weights.shape}\n- Bias: {self.bias.shape}")

    def state_dict(self):
        return OrderedDict({
            "weight": Tensor(self.weights),
            "bias": Tensor(self.bias)
        })
    
    def forward(self, x):
        return Tensor(
            np.dot(self.weight, x) + self.bias
        )

# if __name__ == "__main__":
#     ll = LinearNeuralNetwork(5, 3)
#     ll2 = nn.Linear(5, 3)
#     print(ll.state_dict())
#     print(ll2.state_dict())