import pandas as pd
import numpy as np

from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
import matplotlib.pyplot as plt

class U_Net_Convolution:
    def __init__(self):
        pass

    @staticmethod
    def time_position_embedd(time_steps, time_embed_dim):
        denom_fact = 10000 ** (np.arange(start=0, stop=(time_embed_dim//2)) / (time_embed_dim // 2))
        time_embedding = time_steps[:, None].repeat(time_embed_dim//2, 1) / denom_fact

        print(f"Denominator Shape: {denom_fact.shape}\n")

        time_embedding = np.concatenate((np.sin(time_embedding), np.cos(time_embedding)), axis=1)
        print(f"Time Embedding Shape: {time_embedding.shape}")

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(time_embedding)
        # ax.set_aspect('auto')
        # fig.savefig('equal.png')

        return time_embedding
    

# if __name__ == "__main__":
#     time_steps = np.linspace(start=1e-4, stop=0.002, num=1000)
#     time_embedding = U_Net_Convolution.time_position_embedd(time_steps=np.array(range(0, len(time_steps))), time_embed_dim=32)
    
