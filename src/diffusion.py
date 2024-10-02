import pandas as pd
import numpy as np
import torch
import math
import requests
from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
from model import *
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from datasets import load_dataset

torch.set_default_device("mps")

class DiffusionModel:
    def __init__(self, t_range):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.device = torch.device("mps")
        self.ddpm = Diffusion(self.device)

    def forward(self, x, ts) -> torch.Tensor:
        return self.ddpm(x, ts)

    def beta(self, ts):
        return self.beta_small + (ts / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, ts):
        return 1 - self.beta(ts)

    def alpha_bar(self, ts):
        # Product of alphas from 0 to t
        alpha_cum_prod_input = torch.Tensor([self.alpha(j) for j in torch.arange(start=0, end=ts)]).to(self.device)
        return torch.cumprod(alpha_cum_prod_input, dim=0)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        # Get a random time step for each image in the batch
        ts: torch.Tensor = torch.randint(low=0, high=self.t_range, size=(batch.shape[0],), device=self.device)
        noise_imgs = []
        # Generate noise, one for each image in the batch
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append((math.sqrt(a_hat[-1]) * batch[i]) + (math.sqrt(1 - a_hat[-1]) * epsilons[i]))

        noise_imgs = torch.stack(noise_imgs, dim=0)
        # Run the noisy images through the U-Net, to get the predicted noise
        e_hat = self.forward(noise_imgs, ts)
        # Calculate the loss, that is, the MSE between the predicted noise and the actual noise
        loss = nn.MSELoss(e_hat.reshape(-1, batch.size), epsilons.reshape(-1, batch.size))
        return loss

    def reverse_diffusion_step(self, x: torch.Tensor, t: torch.Tensor):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
            else:
                z = 0
            # Get the predicted noise from the U-Net
            e_hat = self.forward(x, t.view(1).repeat(x.shape[0]))
            # Perform the denoising step to take the image from t to t-1
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

class Diffusion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.time_embedding = time_position_embedding(np.array(range(0, 1000)), 32, self.device)
        self.unet = UnetContructor()
        self.final = UnetOutput(64, 3)
    
    def forward(self, x, ts):

        time = self.time_embedding[ts]
        
        output = self.unet(x, time)
        
        output = self.final(output)
        
        return output