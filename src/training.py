import numpy as np
import torch
import requests
import matplotlib.pyplot as plt

from torch import nn
from convolution import ConvolutionalNeuralNetwork_2D
from arithmetic import _AttentionArithmetic
from linear import LinearNeuralNetwork
from modules import time_position_embedding
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from datasets import load_dataset
from diffusion import Diffusion

from tqdm.auto import tqdm

def load_image_datasets() -> torch.Tensor:
    # load dataset from the hub
    dataset = load_dataset(
        "dalle-mini/open-images", 
        split=f"train[:100]"
    )
    transformed_train_dataset = torch.zeros((100, 3, 512, 512))
    for i in range(len(transformed_train_dataset)):
        url = dataset[i]["url"]
        size = int(dataset[i]["width"])

        image = Image.open(requests.get(url, stream=True).raw)
        transform = Compose([
            Resize(size),
            CenterCrop(size),
            ToTensor(), # turn into torch Tensor of shape CHW, divide by 255       
        ])

        transformed_image = transform(image).unsqueeze(0)
        print(f"Training set shape: {transformed_image}")
        transformed_train_dataset[i] = transformed_image

    return transformed_train_dataset

