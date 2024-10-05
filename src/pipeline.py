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
from training import DiffusionModelTrainer

from tqdm.auto import tqdm

torch.set_default_device("mps")

if __name__ == "__main__":
    model_trainer = DiffusionModelTrainer(10, 10, hf_dataset="dalle-mini/open-images")
    model_trainer.train_model()