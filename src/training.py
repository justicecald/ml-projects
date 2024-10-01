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
import PIL
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from datasets import load_dataset
from diffusion import Diffusion, DiffusionModel

from tqdm.auto import tqdm

torch.set_default_device("mps")

class DiffusionModelTrainer:
    def __init__(self, epochs, num_mini_batches, hf_dataset):
        self.epochs = epochs
        self.n_mini_batches = num_mini_batches
        self.ds_name = hf_dataset
        self.load_ds(self.ds_name)
        self.model = DiffusionModel(1000)

    def load_ds(self, dataset_name: str):
        train = load_dataset(self.ds_name, split=f"train[:100]")
        print(len(train))
        self.train = self.transform_ds(train) # Getting us an array of images

        test = load_dataset(self.ds_name, split="test[:5]")
        self.test = self.transform_ds(test)
    
    def train_model(self) -> None:
        # Define training flow
        train = self.train
        test = self.test
        optimizer = torch.optim.Adam(self.model.ddpm.parameters(), lr=2e-4)
        inc = torch.floor(train.shape[0] / self.n_mini_batches)
        num_mini_batches = train.shape[0] // inc
        test_set = test

        for e in range(self.epochs):
            for m in range(0, num_mini_batches):
                X_Train = train[(inc * m):(inc * (m+1)), :, :, :]
                print(f"X_Train Mini Batch [{m}] Size: {X_Train.shape}")
                loss = self.model.compute_loss(X_Train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if m % 5 == 0:
                    print(f"Epoch [{e + 1}]: Mini Batch [{m + 1}]: Loss == {loss}")
            
            if train.size[0] % inc != 0: # If the size of the mini batches don't completely cover the training set, do one more step
                X_Train = train[(inc * num_mini_batches):, :, :, :]
                print(f"X_Train Mini Batch [{m}] Size: {X_Train.shape}")

                loss = self.model.compute_loss(X_Train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Epoch [{e + 1}]: Mini Batch [{num_mini_batches + 1}]: Loss == {loss}")

    def transform_ds(self, dataset):
        transformed_train_dataset = None
        for i in range(len(dataset)):
            url = dataset[i]["url"]
            print(f"Attempting to download and process image [{i + 1}] | {url} ...")
            size = 512
            try:
                image = Image.open(requests.get(url, stream=True).raw)
                transform = Compose([
                    Resize(size),
                    CenterCrop(size),
                    ToTensor(), # turn into torch Tensor of shape CHW, divide by 255       
                ])

                transformed_image = transform(image).unsqueeze(0)
                print(f"Image [{i + 1}] Has Dim: {transformed_image.shape}")

                if i == 0 or not hasattr(transformed_train_dataset, 'shape'):
                    transformed_train_dataset = transformed_image
                else:
                    transformed_train_dataset = torch.cat((transformed_train_dataset, transformed_image), dim=0)
                print(f"Successfully processed [{i + 1}] | {url} ...\n")

            except PIL.UnidentifiedImageError:
                print(f"Failed to download [{i + 1}] | {url} ...\n")
                continue
        
        print(f"Final size of dataset: {transformed_train_dataset.shape}")

        return transformed_train_dataset

