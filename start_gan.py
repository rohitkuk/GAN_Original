# Overview

    # Create a Discriminator Class

    # Create a Genrator class

    # Load the Data 

    # Hyper paramenters

    # trainer

    # save the ouputs 


import torch.nn as nn
import torch 
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

class Discremenator(nn.Module):
    def __init__(self, img_size):
        super(Discremenator, self).__init__()
        self.FC1 = nn.Linear(img_size, 512)
        self.FC2 = nn.Linear(512, 256)
        self.FC2 = nn.Linear(256, 128)
        self.lek_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.lek_relu(self.FC1(x))
        out = self.lek_relu(self.FC2(out))
        out = self.lek_relu(self.FC3(out))
        return self.sigmoid(out)

