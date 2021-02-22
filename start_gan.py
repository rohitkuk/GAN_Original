# Overview

    # Create a Discriminator Class

    # Create a Genrator class

    # Load the Data 

    # Hyper paramenters

    # trainer

    # save the ouputs 


from torch._C import device
import torch.nn as nn
import torch
from torch.utils import data 
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


class Discremenator(nn.Module):
    def __init__(self, img_size):
        super(Discremenator, self).__init__()
        self.FC1 = nn.Linear(img_size, 512)
        self.FC2 = nn.Linear(512, 256)
        self.FC2 = nn.Linear(256, 128)
        self.lek_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.lek_relu(self.FC1(x))
        out = self.lek_relu(self.FC2(out))
        out = self.lek_relu(self.FC3(out))
        return self.sigmoid(out)


class Generator(nn.Module):
    def __init__(self, z_shape, img_size):
        super(Generator, self).__init__()
        self.FC1 = nn.Linear(z_shape, 512)
        self.FC2 = nn.Linear(512, 256)
        self.FC2 = nn.Linear(256, img_size)
        self.lek_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.tanh()
    
    def forward(self, x):
        out = self.lek_relu(self.FC1(x))
        out = self.lek_relu(self.FC2(out))
        out = self.lek_relu(self.FC3(out))
        return self.tanh(out)


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


# HyperParamters and Some Constants
img_size   = 1*28*28
z_shape    = 1*100
num_epoch  = 50
lr         = 2e-4
batch_size = 32
device_ = 'cuda' if torch.cuda.is_available() else 'cpu'


# Instancitiating the Models
discremenator = Discremenator(img_size).to(device_)
generator = Generator(z_shape, img_size).to(device_)


# Defining Optimizer and Criterion
gen_optimizer = torch.optim.Adam(params =generator.parameters(),  lr = lr )
dis_optimizer = torch.optim.Adam(params =discremenator.parameters(), lr = lr)
criterion = nn.BCELoss()

# There is no need to create a directory explicitily below checks if there else creates
train_dataset = datasets.MNIST('data', train=True, transform=train_transforms)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device_)
        
        gen = 
 



