# Overview

    # Create a Discriminator Class

    # Create a Genrator class

    # Load the Data 

    # Hyper paramenters

    # trainer

    # save the ouputs 


import torch.nn as nn
import torch
import torchvision
from torch.utils import data 
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        self.tanh = nn.Tanh()
    
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
train_dataset = datasets.MNIST('data/MNIST',download = True ,train=True, transform=train_transforms)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0


for epoch in range(num_epoch):
    for batch_idx, (data, _) in enumerate(train_loader):
        
        # Training the generator
        data = data.to(device_)
        fake_noise = torch.rand(batch_size, z_shape).to(device_)
        gen = generator(fake_noise)
        gen_optimizer.zero_grad()
        valid = torch.ones(batch_size).to(device_) 
        fake = torch.zeros(batch_size).to(device_)
        gen_loss = criterion(discremenator(gen), valid)
        gen_loss.backward()
        gen_optimizer.step()

        # Traininig the discremenator
        dis_optimizer.zero_grad()
        disc_real = discremenator(data)
        disc_real_loss = criterion(disc_real, valid)
        disc_gen = gen.detach()
        disc_gen_loss = criterion(disc_gen, fake)
        final_disc_loss = (disc_real_loss+disc_gen_loss)/2
        final_disc_loss.backward()
        dis_optimizer.step()


        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epoch}] Batch {batch_idx}/{len(train_loader)} \
                      Loss D: {final_disc_loss:.4f}, loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = gen.reshape(-1, 1, 28, 28)
                datau = data.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(datau, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1




        
 



