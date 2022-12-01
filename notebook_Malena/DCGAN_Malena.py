#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:56:35 2022

@author: fouilloumalena
"""

"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn

#Discriminateur model
class Discriminator(nn.Module):
    
    def __init__(self, channels_img, features_d):
        
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            
            # entrées : N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1), # Sortie de taille 4x4 
          
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),nn.Sigmoid(),) #Taille 1x1 à la fin

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),)
            

    def forward(self, x):
        return self.disc(x)


#Generator model
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()


        self.net = nn.Sequential(
            # Entrée de taille 1x1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # taille 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # taille 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # taille 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # taille 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ), nn.Tanh(),) # Sortie de taille 64x64

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),)

    def forward(self, x):#, labels):
        return self.net(x)

#initialisation des poids
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success")
test() 



#####################################################


"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from models.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils



# Hyperparameters etc.
# Decide which device we want to run on
device = ''
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

LEARNING_RATE = 2e-4  
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 8
FEATURES_GEN = 8


transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.ToTensor(),transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),])

#Pour print() la loss à la fin + images
img_list = [] #Va contenir les images durant l'entraînement
G_losses = []
D_losses = []


# Téléchargement des données de MNIST
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,
                       download=True)


#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms) #Pour télécharger une autre base de données d'entraînement
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) #données
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device) #générateur
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device) #discriminateur

#initialisation des models
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss() #Définition de la loss

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=device) #Bruit fixé constant


gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Entraînement du discriminateur : max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) #D(x)
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) #D(G(z)) 
        loss_disc = (loss_disc_real + loss_disc_fake)/2 #Loss du discriminator  --> moyenne des deux
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Entraînement du générateur : min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output)) #Loss du générateur 
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print la loss occasionnellement, et affiche sur tensorboard
        if batch_idx % 100 == 0:
          print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \ Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
  
          with torch.no_grad():
              fake = gen(fixed_noise)

              # take out (up to) 32 examples
              img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
              img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

              writer_real.add_image("Real", img_grid_real, global_step=step)
              writer_fake.add_image("Fake", img_grid_fake, global_step=step)

          img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        # Save Losses 
        G_losses.append(loss_gen.item())
        D_losses.append(loss_disc.item())

        step += 1
