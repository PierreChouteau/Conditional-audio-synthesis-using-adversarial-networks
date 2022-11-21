import torch
from torch import nn
from model import Discriminator, Generator
from model import MSGAN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# creation du generateur et du discriminateur
generator = Generator(latent_dim=100, out_channels=1, num_filters=64, num_classes=10).to(device)
discriminator = Discriminator(in_channels=1, num_filters=64, num_classes=10).to(device)
print('discriminator and generator loaded')

# Paramètre pour le training
lr_g = 1e-4
lr_d = 1e-4
num_epochs = 100
loss_function = nn.MSELoss() # LSGAN loss

# checkpoint name
ckpt_path = 'test'

msgan = MSGAN(generator, discriminator, lr_g, lr_d, checkpoint_path=ckpt_path, loss_function=loss_function, num_epochs=num_epochs).to(device)

# Permet de démarer l'entrainement du GAN
print('Training Started')
msgan.train_step()