import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from hparams import BATCH_SIZE


# Permet de transformer la data en entrer
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])


# Load the training set
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)


# Create a batched data loader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)