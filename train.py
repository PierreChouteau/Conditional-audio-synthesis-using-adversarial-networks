import torch
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from models.gansynth import GANSynth

import datasets.dataset as data

from configs import config
from torch.utils.tensorboard import SummaryWriter

import numpy as np


# Defines the device on which we will train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load all the configuration of our gan
config_filepath = "./default_config.yaml"

print(f"Loading config file: {config_filepath}")
model_config = config.load_config(config_filepath)


#########################################################################################################
# LOAD DATA
#########################################################################################################
if model_config.dataset.maxi == 1:
    custom_dataset = data.CustomDataset(
        root_dir=model_config.dataset.root_dir,
        resample_rate=model_config.dataset.resample_rate,
        signal_duration=model_config.dataset.signal_duration,
    )
    print("Dataset loaded without normalisation")

    # Test of the max of the dataset
    print("Research of the max in progress...")
    maxi = 0
    for i, (_, _, melspec_log) in enumerate(custom_dataset):
        abso = np.abs(melspec_log.numpy())
        max1 = np.amax(abso[0])
        maxi = max(maxi, max1)

    model_config.dataset.maxi = float(maxi)


print("Max value of the Melspec is " + str(model_config.dataset.maxi))
custom_dataset = data.CustomDataset(
    root_dir=model_config.dataset.root_dir,
    resample_rate=model_config.dataset.resample_rate,
    signal_duration=model_config.dataset.signal_duration,
    maxi=model_config.dataset.maxi,
)
print("Normalised dataset loaded")


train_loader = DataLoader(
    custom_dataset,
    batch_size=model_config.dataset.batch_size,
    shuffle=True,
    num_workers=0,
)
print("Dataloader is ready to go...")


#########################################################################################################
# TB_WRITER: Initialisation + log model config
#########################################################################################################
writer = SummaryWriter("runs/" + model_config.gan_config.model_name)
writer.add_text("Generator parameters", str(model_config.generator_config))
writer.add_text("Discriminator parameters", str(model_config.discriminator_config))
writer.add_text("Gan_training parameters", str(model_config.gan_config))
writer.add_text("dataset", str(model_config.dataset))


# Save the config file to reopen it later (save in the logs folder runs/model_name)
config_path = "runs/" + model_config.gan_config.model_name
config_name = (
    config_path + "/" + model_config.gan_config.model_name + "_train_config.yaml"
)
print(
    f"Saving train_config file: {model_config.gan_config.model_name + '_train_config.yaml'}"
)
config.save_config(model_config, config_name)


#########################################################################################################
#Initialization of the model
#########################################################################################################
generator = Generator(**model_config.generator_config.dict()).to(device)
generator.initialize_weights()

discriminator = Discriminator(**model_config.discriminator_config.dict()).to(device)
discriminator.initialize_weights()

gansynth = GANSynth(
    train_loader,
    generator,
    discriminator,
    writer,
    maxi=model_config.dataset.maxi,
    **model_config.gan_config.dict(),
)


#########################################################################################################
# Training
#########################################################################################################
print("Training Started")
gansynth.train_step()
