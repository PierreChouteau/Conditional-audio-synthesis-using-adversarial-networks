import torch
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from models.gansynth import GANSynth

import datasets.dataset as data

from configs import config
from torch.utils.tensorboard import SummaryWriter

import numpy as np


# Définit le device sur lequel on va train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Permet de load toute la configuration de notre gan
config_filepath = "./default_config.yaml"

print(f"Loading config file: {config_filepath}")
model_config = config.load_config(config_filepath)


###############################################
# LOAD DATA
###############################################
if model_config.dataset.maxi == 1:
    nsynth_dataset = data.NSynthDataset(
        root_dir=model_config.dataset.root_dir,
        usage="train",
        filter_key="acoustic",
        resample_rate=model_config.dataset.resample_rate,
        signal_duration=model_config.dataset.signal_duration,
    )
    print("Dataset loaded without normalisation")

    # Test du max du dataset pour entier
    print("Research of the max in progress...")
    maxi = 0
    for i, (melspec_log_norm, label, melspec_log) in enumerate(nsynth_dataset):
        abso = np.abs(melspec_log.numpy())
        max1 = np.amax(abso[0])
        maxi = max(maxi, max1)

    model_config.dataset.maxi = float(maxi)

print("Max value of the Melspec is " + str(model_config.dataset.maxi))
nsynth_dataset = data.NSynthDataset(
    root_dir=model_config.dataset.root_dir,
    usage="train",
    filter_key="acoustic",
    resample_rate=model_config.dataset.resample_rate,
    signal_duration=model_config.dataset.signal_duration,
    maxi=model_config.dataset.maxi,
)
print("Normalised dataset loaded")


train_loader = DataLoader(
    nsynth_dataset,
    batch_size=model_config.dataset.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)
print("Dataloader is ready to go...")


###############################################
# TB_WRITER: Initialisation + log model config
###############################################
writer = SummaryWriter("runs/" + model_config.gan_config.model_name)
writer.add_text("Generator parameters", str(model_config.generator_config))
writer.add_text("Discriminator parameters", str(model_config.discriminator_config))
writer.add_text("Gan_training parameters", str(model_config.gan_config))
writer.add_text("dataset", str(model_config.dataset))


# Save le config file pour pouvoir le rouvrir par la suite (save dans le dossier de logs runs/model_name)
# Pour Gansynth il faudra faire une fusion entre le folder 'runs' et trained_model
config_path = "runs/" + model_config.gan_config.model_name
config_name = (
    config_path + "/" + model_config.gan_config.model_name + "_train_config.yaml"
)
print(
    f"Saving train_config file: {model_config.gan_config.model_name + '_train_config.yaml'}"
)
config.save_config(model_config, config_name)


###############################################
# Model Initialisation
###############################################
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


################################################
# Training
################################################
print("Training Started")
gansynth.train_step()
