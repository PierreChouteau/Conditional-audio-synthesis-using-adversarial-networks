import torch
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from models.gansynth import GANSynth

import datasets.dataset as data

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import hydra
from omegaconf import DictConfig
import logging


@hydra.main(config_path="configs", config_name="default_config", version_base="1.3")
def train(cfg: DictConfig):
    
    # Define Logger - TODO: Change print to logger
    log = logging.getLogger(__name__)
    
    # Defines the device on which we will train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #########################################################################################################
    # LOAD DATA
    #########################################################################################################
    if cfg.dataset.maxi == 1:
        custom_dataset = data.CustomDataset(
            root_dir=cfg.dataset.root_dir,
            resample_rate=cfg.dataset.resample_rate,
            signal_duration=cfg.dataset.signal_duration,
        )
        log.info("Dataset loaded without normalisation")

        # Test of the max of the dataset
        log.info("Research of the max in progress...")
        maxi = 0
        for i, (_, _, melspec_log) in enumerate(custom_dataset):
            abso = np.abs(melspec_log.numpy())
            max1 = np.amax(abso[0])
            maxi = max(maxi, max1)

        cfg.dataset.maxi = float(maxi)


    log.info("Max value of the Melspec is " + str(cfg.dataset.maxi))
    custom_dataset = data.CustomDataset(
        root_dir=cfg.dataset.root_dir,
        resample_rate=cfg.dataset.resample_rate,
        signal_duration=cfg.dataset.signal_duration,
        maxi=cfg.dataset.maxi,
        device=device,
    )
    log.info("Normalised dataset loaded")


    train_loader = DataLoader(
        custom_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=0,
    )
    log.info("Dataloader is ready to go...")


    #########################################################################################################
    # TB_WRITER: Initialisation + log model config
    #########################################################################################################
    writer = SummaryWriter("runs/" + cfg.gan_config.model_name)
    writer.add_text("Generator parameters", str(cfg.generator_config))
    writer.add_text("Discriminator parameters", str(cfg.discriminator_config))
    writer.add_text("Gan_training parameters", str(cfg.gan_config))
    writer.add_text("dataset", str(cfg.dataset))


    #########################################################################################################
    # Initialization of the model
    #########################################################################################################
    generator = Generator(**cfg.generator_config).to(device)
    generator.initialize_weights()

    discriminator = Discriminator(**cfg.discriminator_config).to(device)
    discriminator.initialize_weights()

    gansynth = GANSynth(
        train_loader,
        generator,
        discriminator,
        writer,
        logger=log,
        device=device,
        maxi=cfg.dataset.maxi,
        **cfg.gan_config,
    )


    #########################################################################################################
    # Training
    #########################################################################################################
    log.info("Training Started")
    gansynth.train_step()


if __name__ == "__main__":
    train()