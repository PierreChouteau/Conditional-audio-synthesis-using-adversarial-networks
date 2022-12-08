import torch
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from models.gansynth import GANSynth

import datasets.dataset as data
import datasets.helper

from configs import config
from torch.utils.tensorboard import SummaryWriter


# Définit le device sur lequel on va train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Permet de load toute la configuration de notre gan
model_config = config.load_config("./config.yaml")


# Possibilité de changer des paramètres de la config ici avec un parser par exmple...
# Il faudra certainement le coder pour gansynth...


###############################################
# LOAD DATA
###############################################
nsynth_dataset = data.NSynthDataset(root_dir=model_config.dataset.root_dir, usage = 'train')
train_loader = DataLoader(nsynth_dataset, batch_size=model_config.dataset.batch_size, shuffle=True, num_workers=2)

print(nsynth_dataset, train_loader)

# ###
# # A faire.... le dataloader ! + séparation des sources pour avoir que les instruments acoustique. 
# ###


# ###############################################
# # TB_WRITER: Initialisation + log model config 
# ###############################################
# writer = SummaryWriter("runs/"+ model_config.gan_config.model_name)
# writer.add_text('Generator parameters', str(model_config.generator_config))
# writer.add_text('Discriminator parameters', str(model_config.discriminator_config))
# writer.add_text('Gan_training parameters', str(model_config.gan_config))
# writer.add_text('batch_size', str(model_config.batch_size))


# # Save le config file pour pouvoir le rouvrir par la suite (save dans le dossier de logs runs/model_name)
# # Pour Gansynth il faudra faire une fusion entre le folder 'runs' et trained_model 
# config_path = "runs/"+model_config.gan_config.model_name
# config_name = config_path + "/" + model_config.gan_config.model_name + "_train_config.yaml"
# config.save_config(model_config, config_name)


# ###############################################
# # Model Initialisation
# ###############################################
# generator = Generator(**model_config.generator_config.dict()).to(device)
# discriminator = Discriminator(**model_config.discriminator_config.dict()).to(device)
# gansynth = GANSynth(train_loader, generator, discriminator, writer, **model_config.gan_config.dict())
    

# ###############################################
# # Training
# ###############################################
# GANSynth.train_step()