import torch
from models.model import Discriminator, Generator
from models.model import LSGAN
from datasets.dataset import train_loader
import hparams
from configs.config import gen_conf, disc_conf, gan_conf
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definition of the tb_writer, and directly add the parameters of training
writer = SummaryWriter("runs/"+ hparams.Mnist_config['GAN_params']['checkpoint_path'])
writer.add_text('Generator parameters', str(gen_conf))
writer.add_text('Discriminator parameters', str(disc_conf))
writer.add_text('Gan_training parameters', str(gan_conf))


# creation du generateur et du discriminateur
generator = Generator(**gen_conf.dict()).to(device)
discriminator = Discriminator(**disc_conf.dict()).to(device)
lsgan = LSGAN(train_loader, generator, discriminator, writer, **gan_conf.dict())

# Start the training
lsgan.train_step()