from pydantic import BaseModel
import hparams


class Generator_config(BaseModel):
    latent_dim: int
    out_channels: int
    num_filters: int
    num_classes: int
    kernel_size: int 
    

class Discriminator_config(BaseModel):
    in_channels: int
    num_filters: int
    num_classes: int 
    kernel_size: int
    dropout: float
    
    
class GAN_config(BaseModel):
    latent_dim: int
    lr_g: float
    lr_d: float
    checkpoint_path: str
    num_epochs: int
    add_loss: int
    add_figure: int
    save_ckpt: int
    
    
gen_conf = Generator_config(**hparams.Mnist_config['generator_params'])
disc_conf = Discriminator_config(**hparams.Mnist_config['discriminator_params'])
gan_conf = GAN_config(**hparams.Mnist_config['GAN_params'])