from pydantic import BaseModel
import yaml
import os


class Generator_config(BaseModel):
    latent_dim: int
    out_channels: int
    k_width: int
    k_heigth: int
    k_filters: int
    scale_factor: int
    

class Discriminator_config(BaseModel):
    in_channels: int
    out_channels: int
    k_width: int
    k_heigth: int
    k_filters: int
    scale_factor: int
    ksize_down: int
    stride_down: int
    
    
class GAN_config(BaseModel):
    latent_dim: int
    lr_g: float
    lr_d: float
    model_name: str
    num_epochs: int
    add_loss: int
    add_figure: int
    save_ckpt: int
    critic_iteration: int
    lambda_gp: int
    
    
class Dataset(BaseModel):
    batch_size: int
    root_dir: str
    maxi: float
    

class MainConfig(BaseModel):
    gan_config: GAN_config = None
    discriminator_config: Discriminator_config = None
    generator_config: Generator_config = None
    dataset: Dataset = None



def load_config(yaml_filepath="config.yaml"):
    with open(yaml_filepath, "r") as config_f:
        try:
            config_dict = yaml.safe_load(config_f)
            model_dict = {
                "gan_config": GAN_config(**config_dict["gan_config"]),
                "discriminator_config": Discriminator_config(**config_dict["discriminator_config"]),
                "generator_config": Generator_config(**config_dict["generator_config"]),
                "dataset": Dataset(**config_dict["dataset"]),
            }
            main_config = MainConfig(**model_dict)
            return main_config
        
        except yaml.YAMLError as e:
            print(e)


def save_config(main_config, config_name="train_config.yaml"):
    with open(os.path.join(config_name), 'w') as s:
        yaml.safe_dump(main_config.dict(), stream=s)