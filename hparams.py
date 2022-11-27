# Dataset params
BATCH_SIZE = 32


# Generator
OUT_CHANNELS_GEN = 1
NUM_FILTERS_GEN = 64
NUM_CLASSES_GEN = 10
KERNEL_SIZE_GEN = 5


# Discriminator
IN_CHANNELS_DISC = 1
NUM_FILTERS_DISC = 64
NUM_CLASSES_DISC = 10
KERNEL_SIZE_DISC = 5
DROPOUT_DISC = 0.3


# Parameters for the training
LATENT_DIM = 100
LR_GEN = 1e-4
LR_DISC = 1e-4
NUM_EPOCHS = 100
CKPT_PATH = 'test'
ADD_LOSS_TB = BATCH_SIZE - 1
ADD_FIGURE_TB = BATCH_SIZE - 1
SAVE_CKPT = BATCH_SIZE - 1


Mnist_config = {
    'generator_params': {
        'latent_dim': LATENT_DIM,
        'out_channels': OUT_CHANNELS_GEN,
        'num_filters': NUM_FILTERS_GEN,
        'num_classes': NUM_CLASSES_GEN,
        'kernel_size': KERNEL_SIZE_GEN,
    },
    
    'discriminator_params': {
        'in_channels': IN_CHANNELS_DISC,
        'num_filters': NUM_FILTERS_DISC, 
        'num_classes': NUM_CLASSES_DISC,
        'kernel_size': KERNEL_SIZE_DISC,
        'dropout': DROPOUT_DISC,
    },
    
    'GAN_params': {
        'latent_dim': LATENT_DIM,
        'lr_g': LR_GEN,
        'lr_d': LR_DISC,
        'checkpoint_path': CKPT_PATH,
        'num_epochs': NUM_EPOCHS,
        'add_loss': ADD_LOSS_TB,
        'add_figure': ADD_FIGURE_TB,
        'save_ckpt': SAVE_CKPT,
    }
}
