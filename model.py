import torch
from torch import nn
import os
from dataset import train_loader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# Adding batchnorm is better to stabilize the training in both generator and discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, num_filters, num_classes):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_classes = num_classes
        
        in_model_channels = self.in_channels + self.num_classes

        self.label_embedding = nn.Embedding(self.num_classes, 28*28*self.num_classes) # Permet d'encoder les labels pour pouvoir les concaténer avec l'entrée du discriminateur... donc 10 classes dans 10*28*28 dimensions (correspond à une image...)
        self.model = nn.Sequential(
            nn.Conv2d(in_model_channels, self.num_filters, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2), # LeakyRelu pour le discriminateur
            nn.Dropout(0.3), # Pas de batchnorm en entrée de discriminator, cause d'instabilité de training
            
            nn.Conv2d(self.num_filters, 2 * self.num_filters, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2 * self.num_filters),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
                   
            nn.Flatten(),
            nn.Linear(7*7*2*self.num_filters, 1),
            nn.Sigmoid()            
        )

    def forward(self, x, labels):
        # x = x.view(x.size(0), 1, 28, 28)
        c = self.label_embedding(labels)
        c = c.view(-1, self.num_classes, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        output = self.model(x)
        return output
    
        
class Generator(nn.Module):
    def __init__(self, latent_dim, out_channels, num_filters, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.out_channels = out_channels
        input_dim = self.latent_dim + self.num_classes
        
        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes) # Correspond à une autre représentation du one_hot_encode: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
                                                                                # Permet d'encoder les labels de mnist dans un tenseur de 10 valeurs (matrice de 10x10 pour 10 classes => 0 = [0000000000])
                                                                                # => 0 = [1000000000])
                                                                                # => 1 = [0100000000])
                                                                                # ....
        self.fc = nn.Sequential(
          nn.Linear(input_dim, 7*7*self.num_filters*(2**2), bias=False),
          nn.BatchNorm1d(7*7*self.num_filters*(2**2)),
          nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters * (2**2), self.num_filters * 2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(self.num_filters * 2), # Les ajouts de batchnorm permettent de stabiliser le training
            nn.ReLU(), # simple ReLu pour le generateur
            
            nn.ConvTranspose2d(self.num_filters * 2, self.num_filters, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            
            nn.ConvTranspose2d(self.num_filters, self.out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False), # Pas de batchnorm en sortie de generator, cause d'instabilité sinon
            nn.Tanh() # Tanh utile car les images du dataset sont normalisées entre [-1 ; 1].
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x,c], 1)
        output = self.fc(x)
        output = output.view(-1, self.num_filters * (2**2), 7, 7)
        output = self.model(output)
        return output
    
    
    
class MSGAN(nn.Module):
    def __init__(self, generator, discriminator, lr_g, lr_d, checkpoint_path, loss_function, num_epochs=100):
        super(MSGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.checkpoint_path = './trained_model/'+checkpoint_path+'.pt'
        self.writer_path = 'runs/' + checkpoint_path
        self.num_epochs = num_epochs
        self.loss_function = loss_function
    
    def load_checkpoint(self):
        optimizer_generator, optimizer_discriminator = self.configure_optimizer()
        if os.path.isfile(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path)
            self.discriminator.load_state_dict(ckpt['discriminator'])
            self.generator.load_state_dict(ckpt['generator'])

            optimizer_discriminator.load_state_dict(ckpt['optimizer_disc'])
            optimizer_generator.load_state_dict(ckpt['optimizer_gen'])

            start_epoch = ckpt['epoch']
            print('model parameters loaded from '+self.checkpoint_path)
        else:
            start_epoch = 0
            print('new model')
            
        return optimizer_discriminator, optimizer_generator, start_epoch
            
    def configure_optimizer(self):
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        return optimizer_generator, optimizer_discriminator
            
    def train_step(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer_discriminator, optimizer_generator, start_epoch = self.load_checkpoint() 
        print('Optimizers, ok')
        writer = SummaryWriter(self.writer_path)
        print('Tb_Writer, ok')
        
        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            for n, (real_samples, mnist_labels) in enumerate(train_loader):
                ##############################
                ## update the discriminator
                ##############################
                batch_size = real_samples.size(0)
                noise = torch.randn((batch_size, 100))
                noise = noise.to(device)

                real_samples = real_samples.to(device)
                mnist_labels = mnist_labels.to(device)
                
                # zero the parameter gradients
                optimizer_discriminator.zero_grad()
                
                # forward du generator, creation d'un batch de fake samples (noise, puis passage dans generateur)
                fake_samples = self.generator(noise, mnist_labels)
                
                # forward du discriminator
                disc_real_output = self.discriminator(real_samples, mnist_labels)
                disc_fake_output = self.discriminator(fake_samples.detach(), mnist_labels) # on detach fake_samples, car on n'a pas besoin d'avoir accès au gradient du generateur
                
                # calculate the loss for the discriminator
                loss_discriminator = 1/2 * (self.loss_function(disc_real_output, torch.ones_like(disc_real_output)) + self.loss_function(disc_fake_output, torch.zeros_like(disc_fake_output)))
                
                # calculate the gradient for the discriminator
                loss_discriminator.backward()  
                
                # update the discriminator first
                optimizer_discriminator.step()
                
                ##############################
                ## update the Generator
                ##############################
                
                # zero the parameter gradients
                optimizer_generator.zero_grad()

                # forward du discriminator
                # on ne detach pas fake_samples, car on veut garder les gradient pour pouvoir entrainer le generateur
                disc_fake_output = self.discriminator(fake_samples, mnist_labels)
                
                # calculate the loss for the generator
                loss_generator = self.loss_function(disc_fake_output, torch.ones_like(disc_fake_output),)
                
                # calculate the gradient for the discriminator
                loss_generator.backward() 
                
                # update the generator
                optimizer_generator.step()

                # Show loss
                if n == batch_size - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                    
                    writer.add_scalar("Loss/Discriminator_train", loss_discriminator, epoch)
                    writer.add_scalar("Loss/Generator_train", loss_generator, epoch)
                    writer.flush()
                    
                    latent_space_samples = torch.randn(batch_size, 100).to(device)
                    generated_samples = self.generator(latent_space_samples, mnist_labels)
                    generated_samples = generated_samples.cpu().detach()
                    
                    figure = plt.figure()
                    for i in range(4):
                        ax = plt.subplot(1, 4, i+1)
                        img = plt.imshow(generated_samples[i].reshape(28, 28), cmap='gray_r')
                        plt.title('label: ' + str(mnist_labels[i].cpu().detach().numpy()))
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.show()
                  
                    writer.add_figure('4_mnist_images', figure, epoch)
                    writer.flush()

                    # Save checkpoint if the model (to prevent training problem)
                    checkpoint = {
                        'epoch': epoch + 1,
                        'generator': self.generator.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'optimizer_gen': optimizer_generator.state_dict(),
                        'optimizer_disc': optimizer_discriminator.state_dict()
                    }
                    torch.save(checkpoint, self.checkpoint_path)