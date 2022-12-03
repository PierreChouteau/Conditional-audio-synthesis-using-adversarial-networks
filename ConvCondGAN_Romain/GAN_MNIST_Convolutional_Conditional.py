# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import os

#--------------------------Cuda stuff

device = ''
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#--------------------------Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

batch_size = 32
# Load the training set
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Create a batched data loader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

#----------------------------Discriminateur

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        nb_class=10
        nb_ch=1
        nb_in=nb_ch+nb_class
        self.label_emb=nn.Embedding(nb_class,28*28*nb_class) # Vecteur avec des représentations graphiques de 28x28 pour chaque classe (10)
        

        
        
        self.model = nn.Sequential(
            nn.Conv2d(nb_in,64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64,128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(7*7*128,1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        x = x.view(x.size(0), 1, 28,28)
        #print(x.size(0))
        c=self.label_emb(labels)
        #print(c.size())
        c=c.view(x.size(0),10, 28,28)
        #print(c.size(0))
        x=torch.cat([x,c],dim=1)
        #x = x.view(-1, 784)
        output = self.model(x)
        #print(output.size(),"output")
        return output
    
discriminator = Discriminator().to(device=device)

#---------------------------------Générateur

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        nb_l=100
        nb_class=10
        nb_inG=nb_l+nb_class
        self.label_emb=nn.Embedding(nb_class, nb_class) #10 car pareil que nb_class ? Oui, (nb_class, nb_class)
        
        self.fc=nn.Sequential( #changement de dimension
                nn.Linear(nb_inG,7*7*256,bias=False),
                nn.BatchNorm1d(7*7*256),
                nn.ReLU()
            )
        
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(256,128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,1, kernel_size=5, stride=1, padding=2,output_padding=0),
            
            nn.Tanh()
        )

    def forward(self, x,labels):
        #x = x.view(x.size(0), 100)
        #print(x.size())
        c=self.label_emb(labels)
        x=torch.cat([x,c],1)
        output = self.fc(x)
        output = output.view(-1,256,7,7)
        #print(output.shape)
        output = self.model(output)
        #output = output.view(x.size(0), 1, 28, 28)
        #print(output.size())
        return output
    
generator = Generator().to(device=device)


#------------------------------Entraînement

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()#Risque de diverger avec de l'audio, il faudra changer
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

latent_space_samples_plot = torch.randn((16, 100)).to(device=device)

# Load trained NN when it exists, or train a new NN
if os.path.isfile('discriminator.pt') and os.path.isfile('generator.pt'):
    discriminator.load_state_dict(torch.load('./discriminator.pt'))
    generator.load_state_dict(torch.load('./generator.pt'))   
else:
    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader):
        
            #Data
            real_samples=real_samples.to(device=device)
            mnist_labels = mnist_labels.to(device)
            
            real_samples_labels=torch.ones((batch_size,1)).to(device=device)
            LS_samples=torch.randn((batch_size,100)).to(device=device)
            generated_samples=generator(LS_samples, mnist_labels)
            generated_samples_labels=torch.zeros((batch_size,1)).to(device=device)
            
            #all_samples=torch.cat((real_samples,generated_samples)) #En faire deux séparés (pas all samples), envoyer 2 fois dans le discriminateur et additionner les loss,
            #all_samples_labels=torch.cat((real_samples_labels,generated_samples_labels))

            #Discriminator
            discriminator.zero_grad()
            out_discr_real=discriminator(real_samples, mnist_labels)
            out_discr_gene=discriminator(generated_samples, mnist_labels)
            loss_discr_real=loss_function(out_discr_real,torch.ones_like(out_discr_real))
            loss_discr_gene=loss_function(out_discr_gene,torch.zeros_like(out_discr_gene))
            loss_discr=(1/2)*(loss_discr_real+loss_discr_gene)
            loss_discr.backward()
            optimizer_discriminator.step()

            #Generators
            LS_samples=torch.randn((batch_size,100)).to(device=device)
            generator.zero_grad()
            generated_samples=generator(LS_samples,mnist_labels)
            
            out_discr_gen=discriminator(generated_samples, mnist_labels)
            
            loss_gen=loss_function(out_discr_gen,torch.ones_like(out_discr_gen))
            loss_gen.backward()
            optimizer_generator.step()
            #print(n,batch_size)

            # Show loss
            if n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discr}")
                print(f"Epoch: {epoch} Loss G.: {loss_gen}")
                plt.figure(dpi=150)
                for i in range(16):
                    ax = plt.subplot(4, 4, i+1)
                    plt.imshow(generated_samples[i].reshape(28, 28), cmap='gray_r')
                    plt.xticks([])
                    plt.yticks([])
                plt.tight_layout()
                break
                

#-----------------------------Génération à partir du modèle entraîné
latent_space_samples = torch.randn(batch_size, 100).to(device=device)

generated_samples = generator(latent_space_samples,mnist_labels)
generated_samples = generated_samples.cpu().detach()

#---------------------------------Enregistrement entraînement

# Save trained NN parameters
torch.save(generator.state_dict(), 'generator.pt')
torch.save(discriminator.state_dict(), 'discriminator.pt')


plt.figure(dpi=150)
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()





