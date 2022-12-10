import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os 

from datasets import dataset

import librosa

class GANSynth(nn.Module):
    def __init__(self, train_loader, generator, discriminator, writer, maxi, critic_iteration, lambda_gp, latent_dim, lr_g, lr_d, model_name, num_epochs, add_loss, add_figure, save_ckpt):
        super(GANSynth, self).__init__()
        self.train_loader = train_loader
        self.generator = generator
        self.critic = discriminator
        self.critic_iteration = critic_iteration
        self.lambda_gp = lambda_gp
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.trained_model_path = "./trained_model/" + model_name + ".pt"
        self.writer = writer
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        self.add_loss = add_loss
        self.add_figure = add_figure
        self.save_ckpt = save_ckpt
        self.maxi = maxi
        

    def load_checkpoint(self):
        optimizer_generator, optimizer_discriminator = self.configure_optimizer()
        if os.path.isfile(self.trained_model_path):
            ckpt = torch.load(self.trained_model_path)
            self.critic.load_state_dict(ckpt["discriminator"])
            self.generator.load_state_dict(ckpt["generator"])

            optimizer_discriminator.load_state_dict(ckpt["optimizer_disc"])
            optimizer_generator.load_state_dict(ckpt["optimizer_gen"])

            start_epoch = ckpt["epoch"]
            print("model parameters loaded from " + self.trained_model_path)
        else:
            start_epoch = 0
            print("new model")
            
        return optimizer_discriminator, optimizer_generator, start_epoch


    def configure_optimizer(self):
        optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g
        )
        optimizer_discriminator = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_d
        )
        return optimizer_generator, optimizer_discriminator
    
    
    def gradient_penalty(self, real, fake, device):
        alpha = torch.rand((real.size(0), 1, 1, 1)).to(device)
        interpolated_images = (real * alpha + fake * (1 - alpha)).requires_grad_(True)

        # Calculate critic scores
        mixed_scores = self.critic(interpolated_images)
        grad_outputs = torch.ones_like(mixed_scores, device=device, requires_grad=False)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient = gradient.view(gradient.size(0), -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty


    def train_step(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer_critic, optimizer_generator, start_epoch = self.load_checkpoint()
        print("Optimizers, ok")

        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            for n, (real_samples, _) in enumerate(self.train_loader):
                ##############################
                ## update the discriminator
                ##############################
                batch_size = real_samples.size(0)
                real_samples = real_samples.view(batch_size, real_samples.size(1), real_samples.size(3), real_samples.size(2))
                real_samples = real_samples.to(device)
                
                for _ in range(self.critic_iteration):
                    
                    noise = torch.randn(batch_size, self.latent_dim).to(device)
                    fake_samples = self.generator(noise)
                    
                    # zero the parameter gradients
                    optimizer_critic.zero_grad()
                    
                    critic_real = self.critic(real_samples)
                    critic_fake = self.critic(fake_samples.detach())
                    
                    gp = self.gradient_penalty(real_samples, fake_samples, device=device)
                    loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp * gp
                    )
                    
                    # calculate the loss for the discriminator/critic
                    loss_critic.backward(retain_graph=True)
                    # update the discriminator/critic
                    optimizer_critic.step()

                ##############################
                ## update the Generator
                ##############################
                # zero the parameter gradients
                optimizer_generator.zero_grad()

                # forward du discriminator
                disc_fake_output = self.critic(fake_samples)

                # calculate the loss for the generator
                loss_generator = -torch.mean(disc_fake_output)

                # calculate the gradient for the discriminator
                loss_generator.backward()

                # update the generator
                optimizer_generator.step()

                # add loss in tensorboard 
                if n == self.add_loss:
                    print(f"Epoch: {epoch} Loss D.: {loss_critic}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")

                    self.writer.add_scalar(
                        "Loss/Discriminator_train", loss_critic, epoch
                    )
                    self.writer.add_scalar(
                        "Loss/Generator_train", loss_generator, epoch
                    )
                    self.writer.flush()
                    
                # add generated pictures in tensorboard
                if n == self.add_figure:
                    latent_space_samples = torch.randn(batch_size, self.latent_dim).to(device)
                    generated_samples = self.generator(latent_space_samples)
                    
                    # for i, samples in enumerate(real_samples):
                    #     if i < 3:
                    #         samples = samples.view(samples.size(0), samples.size(2), samples.size(1)).cpu()
                    #         audio_real = dataset.mel_to_waveform(samples.detach(), maxi=self.maxi)
                    #         self.writer.add_audio('Real_audio/'+str(i), audio_real, epoch, sample_rate=16000)
                                                                              
                    #         fig, axs = plt.subplots(1, 1)
                    #         im = axs.imshow(librosa.power_to_db(samples.detach()[0]), origin='lower', aspect='auto')
                    #         self.writer.add_figure('Real_mel_spec/'+str(i), fig, epoch)
                            
                    #     else:
                    #         break
                        
                        
                    for i, samples in enumerate(generated_samples):
                        if i < 3:
                            samples = samples.view(samples.size(0), samples.size(2), samples.size(1)).cpu()
                            audio_fake = dataset.mel_to_waveform(samples.detach(), maxi=self.maxi)
                            self.writer.add_audio('Fake_audio/'+str(i), audio_fake, epoch, sample_rate=16000)
                                                                              
                            fig, axs = plt.subplots(1, 1)
                            im = axs.imshow(librosa.power_to_db(samples.detach()[0]), origin='lower', aspect='auto')
                            self.writer.add_figure('Fake_mel_spec/'+str(i), fig, epoch)
                            
                        else:
                            break
                            
                # save wheckpoint
                if n == self.save_ckpt:
                    # Save checkpoint if the model (to prevent training problem)
                    checkpoint = {
                        "epoch": epoch + 1,
                        "generator": self.generator.state_dict(),
                        "discriminator": self.critic.state_dict(),
                        "optimizer_gen": optimizer_generator.state_dict(),
                        "optimizer_disc": optimizer_critic.state_dict(),
                    }
                    torch.save(checkpoint, self.trained_model_path)