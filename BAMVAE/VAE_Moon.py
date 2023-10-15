import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import torch.optim as optim
import sys
sys.path.append("../")
from BAMVAE.MNISTLoader import MNIST_Loader

class VAE(nn.Module):
    def __init__(self, name="VAE_Moon", imgChannels=1, input_dim=28*28, z_dim=256, 
                cnL1=16,cnL2=32,cnL3=64, interm_dim=16,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                learning_rate=1e-3,k1=0.5,k2=0.5):
        super(VAE, self).__init__()
        self.k1=k1
        self.k2=k2
        self.input_dim=input_dim
        self.z_dim=z_dim
        self.device=device
        self.interm_dim=interm_dim
        self.name=name
        self.cnL1=cnL1
        self.cnL2=cnL2
        #self.cnL3=cnL3
        self.high_feature_dim=self.cnL2*20*20
        self.imgChannels= imgChannels
        self.counter_nan=0
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(self.imgChannels, self.cnL1, 5)
        self.encConv2 = nn.Conv2d(self.cnL1, self.cnL2, 5)
        self.encFC1 = nn.Linear(self.high_feature_dim, self.z_dim)
        self.encFC2 = nn.Linear(self.high_feature_dim, self.z_dim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(self.z_dim, self.high_feature_dim)
        self.decConv1 = nn.ConvTranspose2d(self.cnL2, self.cnL1, 5)
        self.decConv2 = nn.ConvTranspose2d(self.cnL1, self.imgChannels, 5)

        # allowes saving the model params and loading them when loading the model in order to prevent having errors of missmatch between layers dimensions
        self.kwargs={"name":name,"input_dim":input_dim,
                    "cnL1":cnL1,"cnL2":cnL2, "cnL3":cnL3, 
                    "z_dim": z_dim,"interm_dim":interm_dim,
                    "device":device}
        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)


    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*20*20)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
        
    def train(self, num_epochs):
        mnist= MNIST_Loader(bs=128, normalise=False,
        root_folder='../mnist_data')
        train_loader,test_loader=mnist.__getdataloaders__()

        """
        Training the network for a given number of epochs
        The loss after every epoch is printed
        """
        train_losses=[]
        for epoch in range(num_epochs):
            train_loss = 0
            for idx, data in enumerate(train_loader, 0):
                imgs, _ = data
                imgs = imgs.to(self.device)

                # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                out, mu, logVar = self(imgs)

                # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                loss = self.k1*F.binary_cross_entropy(out, imgs, size_average=False) + self.k2*kl_divergence

                # Backpropagation based on the loss
                self.optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                print('Epoch {}: Loss {}'.format(epoch, loss))
            average_loss=train_loss / len(train_loader.dataset)
            train_losses.append(average_loss)
            print(f'====> Epoch: {epoch} Average loss: {average_loss:.4f}')
        return train_losses

    def train_test(self, num_epochs):
        mnist= MNIST_Loader(bs=128, normalise=False,
        root_folder='../mnist_data')
        train_loader,test_loader=mnist.__getdataloaders__()

        """
        Training the network for a given number of epochs
        The loss after every epoch is printed
        """
        train_losses=[]
        test_losses=[]
        for epoch in range(num_epochs):
            train_loss = 0
            for idx, data in enumerate(train_loader, 0):
                imgs, _ = data
                imgs = imgs.to(self.device)

                # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                out, mu, logVar = self(imgs)

                # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                loss = self.k1*F.binary_cross_entropy(out, imgs, size_average=False) + self.k2*kl_divergence

                # Backpropagation based on the loss
                self.optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                #print(f'Epoch {epoch}: Loss {loss}')
            average_loss=train_loss / len(train_loader.dataset)
            train_losses.append(average_loss)
            print(f'====> Epoch: {epoch} Average Train loss: {average_loss:.4f}')

            test_loss = 0
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.to(self.device)
                    out, mu, logVar = self(imgs)

                    # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                    kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                    test_loss += F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            avg_test = test_loss.item() / len(test_loader.dataset)
            test_losses.append(avg_test)
            print('====> Test set loss: {:.4f}'.format(avg_test))

        return train_losses, test_losses

    def print_nb_params(self):
        nb_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.nb_params=nb_params
        print(f'The model has {nb_params:,} parameters'.replace(","," "))
        return nb_params

    def view_latent_func(vae,test_dataset):
        """
        Show where a test_dataset maps to in the latent space
        can be called at each epoch to see how the latent space moves through time
        """
        try :
            view_latent_list=vae.view_latent_list
        except:
            # if attribute doesnt exist, we create it
            vae.view_latent_list=[]
            # view_latent_list=vae.view_latent_list
        # after that, we are sure that the attribute is created
        ## view how latent space evolved in time
        visualisation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10_000, shuffle=False)

        for test_images, test_labels in visualisation_loader:  
            break
    
        vae.eval()
        test_images=test_images.to(vae.device)
        # data = data.to(device)
        # recon, mu, log_var = vae(data)
        with torch.no_grad():
            avg,_=vae.encoder(test_images)
        avg=avg.cpu().numpy()
        vae.view_latent_list.append([avg,test_labels])

    def save_model(self,folder=None,model_path=None):
        """save the model weights to a specific folder where name is the name of the model
        in order to force the name of the entire path, input path
        """
        if not model_path:
            output_path=Path(folder,self.name+".h5")
        else :
            output_path=model_path
        torch.save([self.kwargs,self.state_dict()],output_path)
        
