import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

class VAE(nn.Module):
    def __init__(self, name,input_dim=28*28, 
            high_feature_dim=32*20*20, z_dim=2,
            device=torch.device("mps")):
        """
        high_feature_dim : dimension of the vectors just before the FC layers in the encoder
        z_dim: latent space dimension
        device : for mac users, use device = torch.device("mps")
        (GPU acceleration for Mac)
        name: allowes to save the module with specific name
        """
        super(VAE, self).__init__()
        self.input_dim=input_dim
        self.high_feature_dim=high_feature_dim
        self.z_dim=z_dim
        self.device=device
        self.name=name

        # encoder part
        # convolution layers
        self.encConv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1)
        self.encConv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1)
        # fully connected layers where mu and logvariance are sampled from
        self.encFC1=nn.Linear(high_feature_dim,z_dim)
        self.encFC2=nn.Linear(high_feature_dim,z_dim)

        # decoder part
        self.decFC1=nn.Linear(z_dim,high_feature_dim)
        self.decConv1=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5,stride=1)
        self.decConv2=nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=5,stride=1)

    def encoder(self, x):
        """
        Encoder portion 
        Takes an image an outputs the average and logvar of the distribution
        """
        x = nn.LeakyReLU()(self.encConv1(x))
        x = torch.sigmoid(self.encConv2(x))
        x=x.view(-1,self.high_feature_dim)#reshaping the array to a 1d vector
        mu=self.encFC1(x)
        log_var=self.encFC2(x)
        return mu,log_var # mu, log_var
    
    def sampling(self, mu, log_var):
        """
        Sampling a random vector from the gaussian distribution in latent space
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        """
        Decoder portion
        Goes from the latent space to the image space
        """
        x = F.relu(self.decFC1(z))
        x=x.view(-1,32,20,20)#reshaping the array to a 1d vector
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        # reshape to initial shape of the batch
        return x.view(-1,1,28,28)

    
    def forward(self, x):
        """forward pass of the VAE"""
        mu, log_var = self.encoder(x)
        z =  self.sampling(mu,log_var)
        out = self.decoder(z)
        return out, mu, log_var
    
    def save_model(self,folder=None,model_path=None):
        """save the model weights to a specific folder where name is the name of the model
        in order to force the name of the entire path, input path
        """
        if not model_path:
            output_path=Path(folder,self.name+".h5")
        else :
            output_path=model_path.strip()
            if not output_path.endswith('.h5'):
                output_path=output_path+".h5"
        torch.save(self.state_dict(),output_path)

    def generate_from_noise(self,nb_images=10,variance=1.,mean=0.):
        """generating random images from random gaussian noise in the latent space"""
        random_vectors=np.random.randn(nb_images,1,1,self.z_dim)*variance+mean
        random_vectors=torch.tensor(random_vectors,dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            decoded_random=self.decoder(random_vectors)
        return decoded_random
    
    def generate_images(self,grid_size=None,nb_images=1024,display=False):
        """
        Generates nb_images (if nb_images is a perfect square) random images
        and outputs the torch.tensor corresponding to the images
        to plot the images, convert into numpy first by doing :
        x_reconst.cpu().detach().numpy()
        if grid_size is inputed, then there are going to be grid_size^self.z_dim generated images
        else, use nb_images to generate the images
        """
        if not (grid_size or nb_images):
            raise Exception("You have to input either the number of images to produce\
                or the grid size")
        if nb_images:
            grid_size   = math.ceil(nb_images**(1/int(self.z_dim)))
            inperfect_root=(grid_size!=int(nb_images**(1/int(self.z_dim))))
            # inperfect root is when the nb of images we want to generate is not exactly
            # the nb of nodes in the generated grid
        else :
            nb_images=int(grid_size**self.z_dim)
        grid_scale  = 1
        grid=[]

        x=scipy.stats.norm.ppf(np.linspace(0.99, 0.01, grid_size),scale=grid_scale)
        y=scipy.stats.norm.ppf(np.linspace(0.99, 0.01, grid_size),scale=grid_scale)
        grid=np.meshgrid(*(tuple(x),)*self.z_dim)
        grid=np.array(grid).T.reshape(-1,self.z_dim)

        x_reconst = self.decoder(torch.tensor(grid.reshape(-1,1,1,self.z_dim) ,dtype=torch.float32,device=self.device))
        reco_np=x_reconst.detach().cpu().numpy().reshape(-1,28,28)
        if nb_images:
            if inperfect_root:
                # randomly select nb_images 
                indexes=np.random.choice(reco_np.shape[0],size=nb_images,replace=False)
                indexes=np.sort(indexes)
                reco_np=reco_np[indexes]
        if display:
            fig,axs=plt.subplots(32,32,figsize=(10,10))
            axs=axs.ravel()
            for k,image in enumerate(reco_np):
                axs[k].imshow(image)
                axs[k].grid(False)
                axs[k].set_xticks([])
                axs[k].set_yticks([])
            plt.grid(False)
        self.grid=grid
        return reco_np

    def print_nb_params(self):
        nb_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.nb_params=nb_params
        print(f'The model has {nb_params:,} parameters'.replace(","," "))
        return nb_params
