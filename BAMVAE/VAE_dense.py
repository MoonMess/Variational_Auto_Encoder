import torch 
torch.manual_seed(123)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from torchvision import transforms as t
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt; 
plt.rcParams['figure.dpi'] = 200
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
import scipy
import math

class Encoderdense(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)  #28*28
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoderdense(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class Autoencoderdense(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoderdense = Encoderdense(latent_dims)
        self.decoderdense = Decoderdense(latent_dims)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self, x):
        z = self.encoderdense(x)
        return self.decoderdense(z)

    def train(autoencoderdense, data, epochs=20):
        opt = torch.optim.Adam(autoencoderdense.parameters())
        for epoch in range(epochs):
            for x, y in data:
                x = x.to(autoencoderdense.device) # GPU
                opt.zero_grad()
                x_hat = autoencoderdense(x)
                loss = ((x - x_hat)**2).sum()
                loss.backward()
                opt.step()
        return autoencoderdense
        


def plot_reconstructed(autoencoderdense, r0=(-5, 10), r1=(-10, 5), n=10):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(autoencoderdense.device)
            #z = torch.Tensor([[x, y]])
            x_hat = autoencoderdense.decoderdense(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])


class VariationalEncoderdense(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc # hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        """
        There may still be gaps in the latent space because the outputted means may be significantly different and the standard deviations may be small. 
        To reduce that, we add an auxillary loss that penalizes the distribution p(z∣x) for being too far from the standard normal distribution N(0,1). 
        This penalty term is the KL divergence 
        """
        self.kl = 0  #kl divergence
        self.mu = 0
        self.sigma = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        self.mu =  self.linear2(x).to(self.device)
        self.sigma = torch.exp(self.linear3(x)).to(self.device)

        z = self.mu + self.sigma*self.N.sample(self.mu.shape).to(self.device)
        self.kl = (self.sigma**2 + self.mu**2 - torch.log(self.sigma) - 1/2).sum()
        return z, self.mu, self.sigma

class VariationalAutoencoderdense(nn.Module):
    def __init__(self, latent_dims, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.encoder= VariationalEncoderdense(latent_dims)
        self.decoder = Decoderdense(latent_dims)
        self.device = device

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        return self.decoder(z), mu, sigma

    def train(self, data, epoch, k1=0.5,k2=0.5):
        opt = torch.optim.Adam(self.parameters())
        train_loss = 0
        #for epoch in range(epoch):
        for batch_idx, (x, _) in enumerate(data):
                x = x.to(self.device) # GPU
                opt.zero_grad()
                x_hat, mu, sigma = self(x)
                #loss = ((x - x_hat)**2).sum() + self.encoder.kl
                loss = loss_function(x_hat, x, mu, sigma)
                loss.backward()
                train_loss += loss.item()
                opt.step()
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(x)}/{len(data.dataset)} '
                    f'({100. * batch_idx / len(data):.0f}%)]\tLoss: {loss.item() / len(x):.6f}')
        average_loss=train_loss / len(data.dataset)
        print(f'====> Epoch: {epoch} Average loss: {average_loss:.4f}')
        return average_loss

    def plot_latent(self, data):
        for i, (x, y) in enumerate(data):
            z, mu, sigma = self.encoder(x.to(self.device))
            #z = autoencoderdense.encoderdense(x)
            z = z.to('cpu').detach().numpy()
            #z = z.numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10', alpha=0.5, s=20)
        plt.colorbar()
        plt.title("Latent space with dimension 2")
        # plt.savefig("Latent space dim 2.png")

    def test(self,test_loader):
    #informing the model we are evaluating and not training
        opt = torch.optim.Adam(self.parameters())
        device=self.device
        test_loss= 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device) # GPU
                opt.zero_grad()
                x_hat, mu, sigma = self(x)
                loss = loss_function(x_hat, x, mu, sigma)
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)

        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def print_nb_params(self):
        nb_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.nb_params=nb_params
        print(f"The model has {nb_params} parameters")
        return nb_params

    def save_model(self,folder,name):
            """save the model weights to a specific path"""
            torch.save(self.state_dict(),Path(folder,name+".h5"))

    def generate_from_noise(self,nb_images=1024,dim_latent_space=2,variance=1.,mean=0.):
            """generating random images from random gaussian noise in the latent space"""
            random_vectors=np.random.randn(nb_images,1,1,dim_latent_space)*variance+mean
            random_vectors=torch.tensor(random_vectors,dtype=torch.float32).to(self.device)
            with torch.no_grad():
                decoded_random=self.decoder(random_vectors)

            reco_np= decoded_random.detach().cpu().numpy().reshape(1024,28,28)
            fig,axs=plt.subplots(32,32,figsize=(10,10))
            axs=axs.ravel()
            for k,image in enumerate(reco_np):
                axs[k].imshow(image)
                axs[k].grid(False)
                axs[k].set_xticks([])
                axs[k].set_yticks([])
            plt.grid(False)
            return decoded_random

    def generate_images(self,nb_images,display=False):
        """
        Generates nb_images (if nb_images is a perfect square) random images
        and outputs the torch.tensor corresponding to the images
        to plot the images, convert into numpy first by doing :
        x_reconst.cpu().detach().numpy()
        """
        grid_size   = int(math.sqrt(nb_images))
        grid_scale  = 1
        grid=[]
        for y in scipy.stats.norm.ppf(np.linspace(0.99, 0.01, grid_size),scale=grid_scale):
            for x in scipy.stats.norm.ppf(np.linspace(0.01, 0.99, grid_size),scale=grid_scale):
                grid.append( (x,y) )
        grid=np.array(grid)
        x_reconst = self.decoder(torch.tensor(grid.reshape(-1,1,1,2) ,dtype=torch.float32,device=self.device))
        if display:
            reco_np=x_reconst.detach().cpu().numpy().reshape(1024,28,28)
            fig,axs=plt.subplots(32,32,figsize=(10,10))
            axs=axs.ravel()
            for k,image in enumerate(reco_np):
                axs[k].imshow(image)
                axs[k].grid(False)
                axs[k].set_xticks([])
                axs[k].set_yticks([])
            plt.grid(False)
        return x_reconst


    def interpolate(self, x_1, x_2, n=12):
        z_1 = self.encoder(x_1)
        z_2 = self.encoder(x_2)
        z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
        interpolate_list = self.decoder(z)
        interpolate_list = interpolate_list.to('cpu').detach().numpy()

        w = 28
        img = np.zeros((w, n*w))
        for i, x_hat in enumerate(interpolate_list):
            img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    def interpolate_gif(self, filename, x_1, x_2, n=100):
        z_1 = self.encoder(x_1)
        z_2 = self.encoder(x_2)

        z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

        interpolate_list = self.decoder(z)
        interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

        images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
        images_list = images_list + images_list[::-1] # loop back beginning

        images_list[0].save(
            f'{filename}.gif',
            save_all=True,
            append_images=images_list[1:],
            loop=1)

        
    def complete_train_and_test(self,train_loader,test_loader,nb_epochs=20,k1=0.5,k2=0.5,view_latent=False,test_dataset=None):
        """Train the model and return both train and test loss for each epoch"""
        train_losses=[]
        test_losses=[]
        #if view_latent:
            # initialise the latent space where it should be random at first
            #self.view_latent_func(test_dataset)
            #print('Initialised the latent space visualisation before first epoch')
        for epoch in range(1, nb_epochs+1):
            epoch_train_loss=self.train(train_loader, epoch)
            train_losses.append(epoch_train_loss)
            test_loss=self.test(test_loader=test_loader)
            test_losses.append(test_loss)
        return train_losses,test_losses



            
def plot_image(images_input, title=''):
  #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,10) 
    plt.title(title)
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    for idx, image in enumerate(images_input):
      axarr[idx].imshow(image.reshape(28,28).cpu().detach().numpy(), cmap="gray")
      #to remove axis 
      axarr[idx].get_xaxis().set_ticks([])
      axarr[idx].get_yaxis().set_ticks([])


def decompose_loss_function(recon_x, x, mu, log_var):
        # reconstruction loss
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # KL divergence
        # return reconstruction error + KL divergence losses
        KLD = -torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE,KLD

def loss_function(recon_x, x, mu, log_var,k1=0.5,k2=0.5):
        k1=k1/(k1+k2)
        k2=k2/(k1+k2)
        BCE,KLD=decompose_loss_function(recon_x,x,mu,log_var)
        return k1*BCE + k2*KLD
