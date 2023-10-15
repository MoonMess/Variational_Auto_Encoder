import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import numpy as np
import math

##################################
## SECOND IMPLEMENTATION
##################################

class VAE2(nn.Module):
    def __init__(self, name,input_dim=28*28,
            cnL1=32,cnL2=64,cnL3=64, z_dim=2,
            interm_dim=16,
            device=torch.device("mps")):
        """
        high_feature_dim : dimension of the vectors just before the FC layers in the encoder
        z_dim: latent space dimension
        device : for mac users, use device = torch.device("mps")
        (GPU acceleration for Mac)
        name: allowes to save the module with specific name
        """
        super(VAE2, self).__init__()
        self.input_dim=input_dim
        self.z_dim=z_dim
        self.device=device
        self.interm_dim=interm_dim
        self.name=name
        self.cnL1=cnL1
        self.cnL2=cnL2
        self.cnL3=cnL3
        self.high_feature_dim=self.cnL3*16*16
        self.counter_nan=0
        # allowes saving the model params and loading them when loading the model in order to prevent having errors of missmatch between layers dimensions
        self.kwargs={"name":name,"input_dim":input_dim,
                    "cnL1":cnL1,"cnL2":cnL2,"cnL3":cnL3, 
                    "z_dim":z_dim,"interm_dim":interm_dim,
                    "device":device}
        

        # encoder part
        # convolution layers
        self.encConv1=nn.Conv2d(in_channels=1,out_channels=self.cnL1,
                                kernel_size=5,stride=1)
        self.encBN1=nn.BatchNorm2d(self.cnL1)
        self.encConv2=nn.Conv2d(in_channels=self.cnL1,out_channels=self.cnL2,
                                kernel_size=5,stride=1)
        self.encBN2=nn.BatchNorm2d(self.cnL2)
        self.encConv3=nn.Conv2d(in_channels=self.cnL2,out_channels=self.cnL3,
                                kernel_size=5,stride=1)
        self.encBN3=nn.BatchNorm2d(self.cnL3)

        # fully connected layers where mu and logvariance are sampled from
        self.encFCInterm=nn.Linear(self.high_feature_dim,self.interm_dim)
        self.encFC1=nn.Linear(self.interm_dim,z_dim)
        self.encFC2=nn.Linear(self.interm_dim,self.z_dim)

        # decoder part
        self.decFC1=nn.Linear(z_dim,self.high_feature_dim)
        self.decConv1=nn.ConvTranspose2d(in_channels=self.cnL3,out_channels=self.cnL2,kernel_size=5,stride=1)
        self.decBN1=nn.BatchNorm2d(self.cnL2)

        self.decConv2=nn.ConvTranspose2d(in_channels=self.cnL2,out_channels=self.cnL1,kernel_size=5,stride=1)
        self.decBN2=nn.BatchNorm2d(self.cnL1)

        self.decConv3=nn.ConvTranspose2d(in_channels=self.cnL1,out_channels=1,kernel_size=5,stride=1)

    def encoder(self, x):
        """
        Encoder portion 
        Takes an image an outputs the average and logvar of the distribution
        """
        x = nn.LeakyReLU()(self.encBN1(self.encConv1(x)))
        #print(x.mean())
        x = nn.LeakyReLU()(self.encBN2(self.encConv2(x)))
        #print(x.mean())
        x = nn.LeakyReLU()(self.encBN3(self.encConv3(x)))
        #print(x.mean())
        x=x.view(-1,self.high_feature_dim)#reshaping the array to a 1d vector
        x= torch.relu(self.encFCInterm(x))
        mu=self.encFC1(x)
        # print("encend",x.mean())
        log_var=self.encFC2(x)
        # print("mu,logvar",mu.mean(),log_var.mean())
        self.mu=mu
        self.log_var=log_var
        return mu,log_var # mu, log_var
    
    def sampling(self, mu, log_var):
        """
        Sampling a random vector from the gaussian distribution in latent space
        """
        std = torch.exp(0.5*log_var)
        # print("STD;" , std)
        eps = torch.randn_like(std)
        while eps.isnan().any():
            print(std.mean())
            self.counter_nan+=1
            eps = torch.randn_like(std)
        # print("eps;",eps)
        z_sample=eps*std+mu
        self.z_zample=z_sample
        # print("zsample;",z_sample.mean())
        return z_sample # return z sample
        
    def decoder(self, z):
        """
        Decoder portion
        Goes from the latent space to the image space
        """
        x = nn.LeakyReLU()(self.decFC1(z))
        # print("dec1",x.mean())
        x=x.view(-1,self.cnL3,16,16)#reshaping the array to a 1d vector
        x = nn.LeakyReLU()(self.decBN1(self.decConv1(x)))
        # print(x.mean())
        x = nn.LeakyReLU()(self.decBN2(self.decConv2(x)))
        # print(x.mean())
        x = nn.Sigmoid()(self.decConv3(x))
        # print("dec3",x.mean())
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
            output_path=model_path
        torch.save([self.kwargs,self.state_dict()],output_path)
    
    def load(self,path):
        """load a model using the kwargs where params values are saved and the weights themselves"""
        kwargs, state = torch.load(path)
        self = VAE2(**kwargs)
        self.load_state_dict(state)

    def generate_from_noise(self,nb_images=10,variance=1.,mean=0.):
        """generating random images from random gaussian noise in the latent space"""
        random_vectors=torch.randn(nb_images,1,1,self.z_dim)*variance+mean
        random_vectors=random_vectors.to(self.device)
        # random_vectors=torch.tensor(random_vectors,dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            decoded_random=self.decoder(random_vectors)
        return decoded_random

    def generate_images(self,grid_size=None,nb_images=1024):
        import scipy
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

    def print_nb_params(self):
        nb_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.nb_params=nb_params
        print(f'The model has {nb_params:,} parameters'.replace(","," "))
        return nb_params

# MSE=nn.MSELoss(reduction="mean")

def MSE(x_recon,x):
    return torch.sum((x-x_recon)**2)

def decompose_loss_function(recon_x, x, mu, log_var,batch_idx,z):
    # reconstruction loss
    Reconstruction_Loss = MSE(recon_x, x)
    # KL divergence
    # return reconstruction error + KL divergence losses
    KLD = -torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    if any([Reconstruction_Loss.isnan(),KLD.isnan()]):
        # debug in the case where the grad becomes nan and "contaminates the weights"
        print("WARNING : THERE ARE NAN VALUES. PRINTING mu and logvar and x_recon")
        print("mu",mu.mean())
        print("log_var",log_var.mean())
        if Reconstruction_Loss.isnan():
            print("pb comes from recon")
            print('recon loss',Reconstruction_Loss)
            print("Occured on batch id : ", batch_idx)
            print("lasomme:", torch.sum((x-recon_x)**2))
            print("lashape",recon_x.shape)
            print("sampledz",z.mean())
            # print("enc")
        if KLD.isnan():
            print("pb comes from kld")
    # print("losses",Reconstruction_Loss,KLD)
    return Reconstruction_Loss,KLD

def loss_function(recon_x, x, mu, log_var,k1=0.5,k2=0.5,batch_idx=None,z=None):
    # debugging
    # for k,el in enumerate([recon_x, x, mu, log_var]):
    #     print(f"{k}:{el.shape}")
    # debug
    # print(f'{recon_x.shape =},{x.shape =}, {mu.shape = }, {log_var.shape =} ')
    k1=k1/(k1+k2)
    k2=k2/(k1+k2)
    Reconstruction_Loss,KLD=decompose_loss_function(recon_x,x,mu,log_var,batch_idx,z)
    # print(f"{Reconstruction_Loss = } and {KLD = }")
    return k1*Reconstruction_Loss + k2*KLD

def train(vae,epoch,train_loader,optimizer,k1=0.5,k2=0.5):
    #inform the model we are training it (for example stops the dropout layers from functioning)
    device=vae.device
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mu, log_var = vae.encoder(data)
        z=vae.sampling(mu,log_var)
        recon_batch=vae.decoder(z)
        loss = loss_function(recon_batch, data, mu, log_var,k1,k2,batch_idx,z)
        if np.isnan(loss.item()):
            print(f"WARNING :: NAN LOSS! at epoch {epoch}")

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                 f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    average_loss=train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {average_loss:.4f}')
    return average_loss

def test(vae,test_loader,view_latent=False,test_dataset=None,k1=0.5,k2=0.5):
    #informing the model we are evaluating and not training
    vae.eval()
    device=vae.device
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = vae(data)

            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var,k1=k1,k2=k2).item()
    # devide the total loss by the nb of elements in the batch
    test_loss /= len(test_loader.dataset)
    if view_latent:
        view_latent_func(vae,test_dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

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

def complete_train_and_test(vae,train_loader,test_loader,optimizer,nb_epochs=10,k1=0.5,k2=0.5,view_latent=False,test_dataset=None):
    """Train the model and return both train and test loss for each epoch"""
    train_losses=[]
    test_losses=[]
    if view_latent:
        # initialise the latent space where it should be random at first
        view_latent_func(vae,test_dataset)
        print('Initialised the latent space visualisation before first epoch')
    for epoch in range(1, nb_epochs+1):
        epoch_train_loss=train(vae,epoch,train_loader=train_loader,
                                optimizer=optimizer,k1=k1,k2=k2)
        train_losses.append(epoch_train_loss)
        test_loss=test(vae,test_loader=test_loader,view_latent=view_latent,
                        test_dataset=test_dataset,k1=k1,k2=k2)
        test_losses.append(test_loss)
    return train_losses,test_losses
