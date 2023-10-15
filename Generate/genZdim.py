import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

# IMPORT CUSTOM MODULES
import sys
sys.path.append("../")
from BAMVAE.MNISTLoader import MNIST_Loader
from BAMVAE.VAE_v2 import VAE2 as VAE, loss_function,complete_train_and_test,decompose_loss_function
from BAMVAE.VAE_Moon import VAE as VAEMoon

mnist=MNIST_Loader(bs=128,
    root_folder='../mnist_data')

train_dataset,test_dataset=mnist.__getdatasets__()
train_loader,test_loader=mnist.__getdataloaders__()

learning_rate=1e-3

# vae=VAE(name=f"VAE_{nb_epochs}_epochs",z_dim=3).to(device)
# optimizer= optim.Adam(vae.parameters(),lr=learning_rate)
params_list=[k for k in range(10)]
device='cuda'
# images to plot the latent space at the end of training
visualisation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024*2, shuffle=False)

test_images=test_dataset.data[:2048]
test_images=test_images.to(device)
test_images=test_images.to(torch.float32)
test_images=test_images.view(-1,1,28,28)

nb_params=10
dic_models={}
nb_epochs = 1
for k in range(nb_params):
    print("---"*10)
    print(f"Training model {k+1}/{nb_params}",end="\n")
    k1=k/nb_params
    k2=(nb_params-k)/nb_params
    dic_models[k]={"vae":VAEMoon(z_dim=2,name=f"params_{k}",k1=k1,k2=k2).to(device)}

    ### train the model with the above parameters
    losses=dic_models[k]["vae"].train_test(num_epochs=nb_epochs)
    # saving the models
    dic_models[k]["vae"].save_model(model_path=f"../models/varying_loss_par/reco={k1}_kld={k2}.h5")
    ### evaluate the models performance
    ### evaluate the models FID
    ### keep track of the losses to plot in the end
    ### View the latent space

    dic_models[k]["vae"].eval()
    # data = data.to(device)
    # recon, mu, log_var = dic_models[k]["vae"](data)
    with torch.no_grad():
        avg,_=dic_models[k]["vae"].encoder(test_images)
    avg=avg.cpu().numpy()
    dic_models[k]["avg"]=avg
    np.save(f"../Exps/latent_space_view/latent_reco={k1}_kld={k2}.npy",avg)
    # generate random images
    with torch.no_grad():
            sample = torch.randn(1024,1,1,dic_models[k].z_dim).to(device)
            x = dic_models[k].decoder(sample)
            reco_np=x.detach().cpu().numpy().reshape(-1,28,28)
            np.save(os.path.join('samples', f'{dic_models[k].name}.npy'),reco_np)  
    np.save(f"../Exps/gen_images/latent_reco={k1}_kld={k2}_images.npy",reco_np.detach().cpu().numpy().reshape(-1,28,28))
    ### TODO : keep track of both crossentropy loss and the KLD
    