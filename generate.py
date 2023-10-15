import torch 
import torchvision
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
import sys
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from torch.nn import DataParallel
# from torch.optim import lr_scheduler

# IMPORT CUSTOM MODULES
from BAMVAE.VAE_v2 import VAE2
from BAMVAE.VAE_Moon import VAE

# def load_model(model):
#     state = torch.load(os.path.join('models', 'VAE_v2_30_epochs_2D.h5'))
#     model.load_state_dict(state['model_state_dict'])
#     return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate VAE')
    parser.add_argument("--model_path", type=str, default="models/VAE_Moon_zdim_16.h5",
                 help="Path from which to load the model.")
    parser.add_argument("--model_creator", type=str, default="Moon",
                 help="To know which model we want to use")
    args = parser.parse_args()
    model_path=args.model_path
    model_creator=args.model_creator

    # Data Pipeline
    print('Dataset loading...')

    print('Model loading...')

    #check Mac or Linux/windows system
    if sys.platform=="darwin":
        try:
            device = torch.device("mps") # GPU acceleration for Mac
        except:
            device="cpu"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # Cuda for GPU

    #create the model framework
    # model = VAE2(name=f"VAE_{z_dim}_dim",z_dim=z_dim,cnL1=16,cnL2=32,cnL3=64, device=device).to(device)
    kwargs,state=torch.load(model_path)
    if model_creator=='Moon' : 
        model=VAE(**kwargs).to(device)
    else:
        model=VAE2(**kwargs).to(device)
        
    print(kwargs)
    z_dim=kwargs["z_dim"]
    #load a trained model 
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(state)
    #model = load_model(model)
    model.eval() #no more train
    print('Model loaded.')

    print('Start Generating :')
    os.makedirs('samples', exist_ok=True)
    with trange(1024, desc="Generated", unit="img") as te:
         for idx in te:
            with torch.no_grad():
                sample = torch.randn(1,1,1,z_dim).to(device)
                x = model.decoder(sample)
                torchvision.utils.save_image(x, os.path.join('samples', f'{idx}.png'))            
    print("Successfully generated the 1024 images")