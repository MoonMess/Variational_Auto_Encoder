import numpy as np
import torch
import sys
from pathlib import Path
import os
sys.path.append("../BAMVAE")
sys.path.append(str(Path(__file__).parent.parent))
import argparse

"""Call the function with  
python3 Generate/train_model.py --nb_epochs 100 --z_dim 16 --losses_folder './' --k1 0.5 --k2 0.5"""

from BAMVAE.MNISTLoader import MNIST_Loader
from BAMVAE.VAE_v2 import VAE2,complete_train_and_test
import torch.optim as optim


def train_model_func(z_dim,nb_epochs,batch_size,k1,k2,model_folder=None,learning_rate=1e-3,model_path=None,losses_folder=None):
    if sys.platform=="darwin":
        try:
            device_ = torch.device("mps") # GPU acceleration for Mac
        except:
            device_="cpu"
    else:
        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_)
    print(f'{"#"*30:<30}')
    print(f"Lauching training for {nb_epochs = }, {z_dim = }, {batch_size = }, {learning_rate =}")
    print(f'{"#"*30:>30}')

    print("Uploading the MNIST dataset")
    mnist=MNIST_Loader(bs=batch_size,normalise=True,
                            root_folder='../mnist_data')
    train_dataset,test_dataset=mnist.__getdatasets__()
    train_loader,test_loader=mnist.__getdataloaders__()

    # on previous models, tried cnL1=8,cnL2=16,cnL3=32

    vae=VAE2(name=f"VAE_{nb_epochs}epochs_{z_dim}_dim_k1_{k1}",
            z_dim=z_dim,cnL1=16,cnL2=32,cnL3=64,device=device).to(device)

    optimizer= optim.Adam(vae.parameters(),lr=learning_rate)
    print(vae.print_nb_params())

    train_losses,test_losses=complete_train_and_test(vae,train_loader=train_loader,
                            test_loader=test_loader,optimizer=optimizer,
                            nb_epochs=nb_epochs,view_latent=False,
                            test_dataset=test_dataset,
                            k1=k1,k2=k2)

    print(f"There where {vae.counter_nan} clashes of nan values in the process with {nb_epochs} epochs")
    if model_folder:
        os.makedirs(model_folder,exist_ok=True)
        vae.save_model(folder=model_folder)
        print("Model saved to ", model_folder)
    if model_path:
        model_path_=Path(model_path)
        parent_folder=model_path_.parent
        os.makedirs(parent_folder,exist_ok=True)
        vae.save_model(model_path=model_path)
        print("Model saved to ", model_path)
    if losses_folder:
        os.makedirs(losses_folder,exist_ok=True)
        os.makedirs(losses_folder,exist_ok=True)
        np.save(Path(losses_folder,f'train_loss_{vae.name}.npy'),train_losses)
        np.save(Path(losses_folder,f"test_loss_{vae.name}.npy"),test_losses)
        print(f"Saved the losses arrays at {losses_folder} ")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training BAM VAE')
    parser.add_argument("--batch_size", type=int, default=128,
                      help="The batch size to use for training.")
    parser.add_argument("--z_dim", type=int, default=2,
                      help="Latent space dimension.")
    parser.add_argument("--nb_epochs", type=int, default=30,
                      help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                      help="Learning rate.")
    parser.add_argument("--model_folder",type=str,default=None,
                            help="Folder to save the model and the name will be gen automatically")
    parser.add_argument("--model_path",type=str,default=None,
                            help="Full path to save the model")
                            
    parser.add_argument("--losses_folder",type=str,default=None,help='save the losses for further study')
    parser.add_argument("--k1",type=float,default=0.5)
    parser.add_argument("--k2",type=float,default=0.5)
    
    args=parser.parse_args()
    z_dim=args.z_dim
    learning_rate=args.learning_rate
    nb_epochs=args.nb_epochs
    batch_size=args.batch_size
    model_folder=args.model_folder
    model_path=args.model_path
    losses_folder=args.losses_folder
    k1=args.k1
    k2=args.k2

    train_model_func(z_dim=z_dim,learning_rate=learning_rate,
                            nb_epochs=nb_epochs,batch_size=batch_size,
                            model_folder=model_folder,model_path=model_path,
                            losses_folder=losses_folder,k1=k1,k2=k2)