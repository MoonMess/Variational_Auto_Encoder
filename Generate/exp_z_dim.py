from train_model import train_model_func
import numpy as np
import argparse
from pathlib import Path

"""call using
python3 Generate/exp_z_dim.py --nb_epochs 50 --model_folders './experiments_lamsade/models_lamsade/testing_z_dim/' --losses_folders './experiments_lamsade/losses_lamsade/losses_z_dim/'
"""

n_latent=11
z_dims=[2**k for k in range(n_latent)]

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment different z_dims')
    parser.add_argument("--nb_epochs", type=int, default=50)
    parser.add_argument("--model_folders", type=str, default="./",help="where to export the models")
    parser.add_argument("--losses_folders", type=str, default="./",help="where to export the models")
    
    args=parser.parse_args()
    nb_epochs=args.nb_epochs
    model_folders=args.model_folders
    losses_folders=args.losses_folders

    for index_zdim,z_dim in enumerate(z_dims):
        print(f"Training step {index_zdim}/{n_latent}, {z_dim = }",end="\r")
        model_path=Path(model_folders,f"model z_dim={z_dim}_epochs={nb_epochs}.h5")
        train_model_func(z_dim=z_dim,learning_rate=1e-3,nb_epochs=nb_epochs,
                        batch_size=128,model_path=model_path,
                        losses_folder=losses_folders,k1=0.5,k2=0.5)
