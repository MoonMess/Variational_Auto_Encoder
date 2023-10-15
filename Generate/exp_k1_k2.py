from train_model import train_model_func
import numpy as np
import argparse
from pathlib import Path

"""call using
python Generate/exp_k1_k2.py --z_dim 2 --nb_epochs 30 --model_folders 'testing_k1_k2' --losses_folders 'losses_k1_k2'
"""

nb_points=20
k2_arr =np.linspace(1,0,nb_points +1)
k1_arr =np.linspace(0,1,nb_points +1)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment different k1 and k2')
    parser.add_argument("--z_dim", type=int, default=2)
    parser.add_argument("--nb_epochs", type=int, default=20)
    parser.add_argument("--model_folders", type=str, default="./",help="where to export the models")
    parser.add_argument("--losses_folders", type=str, default="./",help="where to export the models")
    
    args=parser.parse_args()
    z_dim=args.z_dim
    nb_epochs=args.nb_epochs
    model_folders=args.model_folders
    losses_folders=args.losses_folders

    for step,(k1,k2) in enumerate(zip(k1_arr,k2_arr)):
        print(f"Training step {step}",end="\r")
        model_path=Path(model_folders,f"model k1={k1}.h5")
        train_model_func(z_dim=z_dim,learning_rate=1e-3,nb_epochs=nb_epochs,
                        batch_size=128,model_path=model_path,
                        losses_folder=losses_folders,k1=k1,k2=k2)
