import numpy as np
import torch
import torchvision
import sys
from pathlib import Path
import os
sys.path.append("../BAMVAE")
sys.path.append(str(Path(__file__).parent.parent))
from BAMVAE.VAE_v2 import VAE2,complete_train_and_test
import argparse

"""Example call
python3 Generate/generate_from_noise.py --model_path "models/model_name" --nb_images 1024 --output_type 'npy' --output_folder 'output_images/'

"""

def generate_images_func(model_path,nb_images,output_type,output_folder,custom_filename):
    if model_path is None:
        raise Exception("Enter a path of the model you want to use")
    #check Mac or Linux/windows system
    if sys.platform=="darwin":
        try:
            device = torch.device("mps") # GPU acceleration for Mac
        except:
            device="cpu"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # Cuda for GPU

    kwargs, state = torch.load(model_path)
    model = VAE2(**kwargs).to(device)
    model.load_state_dict(state)
    generated_images=model.generate_from_noise(nb_images=nb_images)
    os.makedirs(output_folder,exist_ok=True)
    if output_type=="npy":
        generated_images_np=generated_images.detach().cpu().numpy()
        if custom_filename:
            np.save(Path(output_folder,custom_filename),generated_images_np)
        else:
            np.save(Path(output_folder,"generated_images.npy"),generated_images_np)
            
        print("Saved images as npy file at ",Path(output_folder,"generated_images.npy"))
    elif output_type in ["png","jpg"]:
        for idx,image in enumerate(generated_images):
            torchvision.utils.save_image(image, os.path.join(output_folder, f'{idx}.{output_type}'))         
        print("Saved images as png filed in ",output_folder)

    print("Finished generating images")


if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Generate random images from gaussian noise in the VAE latent space')
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path of the model to save.")
    parser.add_argument("--nb_images", type=int, default=1024,help="Nb of images to generate")
    parser.add_argument("--output_type", type=str, default="npy", help="Type of output (npy file, png files...)")
    parser.add_argument("--output_folder", type=str, default="./", help="Output folder where to save the images")
    parser.add_argument("--custom_filename", type=str, default="./", help="Customise the name of the file")

    args=parser.parse_args()
    model_path=args.model_path
    nb_images=args.nb_images
    output_type=args.output_type
    output_folder=args.output_folder
    custom_filename=args.custom_filename

    generate_images_func(model_path=model_path,nb_images=nb_images,
                    output_type=output_type,output_folder=output_folder)

    

