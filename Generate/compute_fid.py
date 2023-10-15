import sys
from pathlib import Path
import numpy as np
import argparse
sys.path.append("../BAMVAE")
sys.path.append(str(Path(__file__).parent.parent))
from BAMVAE.FID import FrechIncDist
import os


# load images
"""call by using
python3 Generate/compute_fid.py --images_1 './images_1_folder.npy' --images_2 './images_2_folder.npy' --output_folder --output_folder 'output_folder'
"""

def compute_fid_func(images_1,images_2,output_path,images_gen,images_true):
    # instanciate the FID object
    Fid=FrechIncDist()
    # compute the distance (takes ~30 seconds)
    fid_score=Fid.compute_FID_images(images_gen,images_true)
    print(f"Fid score between {images_1} and {images_2} = {fid_score}")
    # os.makedirs(output_path,exist_ok=True)
    if output_path:
        with open(output_path,"w") as f:
            f.write(str(fid_score))
        print(f"Saved fid score in a file at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computing the FID score between the selected generated images and MNIST images')
    parser.add_argument("--images_1", type=str, default=None, help="First set of images against which to compute FID")
    parser.add_argument("--images_2", type=str, default=None, help="Second set of images against which to compute FID")
    parser.add_argument("--output_path", type=str, default=None, help="Where to store a text file with the FID score")
    
    args=parser.parse_args()
    images_1=args.images_1
    images_2=args.images_2
    output_path=args.output_path
    images_gen=np.load(images_1)
    images_true=np.load(images_2)
    compute_fid_func(images_1=images_1,images_2=images_2,
                        output_path=output_path,images_gen=images_gen,
                        images_true=images_true)
    
    
