# Datascience Project 2 : VAE

Variational Auto Encoder using PyTorch on the MNIST dataset
- Benjamin
- Alik
- Mounir

## Ideas
- **Show the number of parameters for each model**
- Latent space visualisation while morphing (TSNE ? Impose 2D latent space ?)
- Adding "salt and pepper" noise to help better reconstruction --> **Mounir**
- Compare (with Frechet Inception) only dense layers and convolution layers (with a fixed number of parameters) --> **Alik**
- Tuning loss function (KL divergence vs Reconstruction Loss) --> **Benjamin**
- Testing different impacts of the latent space on the reconstruction --> **Benjamin**
- Visualisation of evolution of latent space through time
- (Bonus) Try to reconstruct a missing part of an image with a VAE ?

## Using the modules
### Train a model
Use the `Generate/train_model.py` file with the parameters specified. For example : `python3 Generate/train_model.py --z_dim 2 --nb_epochs 1 --model_folder ./models_lamsade/ --losses_path losses_lamsade/ --k1 0.5 --k2 0.5 `
The model will be saved in the specified folder and the losses as well.

## To do group
### Priority
- Top à faire)  sortir 1024 images du Dense (k1= 1/2, k2= 1/2)    (avant prez)     Mounir
- print paramètre du model  (avant prez)    Mounir pr Dense et BenJ Conv
- diapo à faire sur Overleaf   Alsyr 
### Secondary
- Amélioration du train : par défaut k1=k2=1/2
- Utilisation des images bruités  (avant prez) Mounir et BenJ
- Tuner la latent_dim (avant prez) et voir jusqu'où on peut aller     Alsyr
- normaliser les inputs??   (avant prez)    à voir 
- Tuner le k1 et k2 et voir les variations de performance (avant prez)   BenJ
- regarder le Beta-VAE/Design Tangal   (après prez)   à voir


## Computing the FID (Frechet Inception Distance)

Computes the distance from our generated data to the original data, using the Inception model and using the Frechet distance of the data distributions in the Inception feature space.

Frechet inception distance :

$$\boxed{FID^2 = ||\mu_1 – \mu_2||^2 + Tr(C_1 + C_2 – 2.sqrt(C_1.C_2))}$$
Where $\mu_i$ is the average of the distribution of dataset $i$ in the Inception feature space and $C_i$ is the covariance matrix of the dataset $i$

For an original model trained on 10 epochs, we obtain a `fid=24.76`

### Using the FID module
In order to compute the FID distance, first, generate the 1024 images with the model, export the image to numpy `generated1024.npy` serialisable object.
Then :

```python
from FID import FrechIncDist
import numpy as np

# load images

images_gen=np.load("../generated_data/Generated_1024.npy")
images_true=np.load("../generated_data/true_MNIST.npy")
# instanciate the FID object
Fid=FrechIncDist()
# compute the distance (takes ~30 seconds)
Fid.compute_FID_images(images_gen,images_true)
```
