import torch 
import torchvision
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.optim import lr_scheduler


# IMPORT CUSTOM MODULES
import sys
sys.path.append("../")
from BAMVAE.MNISTLoader import MNIST_Loader
from BAMVAE.VAE_Moon import *
from BAMVAE.FID import FrechIncDist

def load_model(model):
    state = torch.load(os.path.join('checkpoint', 'bestmodel.pth'))
    model.load_state_dict(state['model_state_dict'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate VAE')
    #parser.add_argument("--batch_size", type=int, default=256,
    #                  help="The batch size to use for training.")
    #parser.add_argument("--num_channels", type=int, default=16,
    #                  help="Number of channels to use in the model.")
    #parser.add_argument("--num_features", type=int, default=256)
    #parser.add_argument("--num_steps", type=int, default=5,
    #                  help="Depth of the model.")
    #parser.add_argument("--batchnorm", default=False, action='store_false')
    #parser.add_argument("--actnorm", default=True, action='store_true')
    #parser.add_argument("--activation", type=str, default='relu') 
    #parser.add_argument("--n_samples", type=int, default=1024,
    #                  help="Number of generated samples.")

    args = parser.parse_args()
    """
    Initialize Hyperparameters
    """
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10


    # Data Pipeline
    print('Dataset loading...')
    x_dim = (1, 32, 32)
    #args.num_levels = int(np.log2(x_dim[1]))-1
    mnist=MNIST_Loader(bs=batch_size,
    root_folder='../mnist_data')
    train_loader,test_loader=mnist.__getdataloaders__()
    print('Dataset Loaded.')


    print('Model training...')

    # Model Pipeline
    #Guided Image Generation with Conditional Invertible Neural Networks
    #model = Glow(in_channels = 1,
    #              num_channels = args.num_channels,
    #              num_levels = args.num_levels, 
    #              num_steps = args.num_steps,
    #              params = args).cuda()
    #model = load_model(model)
    #model.eval()
    #model = DataParallel(model).cuda()
    latent_dims = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    model = net.train(num_epochs, train_loader, optimizer)

    print('Model trained.')

    print('Start Generating :')
    with torch.no_grad():
        for data in random.sample(list(test_loader), 1):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, logVAR = net(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            break
    # os.makedirs('samples', exist_ok=True)
    # with trange(1024, desc="Generated", unit="img") as te:
    #     for idx in te:
    #         sample = torch.randn(1,
    #                       x_dim[1]*x_dim[2],
    #                       1,
    #                       1).cuda()
    #         x, _ = model(sample, None, True)
    #         x = x[:, :, 2:30, 2:30]
    #         torchvision.utils.save_image(x, os.path.join('samples', f'{idx}.png'))            
