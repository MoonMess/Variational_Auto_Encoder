import torch
from torchvision import datasets, transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MNIST_Loader(object):
    def __init__(self,bs=128,normalise=True,root_folder='./mnist_data/') -> None:
        """
        bs: batch size
        """
        #Normal MNIST
        # MNIST Dataset
        
        if normalise:
            mean=0.1307
            std=0.3081
            transform= transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
        else :
            transform=transforms.ToTensor()
        self.train_dataset = datasets.MNIST(root=root_folder, train=True, 
                    transform=transform, download=True)
        self.test_dataset = datasets.MNIST(root=root_folder, train=False, 
                    transform=transform, download=True)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                            batch_size=bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                            batch_size=bs, shuffle=False)

        #Noised MNIST for training
        # MNIST Noised Dataset
        transform_GaussianNoise=transforms.Compose([
                                transforms.ToTensor(),
                                #t.Normalize((0.1307,), (0.3081,)),
                                transforms.RandomApply([AddGaussianNoise(0., .4)], p=0.5)
                                ])

        self.noised_train_dataset = datasets.MNIST(root=root_folder, train=True, 
                    transform=transform_GaussianNoise, download=True)
        # Data Loader (Input Pipeline)
        self.noised_train_loader = torch.utils.data.DataLoader(dataset=self.noised_train_dataset, 
                                            batch_size=bs, shuffle=True)

    def __getdatasets__(self) :
        return self.train_dataset,self.test_dataset

    def __getdataloaders__(self):
        return self.train_loader,self.test_loader

    def __getnoiseddatasets__(self) :
        return self.noised_train_dataset

    def __getnoiseddataloaders__(self):
        return self.noised_train_loader
            