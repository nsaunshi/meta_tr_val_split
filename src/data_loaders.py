import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

import torchmeta
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid

DATA_DIR = '/n/fs/ptml/nsaunshi/meta_split/data/'

#### Omniglot data functions

def omniglot_loaders(args):
    tr_loader = omniglot_loader(args.N_way, args.tr_n_tr, args.tr_n_te, bs=args.batch_size, split='train')
    va_loader = omniglot_loader(args.N_way, args.te_n_tr, args.te_n_te, bs=args.batch_size, split='val')
    te_loader = omniglot_loader(args.N_way, args.te_n_tr, args.te_n_te, bs=args.batch_size, split='test')

    return tr_loader, va_loader, te_loader


def omniglot_kshotNway(k, N, bs=32, split='test'):
    return omniglot_loader(N, k, 1, bs=bs, split=split)


def omniglot_loader(N, n_tr, n_te, bs=32, split='train'):
    dataset = Omniglot(DATA_DIR,
                       num_classes_per_task=N,
                       transform=Compose([Resize(28), ToTensor()]),
                       target_transform=Categorical(num_classes=N),
                       class_augmentations=[Rotation([90, 180, 270])],
                       meta_split=split,
                       download=True)
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=n_tr, num_test_per_class=n_te)
    loader = BatchMetaDataLoader(dataset, batch_size=bs)
    return loader



#### MiniImagenet data functions

def miniimagenet_loaders(args):
    tr_loader = miniimagenet_loader(args.N_way, args.tr_n_tr, args.tr_n_te, bs=args.batch_size, split='train')
    va_loader = miniimagenet_loader(args.N_way, args.te_n_tr, args.te_n_te, bs=args.batch_size, split='val')
    te_loader = miniimagenet_loader(args.N_way, args.te_n_tr, args.te_n_te, bs=args.batch_size, split='test')

    return tr_loader, va_loader, te_loader


def miniimagenet_kshotNway(k, N, bs=16, split='test'):
    return miniimagenet_loader(N, k, 1, bs=bs, split=split)


def miniimagenet_loader(N, n_tr, n_te, bs=16, split='train'):
    dataset = MiniImagenet(DATA_DIR,
                           num_classes_per_task=N,
                           transform=Compose([Resize(84), ToTensor()]),
                           target_transform=Categorical(num_classes=N),
                           class_augmentations=[Rotation([90, 180, 270])],
                           meta_split=split,
                           download=True)
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=n_tr, num_test_per_class=n_te)
    loader = BatchMetaDataLoader(dataset, batch_size=bs)
    return loader



#### Sine data functions

def sine_loaders(args):
    tr_loader = sine_loader(args.tr_n_tr, args.tr_n_te, bs=args.batch_size, sigma=args.sigma)
    va_loader = sine_loader(args.te_n_tr, args.te_n_te, bs=args.batch_size, sigma=args.sigma)
    te_loader = sine_loader(args.te_n_tr, args.te_n_te, bs=args.batch_size, sigma=args.sigma)

    return tr_loader, va_loader, te_loader


def sine_kshotNway(k, bs=16):
    return sine_loader(k, 1, bs=bs)


def sine_loader(n_tr, n_te, T=700000, bs=16, sigma=0.):
    n_total = n_tr + n_te
    splitter = lambda dataset: ClassSplitter(dataset, shuffle=True, num_train_per_class=n_tr, num_test_per_class=n_te)
    dataset = torchmeta.toy.Sinusoid(
        n_total, num_tasks=T, noise_std=sigma, transform=None,
        target_transform=None, dataset_transform=splitter
    )
    loader = BatchMetaDataLoader(dataset, batch_size=bs)
    return loader



#### Simulation data functions
class SimulDataset(Dataset):
    def __init__(self, T, inp_dim, k, n_tr, n_te=None, sigma=0.):
        self.exists_test = (not n_te is None)

        self.X_tr = torch.stack([torch.randn(size=(n_tr, inp_dim)).float()
                                 for _ in range(T)])
        if self.exists_test:
            self.X_te = torch.stack([torch.randn(size=(n_te, inp_dim)).float()
                                     for _ in range(T)])

        self.clf = torch.stack([torch.randn(size=(inp_dim, 1)).float()
                                for _ in range(T)])
        self.clf /= np.sqrt(k)
        self.clf[:,k:] = 0.
        
        self.Y_tr = torch.stack([self.X_tr[i].mm(self.clf[i]) + sigma * torch.randn(size=(n_tr, 1))
                                 for i in range(T)])
        self.Y_te = torch.stack([self.X_te[i].mm(self.clf[i]) + sigma * torch.randn(size=(n_te, 1))
                                 for i in range(T)])
        

    def __getitem__(self, index):
        tr_data = (self.X_tr[index], self.Y_tr[index])
        te_data = (self.X_te[index], self.Y_te[index])
        clf = self.clf[index]
        return {'train': tr_data, 'test': te_data, 'clf': clf}
    
    def __len__(self):
        return len(self.X_tr)


def simul_loaders(args):
    tr_loader = simul_loader(args.tr_n_tr, args.tr_n_te, T=args.T, bs=args.batch_size, inp_dim=args.inp_dim, k=args.k, sigma=args.sigma)
    va_loader = simul_loader(args.te_n_tr, args.te_n_te, T=args.T, bs=args.batch_size, inp_dim=args.inp_dim, k=args.k, sigma=args.sigma)
    te_loader = simul_loader(args.te_n_tr, args.te_n_te, T=args.T, bs=args.batch_size, inp_dim=args.inp_dim, k=args.k, sigma=args.sigma)

    return tr_loader, va_loader, te_loader


def simul_kshotNway(k, bs=16):
    return simul_loader(k, 1, bs=bs)


def simul_loader(n_tr, n_te, T=1000, bs=16, inp_dim=100, k=5, sigma=0.):
    dataset = SimulDataset(T, inp_dim, k, n_tr, n_te=n_te, sigma=sigma)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    return loader
