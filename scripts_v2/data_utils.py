
import torch 
import torch.nn.functional as F 
import torchvision.transforms as T
from torchvision import datasets 
from tqdm import tqdm
from augment import Augment
import faiss
import random
import numpy as np 


datasets = {
    'cifar10': {'data': datasets.CIFAR10, 'n_classes': 10},
    'cifar100': {'data': datasets.CIFAR100, 'n_classes': 100},
    'stl10': {'data': datasets.STL10, 'n_classes': 10}
}

norms = {
    'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
    'stl10': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
}


class NeighborsDataset(torch.utils.data.Dataset):

    def __init__(self, name, neighbors_idx):
        super(NeighborsDataset, self).__init__()
        self.data = datasets[name]['data'](root='../data/{}'.format(name), train=True, transform=None, download=True)
        self.neighs = neighbors_idx
        self.anchor_transform = T.RandomResizedCrop(size=32, scale=(0.08, 1.0), ratio=(0.75, 1.3333))
        self.neighbor_transform = Augment(n=4)
        self.tensor_transform = T.Compose([T.ToTensor(), T.Normalize(norms[name]['mean'], norms[name]['std'])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        anchor, label = self.data[i]
        neighbor, _ = self.data[random.choice(self.neighs[i])]
        anchor_ = self.tensor_transform(self.anchor_transform(anchor))
        neighbor_ = self.tensor_transform(self.neighbor_transform(neighbor))

        return {'anchor': anchor_, 'neighbor': neighbor_, 'label': label}


class MemoryBank:

    """ Compiles dataset, generates dataloaders, generates embeddings """

    def __init__(self, size, dim, num_classes):
        self.size = size
        self.features = torch.FloatTensor(size, dim)
        self.targets = torch.LongTensor(size)
        self.dim = dim
        self.num_classes = num_classes 
        self.ptr = 0


    def mine_nearest_neighbors(self, k, compute_accuracy=True):
        features = self.features.cpu().numpy()
        dim = features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        _, idx = index.search(features, k+1)

        if compute_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, idx[:, 1:], axis=0) 
            anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
            accuracy = (neighbor_targets == anchor_targets).mean()
            return idx, accuracy
        else:
            return idx


    def reset(self):
        self.ptr = 0


    def update(self, features, targets):
        b = features.size(0)

        if (b + self.ptr) <= self.size:
            self.features[self.ptr : self.ptr+b].copy_(features.detach())
            self.targets[self.ptr : self.ptr+b].copy_(targets.detach())
            self.ptr += b
        else:
            split = self.size - self.ptr
            self.features[self.ptr : self.size].copy_(features[:split].detach())
            self.targets[self.ptr : self.size].copy_(targets[:split].detach())
            self.reset()
            self.features[self.ptr : self.ptr+b-split].copy_(features[split:].detach())
            self.targets[self.ptr : self.ptr+b-split].copy_(targets[split:].detach())
            self.ptr += b-split


    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda') if torch.cuda.is_available() else self.to('cpu')


class Scalar:

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.mean = 0
        self.last_value = None

    def update(self, x):
        self.last_value = x
        self.count += 1
        self.sum += x
        self.mean = self.sum/self.count