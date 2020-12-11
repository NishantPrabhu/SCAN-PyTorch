
"""
Datasets and functions to load them.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import torch
import random
import numpy as np
from PIL import Image
from torchvision import datasets 
from torch.utils.data import DataLoader, WeightedRandomSampler


DATASET_HELPER = {
    'cifar10': {'data': datasets.CIFAR10, 'classes': 10},
    'cifar100': {'data': datasets.CIFAR100, 'classes': 100},
    'stl10': {'data': datasets.STL10, 'classes': 10}
}


def sample_weights(labels):
    """
    Computes sample weights for sampler in dataloaders,
    based on class occurence.
    """
    cls_count = np.unique(labels, return_counts=True)[1]
    cls_weights = 1./torch.Tensor(cls_count)
    return cls_weights[list(map(int, labels))]


def get_dataset(config, split, transforms, return_items):
    """
    Generates a dataset object with required images and/or labels 
    transformed as specified.
    """
    name = config['dataset'].get('name', None)
    if name not in list(DATASET_HELPER.keys()):
        raise ValueError('Invalid dataset; should be one of (cifar10, cifar100, stl10)')
    base_class = DATASET_HELPER[name]['data']
    root = config['dataset'].get('root', './')

    # Image dataset class
    class ImageDataset(base_class):

        def __init__(self, root, transforms, return_items, train, download=True):
            super().__init__(root=root, train=train, download=download)
            self.transforms = transforms 
            self.return_items = return_items

        def __getitem__(self, i):
            # Load image and target
            img, target = self.data[i], self.targets[i]
            img = Image.fromarray(img)

            # Perform transformations 
            data = {}
            for key, transform in self.transforms.items():
                data[key] = transform(img)
            data['target'] = target
            return {key: data[key] for key in self.return_items}

    # Return dataset object
    return ImageDataset(root=root, train=split=='train', transforms=transforms, return_items=return_items)


class NeighborDataset:

    def __init__(self, img_dataset, neighbor_indices):
        self.img_dataset = img_dataset
        if 'img' not in list(self.img_dataset.transforms.keys()):
            raise ValueError('img key not found in transforms')
        self.img_dataset.return_items = ['img', 'target'] # sanity check
        self.nbr_indices = neighbor_indices

    def __getitem__(self, i):
        # Get anchor and choose one of the possible neighbors
        anchor = self.img_dataset[i]
        pos_nbrs = self.nbr_indices[i]
        nbr_idx = random.choices(pos_nbrs, k=1)[0]
        nbr = self.img_dataset[nbr_idx]
        return {'anchor': anchor['img'], 'neighbour': nbr['img'], 'target': anchor['target']}

    def __len__(self):
        return len(self.img_dataset)


def get_dataloader(config, dataset, weigh=False, shuffle=False):
    """ Returns a DataLoader with specified configuration """

    if weigh:
        weights = sample_weights(dataset.targets)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=shuffle)

