
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


def get_dataset(name, dataroot, split, transforms, return_items):
    """
    Generates a dataset object with required images and/or labels 
    transformed as specified.
    """
    base_class = DATASET_HELPER[name]['data']

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
    return ImageDataset(root=dataroot, train=split=='train', transforms=transforms, return_items=return_items)


class NeighborDataset:

    def __init__(self, img_dataset, neighbor_indices):
        self.img_dataset = img_dataset
        if 'img' not in list(self.img_dataset.transforms.keys()):
            raise ValueError('img key not found in transforms')
        self.img_dataset.return_items = ['img', 'target'] # sanity check
        self.nbr_indices = np.load(neighbor_indices)

    def __getitem__(self, i):
        # Get anchor and choose one of the possible neighbors
        anchor = self.img_dataset[i]
        pos_nbrs = self.nbr_indices[i]
        nbr_idx = random.choices(pos_nbrs, k=1)[0]
        nbr = self.img_dataset[nbr_idx]
        return {'anchor': anchor['img'], 'neighbor': nbr['img'], 'target': anchor['target']}

    def __len__(self):
        return len(self.img_dataset)


class RotnetDataset(torch.utils.data.Dataset):

    def __init__(self, config, split, transforms):
        
        name = config.get('dataset', None)
        root = config.get('root', './')
        assert name in DATASET_HELPER.keys(), f'dataset {name} is an impostor, use something else'
        self.transform = transforms['img'] 
        
        if split == 'train':
            self.data = DATASET_HELPER[name]['data'](root=root, train=True, transform=None, download=True)
        else:
            self.data = DATASET_HELPER[name]['data'](root=root, train=False, transform=None, download=True)
        

    def __len__(self):

        return len(self.data)


    def __getitem__(self, i):

        img, _ = self.data[i]
        rot0 = self.transform(img).unsqueeze(0)
        rot90 = self.transform(img.rotate(90)).unsqueeze(0)
        rot180 = self.transform(img.rotate(180)).unsqueeze(0)
        rot270 = self.transform(img.rotate(270)).unsqueeze(0)
        images = torch.cat((rot0, rot90, rot180, rot270), dim=0)
        labels = torch.LongTensor([0, 1, 2, 3])
        
        return {'img': images, 'target': labels}


def get_dataloader(dataset, batch_size, num_workers=1, shuffle=False, weigh=False, drop_last=False, collate_fn=None):
    """ Returns a DataLoader with specified configuration """

    if weigh:
        weights = sample_weights(dataset.targets)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, drop_last=drop_last, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)