
"""
Utility functions

@author: Nishant Prabhu
"""

# Dependencies
import os
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from termcolor import cprint

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


datasets = {
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
    'stl10': torchvision.datasets.STL10
}

@torch.no_grad()
def generate_embeddings(name, data, encoder, transform, save_root, regenerate=False):
    """
        Generates SimCLR feature vectors for each image in
        unaugmented dataset.

        Args:
            data <?>
                I don't remember what this is supposed to be, prolly a tensor
            encoder <torch.nn.Module>
                SimCLR encoder network from models.py
            transform <torchvision.transforms>
                Transformation pipeline for images
            save_root <os.path>
                Directory where pkl of generated vectors will be stored
            regenerate <bool>
                If set to True, generates the vectors again even if
                they have already been saved at save_root

        Returns:
            Tensor of SimCLR vectors for each image in dataset
    """

    cprint("\n[INFO] Generating embeddings", 'yellow')
    if not os.path.exists(save_root + 'simclr_{}_embeds'.format(name)) or regenerate:

        vectors = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in tqdm(range(len(data)), leave=False):
            img = data[i][0]
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_vector = encoder(img_tensor).detach().cpu()
            vectors.append(img_vector)

        # Convert to torch tensor for immediate use
        tensors_ = torch.cat(vectors, dim=0)
        torch.save(tensors_, save_root + 'simclr_{}_embeds'.format(name))
    else:
        print("Embeddings already exist at {}".format(save_root + 'simclr_{}_embeds'.format(name)))
        print("If you wish to generate them again, set regenerate=True")
        tensors_ = torch.load(save_root + 'simclr_{}_embeds'.format(name))

    return tensors_


def find_neighbors(embeddings, k):
    """
        Finds k nearest neighbors of each embedding

        Args:
            embeddings <torch.Tensor>
                Tensor of shape (dataset_size, embedding_size)
            k <int>
                Number of neighbors to mine for each sample

        Returns:
            Tensor of shape (dataset_size, k) with indices
            of k nearest neighbors of each sample
    """
    neighbor_indices = []

    for i in range(embeddings.shape[0]):
        sim_scores = embeddings @ embeddings[i].t()
        idx = torch.topk(sim_scores, k+1, dim=0)[1][1:]
        neighbor_indices.append(idx.reshape(1, -1))

    retval = torch.cat(neighbor_indices, dim=0)
    assert retval.shape == (embeddings.size(0), k), "Neighbors return shape error"

    return retval


def load_validation_data(name, transform):
    """
        Loads dataset from torchvision. Valid names are :
            cifar10, cifar100, stl10
    """
    val_ = datasets[name](root='../saved_data/datasets/'+name, train=False, transform=transform, download=True)
    return val_


class NeighborsDataset(Dataset):

    """
        For each image (anchor) in the original dataset, K of its neighbors
        are searched in SimCLR space and aggregated (making it K+1 images).
        One of these K neighbors is randomly selected. Both the anchor and the
        neighbor are transformed correctly and returned as sa dictionary

        Args:
            dataset <json like>
                JSON object containing image metadata
            embeddings <torch.Tensor>
                Tensor of SimCLR embeddings for images in dataset
            augment_fun <Augment obj>
                Augmentation object from augment.py
            transform <torchvision.transforms>
                Transformation pipeline for image and augmentations
            n_neighbors <int>
                Number of nearest neighbors to mine for each sample

        Return:
            Read description above
    """

    def __init__(self, name, embedding_net, augment_fun, transforms, n_neighbors):

        self.dataset = datasets[name](root='../saved_data/datasets/'+name, train=True, transform=None, download=True)
        self.embeddings = generate_embeddings(name, self.dataset, embedding_net, transforms['standard'], save_root='../saved_data/other/', regenerate=False)
        self.neighbor_indices = find_neighbors(self.embeddings, k=n_neighbors)
        self.augment_fun = augment_fun
        self.image_transform = transforms['standard']
        self.augment_transform = transforms['augment']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return_obj = {}
        sample = self.dataset[idx]
        neighbors = self.neighbor_indices[idx]

        # Choose random neighbor
        random_neighbor = neighbors[torch.randint(0, neighbors.size(0), (1,))]
        neighbor_meta = self.dataset[random_neighbor]

        return_obj['image'] = self.image_transform(self.augment_fun(sample[0]))
        return_obj['neighbor'] = self.augment_transform(self.augment_fun(neighbor_meta[0]))
        return_obj['label'] = sample[1]
        return_obj['possible_neigbors'] = neighbors

        return return_obj


def generate_data_loaders(name, batch_size, n_neighbors, transforms, embedding_net, augment_fun):
    """
        Torch dataloaders for image dataset.

        Args:
            name <str>
                One of cifar10, cifar100, stl10
            embedding_net <torch.Tensor>
                SimCLR network to generate embeddings
            n_neighbors <int>
                Number of neighbors to mine for each image
            augment_fun <Augment obj>
                Augmentation
            cutout_fun <Cutout obj>
                Cutout augmentation
            transforms <torchvision.Transforms>
                Dictionary of transformations for image and augmentations

        Returns:
            Train and validation data loaders
    """

    train_dset = NeighborsDataset(name, embedding_net, augment_fun, transforms, n_neighbors)
    val_dset = load_validation_data(name, transforms['standard'])

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, val_loader
