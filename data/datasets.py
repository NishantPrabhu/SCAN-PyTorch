"""
Datasets and functions to load them.

Authors: Mukund Varma T, Nishant Prabhu
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets

cifar10 = {
    "data": datasets.CIFAR10,
    "classes": 10,
    "norm": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
}

cifar100 = {
    "data": datasets.CIFAR100,
    "classes": 100,
    "norm": {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]},
}

DATASET_HELPER = {"cifar10": cifar10, "cifar100": cifar100}


def sample_weights(labels):
    """
    Computes sample weights for sampler in dataloaders,
    based on class occurence.
    """
    cls_count = np.unique(labels, return_counts=True)[1]
    cls_weights = 1.0 / torch.Tensor(cls_count)
    return cls_weights[list(map(int, labels))]


def get_dataset(name, dataroot, split, transforms, return_items):
    """
    Generates a dataset object with required images and/or labels
    transformed as specified.
    """
    base_class = DATASET_HELPER[name]["data"]

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
            data["target"] = target
            return {key: data[key] for key in self.return_items}

    # Return dataset object
    return ImageDataset(root=dataroot, train=split == "train", transforms=transforms, return_items=return_items)


class NeighborDataset:
    def __init__(self, img_dataset, neighbor_indices):
        self.img_dataset = img_dataset
        if "img" not in list(self.img_dataset.transforms.keys()):
            raise ValueError("img key not found in transforms")
        self.img_dataset.return_items = ["img", "target"]  # sanity check
        self.nbr_indices = np.load(neighbor_indices)

    def __getitem__(self, i):
        # Get anchor and choose one of the possible neighbors
        anchor = self.img_dataset[i]
        pos_nbrs = self.nbr_indices[i]
        nbr_idx = random.choices(pos_nbrs, k=1)[0]
        nbr = self.img_dataset[nbr_idx]
        return {
            "anchor": anchor["img"],
            "neighbor": nbr["img"],
            "target": anchor["target"],
        }

    def __len__(self):
        return len(self.img_dataset)


class RotNetCollate:
    def __init__(self, rotate_angles):
        self.rot_matrices = []
        for angle in rotate_angles:
            angle = torch.tensor(angle / 180 * np.pi)
            self.rot_matrices.append(
                torch.tensor([[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0]])
            )

    def rot_img(self, x, rot_matrix):
        rot_mat = rot_matrix[None, ...].repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def __call__(self, batch):
        assert "img" in batch[0].keys()
        imgs = torch.cat([b["img"].unsqueeze(0) for b in batch], axis=0)
        labels = torch.zeros(len(imgs) * len(self.rot_matrices))
        batch = []
        for i in range(len(self.rot_matrices)):
            labels[i * len(imgs) : (i + 1) * len(imgs)] = i
            batch.append(self.rot_img(imgs, self.rot_matrices[i]))
        # get tensors and shuffle
        imgs = torch.cat(batch, axis=0)
        labels = labels.long()
        shuffle_indices = torch.randperm(imgs.size()[0])
        data = {"img": imgs[shuffle_indices], "target": labels[shuffle_indices]}
        return data


def get_dataloader(dataset, batch_size, num_workers=1, shuffle=False, weigh=False, drop_last=False, collate_fn=None):
    """ Returns a DataLoader with specified configuration """

    if weigh:
        weights = sample_weights(dataset.targets)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
