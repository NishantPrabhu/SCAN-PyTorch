from torchvision import datasets, transforms
from PIL import Image, ImageFilter
import random
import torch
import numpy as np

# helper for the datasets used in the paper
DATASET_HELPER = {
    "cifar10": datasets.CIFAR10
}

def sample_weights(labels):
    # calculate sample weights (based on class occurence)
    cls_count = np.unique(labels, return_counts=True)[1]
    cls_weights = 1./torch.tensor(cls_count) 
    return cls_weights[list(map(int, labels))]

def get_dataset(config, split, transforms, return_items):
    name = config["dataset"]
    if name not in list(DATASET_HELPER.keys()):
        raise ValueError(f"invalid dataset")
    base_class = DATASET_HELPER[name]
    
    # image dataset class
    class ImageDataset(base_class):
        def __init__(self, root, train=True, download=True, transforms=None, return_items=None):
            super().__init__(root=root, train=train, download=download)
            self.transforms = transforms
            self.return_items = return_items
        
        def __getitem__(self, indx):
            # load image and target
            img = self.data[indx]
            target = self.targets[indx]
            img = Image.fromarray(img)
            
            # perform transformations
            data = {}
            for key, transform in self.transforms.items():
                data[key] = transform(img)
            data["target"] = target
            return {key: data[key] for key in self.return_items}

    return ImageDataset(config["data_root"], train=split=="train", transforms=transforms, return_items=return_items)

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# image transform helper
TRANSFORM_HELPER = {
    "gaussian_blur": GaussianBlur,
    "color_jitter": transforms.ColorJitter,
    "random_gray": transforms.RandomGrayscale,
    "random_crop": transforms.RandomResizedCrop,
    "center_crop": transforms.CenterCrop,
    "resize": transforms.Resize,
    "random_flip": transforms.RandomHorizontalFlip,
    "to_tensor": transforms.ToTensor,
    "normalize": transforms.Normalize,
}

def get_transform(config):
    transform = []
    # get transforms based on configuration (order and parameters)
    for key, value in config.items():
        if value is not None:
            p = value.pop("apply_prob", None)
            tr = TRANSFORM_HELPER[key](**value)
            if p is not None:
                tr = transforms.RandomApply([tr], p=p)
        else:
            tr = TRANSFORM_HELPER[key]()
        transform.append(tr)
    return transforms.Compose(transform)