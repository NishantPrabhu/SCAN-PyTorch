from torchvision import datasets, transforms
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw
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

class NeighbourDataset():
    def __init__(self, img_dataset, neighbour_indices):
        self.img_dataset = img_dataset
        if "img" not in list(self.img_dataset.transforms.keys()):
            raise ValueError("image dataset does not return required items")
        self.img_dataset.return_items = ["img", "target"] # sanity sakes
        self.neighbour_indices = neighbour_indices
    
    def __getitem__(self, indx):
        # get anchor data
        anchor = self.img_dataset[indx]
        
        # get possible neighbours and choose one
        possible_neighbours = self.neighbour_indices[indx]
        neighbour_indx = random.choices(possible_neighbours, k=1)[0]
        neighbour = self.img_dataset[neighbour_indx]
        return {"anchor_img": anchor["img"], "neighbour_img": neighbour["img"], "target": anchor["target"]}

    def __len__(self):
        return len(self.img_dataset)

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class Cutout(object):
    def __init__(self, n_cuts, max_len):
        self.n_cuts = n_cuts
        self.max_len = max_len
    
    def __call__(self, img):
        h, w = img.size()[1:3]
        cut_len = random.randint(1, self.max_len)
        mask = np.ones((h,w), np.float32)
        
        for _ in range(self.n_cuts):
            x,y = random.randint(0,w), random.randint(0,h)
            x_1 = np.clip(x-cut_len//2, 0, w)
            x_2 = np.clip(x+cut_len//2, 0, w)
            y_1 = np.clip(y-cut_len//2, 0, h)
            y_2 = np.clip(y+cut_len//2, 0, h)
            mask[y_1:y_2, x_1:x_2] = 0
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img*mask

# random augment class
class RandomAugment(object):
    def __init__(self, n_aug):
        self.n_aug = n_aug
        self.aug_list = [
            ("identity", 1, 1),
            ("autocontrast", 1, 1),
            ("equalize", 1, 1),
            ("rotate", -30, 30),
            ("solarize", 1, 1),
            ("color", 1, 1),
            ("contrast", 1, 1),
            ("brightness", 1, 1),
            ("sharpness", 1, 1),
            ("shear_x", -0.1, 0.1),
            ("shear_y", -0.1, 0.1),
            ("translate_x", -0.1, 0.1),
            ("translate_y", -0.1, 0.1),
            ("posterize", 1, 1),
        ]
            
    def __call__(self, img):
        aug_choices = random.choices(self.aug_list, k=self.n_aug)
        for aug, min_value, max_value in aug_choices:
            v = random.uniform(min_value, max_value)
            if aug == "identity":
                pass
            elif aug == "autocontrast":
                img = ImageOps.autocontrast(img)
            elif aug == "equalize":
                img = ImageOps.equalize(img)
            elif aug == "rotate":
                if random.random() > 0.5:
                    v = -v
                img = img.rotate(v)
            elif aug == "solarize":
                img = ImageOps.solarize(img, v)
            elif aug == "color":
                img = ImageEnhance.Color(img).enhance(v)
            elif aug == "contrast":
                img = ImageEnhance.Contrast(img).enhance(v)
            elif aug == "brightness":
                img = ImageEnhance.Brightness(img).enhance(v)
            elif aug == "sharpness":
                img = ImageEnhance.Sharpness(img).enhance(v)
            elif aug == "shear_x":
                if random.random() > 0.5:
                    v = -v
                img = img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))
            elif aug == "shear_y":
                if random.random() > 0.5:
                    v = -v
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))
            elif aug == "translate_x":
                if random.random() > 0.5:
                    v = -v
                v = v * img.size[0]
                img = img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
            elif aug == "translate_y":
                if random.random() > 0.5:
                    v = -v
                v = v * img.size[1]
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
            elif aug == "posterize":
                img = ImageOps.posterize(img, int(v))
            else:
                raise NotImplementedError(f"{aug} not implemented")
        return img

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
    "rand_aug": RandomAugment,
    "cutout": Cutout,
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