
from torchvision import datasets, transforms 
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import random 
import torch 
import numpy as np 


class GaussianBlur:

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma 

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img 


class Cutout:

    def __init__(self, n_cuts, maxlen):
        self.n_cuts = n_cuts 
        self.maxlen = maxlen 

    def __call__(self, img):
        """ img is a tensor """

        h, w = img.size()[1:3]
        cut_len = random.randint(1, self.maxlen)
        mask = np.ones((h, w), dtype=np.float32)

        for _ in range(self.n_cuts):
            x, y = random.randint(0, w), random.randint(0, h)
            x1 = np.clip(x-cut_len//2, 0, w)
            x2 = np.clip(x+cut_len//2, 0, w)
            y1 = np.clip(y-cut_len//2, 0, h)
            y2 = np.clip(y+cut_len//2, 0, h)
            mask[y1: y2, x1: x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)

        return mask * img 


class RandAugment:

    def __init__(self, n=4):
        self.n = n
        self.aug_list = [
            ('identity', 1, 1),
            ('autocontrast', 1, 1),
            ('equalize', 1, 1),
            ('rotate', -30, 30),
            ('solarize', 1, 1),
            ('color', 1, 1),
            ('contrast', 1, 1),
            ('brightness', 1, 1),
            ('sharpness', 1, 1),
            ('shear_x', -0.1, 0.1),
            ('shear_y', -0.1, 0.1),
            ('translate_x', -0.1, 0.1),
            ('translate_y', -0.1, 0.1),
            ('posterize', 1, 1)
        ]

    def __call__(self, img):
        """ img is a PIL Image """

        aug_choices = random.choices(self.aug_list, k=self.n)
        for aug, min_val, max_val in aug_choices:
            v = random.uniform(min_val, max_val)

            if aug == 'identity':
                pass
            elif aug == 'autocontrast':
                img = ImageOps.autocontrast(img)
            elif aug == 'equalize':
                img = ImageOps.equalize(img)
            elif aug == 'rotate':
                if random.random() > 0.5:
                    v = -v 
                img = img.rotate(v)
            elif aug == 'solarize':
                img = ImageOps.solarize(img)
            elif aug == 'color':
                img = ImageEnhance.Color(img).enhance(v)
            elif aug == 'contrast':
                img = ImageEnhance.Contrast(img).enhance(v)
            elif aug == 'brightness':
                img = ImageEnhance.Brightness(img).enhance(v)
            elif aug == 'sharpness':
                img = ImageEnhance.Sharpness(img).enhance(v)
            elif aug == 'shear_x':
                if random.random() > 0.5:
                    v = -v
                img = img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))
            elif aug == 'shear_y':
                if random.random() > 0.5:
                    v = -v
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))
            elif aug == 'translate_x':
                if random.random() > 0.5:
                    v = -v 
                v *= img.size[0]
                img = img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
            elif aug == 'translate_y':
                if random.random() > 0.5:
                    v = -v
                v *= img.size[1]
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
            elif aug == 'posterize':
                img = ImageOps.posterize(img, int(v))
            else:
                raise NotImplementedError(f'{aug} not implemented')
        
        return img
            

TRANSFORM_HELPER = {
    'gaussian_blur': GaussianBlur,
    'color_jitter': transforms.ColorJitter,
    'random_gray': transforms.RandomGrayscale,
    'random_crop': transforms.RandomCrop,
    'center_crop': transforms.CenterCrop,
    'resize': transforms.Resize,
    'random_flip': transforms.RandomHorizontalFlip,
    'to_tensor': transforms.ToTensor,
    'normalize': transforms.Normalize,
    'rand_aug': RandAugment,
    'cutout': Cutout
}

DATASETS = {
    'cifar10': {'data': datasets.CIFAR10, 'classes': 10},
    'cifar100': {'data': datasets.CIFAR100, 'classes': 100},
    'stl10': {'data': datasets.STL10, 'classes': 10}
}


def get_transform(config):
    transform = []

    for key, value in config.items():
        if value is not None:
            p = value.pop('apply_prob', None)
            tr = TRANSFORM_HELPER[key](**value)
            if p is not None:
                tr = transforms.RandomApply([tr], p=p)

        else:
            tr = TRANSFORM_HELPER[key]()
        transform.append(tr)

    return transforms.Compose(transform)


# Datasets 

def sample_weights(labels):
    # calculate sample weights (based on class occurence)
    cls_count = np.unique(labels, return_counts=True)[1]
    cls_weights = 1./torch.Tensor(cls_count) 
    return cls_weights[list(map(int, labels))]


def get_dataset(config, split, transforms, return_items):

    assert config['dataset']['name'] in DATASETS.keys(), 'invalid dataset'
    base_class = DATASETS[config['dataset']['name']]['data']

    class ImageDataset(base_class):

        def __init__(self, root, train=True, download=True, transforms=None, return_items=None):
            super().__init__(root=root, train=train, download=download)
            self.transforms = transforms
            self.return_items = return_items

        def __getitem__(self, i):
            img = self.data[i]
            target = self.targets[i]
            img = Image.fromarray(img)

            # Transforms
            data = {}
            for key, transform in self.transforms.items():
                data[key] = transform(img)
            data['target'] = target

            return {key: data[key] for key in self.return_items}

    return ImageDataset(root=config['dataset']['root'], train=split=='train', transforms=transforms, return_items=return_items) 


class NeighborDataset:

    def __init__(self, img_dataset, neighbor_idx):
        self.img_dataset = img_dataset
        if img not in img_dataset.transforms.keys():
            raise ValueError('Image dataset missing image key')
        self.img_dataset.return_items = ['img', 'target']
        self.neighbor_idx = neighbor_idx 

    def __getitem__(self, i):
        anchor = self.img_dataset[i]
        possible_neighbors = self.neighbor_idx[i]
        neigh_idx = random.choices(possible_neighbors, k=1)[0]
        neighbor = self.img_dataset[neigh_idx]
        return {'anchor_img': anchor['img'], 'neighbor_img': neighbor['img'], 'target': anchor['target']}   

    def __len__(self):
        return len(self.img_dataset)
