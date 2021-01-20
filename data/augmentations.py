"""
Augmentation pipelines and functions to generate them.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw
from torchvision import transforms
import numpy as np
import random
import torch
from . import datasets
import nltk

try:
    from nltk.corpus import wordnet
    from nltk.corpus import stopwords

    stop_words = stopwords.words("english")
except:
    nltk.download("wordnet")
    nltk.download("stopwords")
    from nltk.corpus import wordnet
    from nltk.corpus import stopwords

    stop_words = stopwords.words("english")


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Cutout:
    def __init__(self, n_cuts=0, max_len=1):
        self.n_cuts = n_cuts
        self.max_len = max_len

    def __call__(self, img):
        h, w = img.shape[1:3]
        cut_len = random.randint(1, self.max_len)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_cuts):
            x, y = random.randint(0, w), random.randint(0, h)
            x1 = np.clip(x - cut_len // 2, 0, w)
            x2 = np.clip(x + cut_len // 2, 0, w)
            y1 = np.clip(y - cut_len // 2, 0, h)
            y2 = np.clip(y + cut_len // 2, 0, h)
            mask[y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class RandomAugment:
    def __init__(self, n_aug=4):
        self.n_aug = n_aug
        self.aug_list = [
            ("identity", 0, 1),
            ("autocontrast", 0, 1),
            ("equalize", 0, 1),
            ("rotate", -30, 30),
            ("solarize", 0, 256),
            ("color", 0.05, 0.95),
            ("contrast", 0.05, 0.95),
            ("brightness", 0.05, 0.95),
            ("sharpness", 0.05, 0.95),
            ("shear_x", -0.1, 0.1),
            ("shear_y", -0.1, 0.1),
            ("translate_x", -0.1, 0.1),
            ("translate_y", -0.1, 0.1),
            ("posterize", 4, 8),
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


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in " qwertyuiopasdfghjklzxcvbnm"])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


class SynonymReplacement:
    def __init__(self, ratio=0):
        self.ratio = ratio

    def __call__(self, word_list):
        n = int(self.ratio * len(word_list))
        not_stopwords = list(set([word for word in word_list if word not in stop_words]))
        rand_word_list = random.choices(not_stopwords, k=n)
        for word in rand_word_list:
            synonyms = get_synonyms(word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                word_list = [synonym if w == word else w for w in word_list]
        return word_list


class RandomDeletion:
    def __init__(self, keep_ratio=0):
        self.keep_ratio = keep_ratio

    def __call__(self, word_list):
        if len(word_list) == 1:
            return word_list
        n = int(self.keep_ratio * len(word_list))
        if n == 0:
            return word_list[random.randint(0, len(word_list) - 1)]
        else:
            return random.choices(word_list, k=n)


class RandomSwap:
    def __init__(self, ratio=0):
        self.ratio = ratio

    def swap_word(self, word_list):
        indx_1, indx_2 = random.randint(0, len(word_list) - 1), random.randint(0, len(word_list) - 1)
        word_1, word_2 = word_list[indx_1], word_list[indx_2]
        word_list[indx_1] = word_2
        word_list[indx_2] = word_1
        return word_list

    def __call__(self, word_list):
        n = int(self.ratio * len(word_list))
        for _ in range(n):
            word_list = self.swap_word(word_list)
        return word_list


class RandomInsert:
    def __init__(self, ratio=0):
        self.ratio = ratio

    def add_word(self, word_list):
        synonyms = []
        for word in word_list:
            synonyms.extend(get_synonyms(word))
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            indx = random.randint(0, len(word_list) - 1)
            word_list.insert(indx, synonym)
        return word_list

    def __call__(self, word_list):
        n = int(self.ratio * len(word_list))
        for _ in range(n):
            word_list = self.add_word(word_list)
        return word_list


class Identity:
    def __init__(self):
        pass

    def __call__(self, word_list):
        return word_list


# class RandomSentAugment:
#     def __init__(self, n_aug=4):
#         self.n_aug = n_aug
#         self.aug_list = [
#             ("identity", 0, 1),
#             ("synonym_replace", 0, 0.1),
#             ("random_delete", 0.8, 1),
#             ("random_swap", 0, 0.1),
#             ("random_insert", 0, 0.1),
#             ("random_crop", 0.8, 1),
#             ("pad_insert", 0, 0.2),
#             ("flip", 0, 1),
#         ]

#     def __call__(self, word_list):
#         aug_choices = random.choices(self.aug_list, k=self.n_aug)
#         for aug, min_value, max_value in aug_choices:
#             if len(word_list) == 1:
#                 return word_list
#             v = random.uniform(min_value, max_value)
#             if aug == "identity":
#                 pass
#             elif aug == "synonym_replace":
#                 n = int(v*len(word_list))
#                 not_stopwords = list(set([word for word in word_list if word not in stop_words]))
#                 rand_word_list = random.choices(not_stopwords, k=n)
#                 for word in rand_word_list:
#                     synonyms = get_synonyms(word)
#                     if len(synonyms) >= 1:
#                         synonym = random.choice(synonyms)
#                         word_list = [synonym if w == word else w for w in word_list]
#             elif aug == "random_delete":
#                 n = int(v*len(word_list))
#                 if n == 0:
#                     return word_list[random.randint(0, len(word_list)-1)]
#                 else:
#                     word_list = random.choices(word_list, k=n)
#             elif aug == "random_swap":
#                 n = int(v*len(word_list))
#                 for _ in range(n):
#                     indx_1, indx_2 = random.randint(0, len(word_list)-1), random.randint(0, len(word_list)-1)
#                     word_1, word_2 = word_list[indx_1], word_list[indx_2]
#                     word_list[indx_1] = word_2
#                     word_list[indx_2] = word_1
#             elif aug == "random_insert":
#                 n = int(v*len(word_list))
#                 for _ in range(n):
#                     synonyms = []
#                     for word in word_list:
#                         synonyms.extend(get_synonyms(word))
#                     if len(synonyms) >= 1:
#                         synonym = random.choice(synonyms)
#                         indx = random.randint(0, len(word_list)-1)
#                         word_list.insert(indx, synonym)
#             elif aug == "random_crop":
#                 n = int(v*len(word_list))
#                 indx = random.randint(0, len(word_list)-n)
#                 word_list = word_list[indx: indx+n]
#             elif aug == "pad_insert":
#                 n = int(v*len(word_list))
#                 indx = random.randint(0, len(word_list)-1)
#                 for _ in range(n):
#                     word_list.insert(indx, "<unk>")
#             elif aug == "flip":
#                 word_list.reverse()
#             else:
#                 raise NotImplementedError(f"{aug} not implemented")
#         return word_list

# Transformation helpers
IMG_TRANSFORMS = {
    "gaussian_blur": GaussianBlur,
    "color_jitter": transforms.ColorJitter,
    "random_gray": transforms.RandomGrayscale,
    "random_crop": transforms.RandomCrop,
    "random_resized_crop": transforms.RandomResizedCrop,
    "center_crop": transforms.CenterCrop,
    "resize": transforms.Resize,
    "random_flip": transforms.RandomHorizontalFlip,
    "to_tensor": transforms.ToTensor,
    "normalize": transforms.Normalize,
    "rand_aug": RandomAugment,
    "cutout": Cutout,
}

SENT_TRANSFORMS = {
    "synonym_replace": SynonymReplacement,
    "random_delete": RandomDeletion,
    "random_swap": RandomSwap,
    "random_insert": RandomInsert,
    "identity": Identity,
}


def get_transform(config, dataset):
    """
    Generates a torchvision.transforms.Compose pipeline
    based on given configurations.
    """
    transform = []

    if dataset in ["cifar10", "cifar100"]:
        # if normalization values are not given choose default values
        if config["normalize"] is None:
            config["normalize"] = datasets.DATASET_HELPER[dataset]["norm"]

        # Obtain transforms from config in sequence
        for key, value in config.items():
            if value is not None:
                p = value.pop("apply_prob", None)
                tr = IMG_TRANSFORMS[key](**value)
                if p is not None:
                    tr = transforms.RandomApply([tr], p=p)
            else:
                tr = IMG_TRANSFORMS[key]()
            transform.append(tr)
    else:
        # Obtain transforms from config in sequence
        for key, value in config.items():
            if value is not None:
                p = value.pop("apply_prob", None)
                tr = SENT_TRANSFORMS[key](**value)
                if p is not None:
                    tr = transforms.RandomApply([tr], p=p)
            else:
                tr = SENT_TRANSFORMS[key]()
            transform.append(tr)

    return transforms.Compose(transform)
