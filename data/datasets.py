"""
Datasets and functions to load them.

Authors: Mukund Varma T, Nishant Prabhu
"""
import random
import os
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets
import re

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

sst1 = {"classes": 5}

sst2 = {"classes": 2}

DATASET_HELPER = {"cifar10": cifar10, "cifar100": cifar100, "sst1": sst1, "sst2": sst2}


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

    if name in ["cifar10", "cifar100"]:
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

        return ImageDataset(root=dataroot, train=split == "train", transforms=transforms, return_items=return_items)

    else:

        class SSTDataset(Dataset):
            def __init__(self, root, transforms, return_items, split):
                try:
                    with open(os.path.join(root, "db.pickle"), "rb") as handle:
                        pickled_data = pickle.load(handle)
                except:
                    if not os.path.exists(root):
                        os.makedirs(root)
                    if not os.path.exists(os.path.join(root, "SST.zip")):
                        os.system(
                            f"wget -q --show-progress 'http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip' -O {os.path.join(root, 'SST.zip')}"
                        )
                    if not os.path.exists(os.path.join(root, "glove.zip")):
                        os.system(
                            f"wget -q --show-progress 'http://nlp.stanford.edu/data/glove.6B.zip' -O {os.path.join(root, 'glove.zip')}"
                        )

                    os.system(f"rm -f {os.path.join(root, '*.txt')}")
                    os.system(f"unzip -j -q {os.path.join(root, 'SST.zip')} 'stanfordSentimentTreebank/*' -d {root}")
                    os.system(f"unzip -j -q {os.path.join(root, 'glove.zip')} 'glove.6B.300d.txt' -d {root}")
                    embeddings, w2ind, train_db, val_db, test_db = self.preprocess(root)
                    os.system(f"rm -f {os.path.join(root, '*.txt')}")

                    pickled_data = {"w2ind": w2ind, "train": train_db, "val": val_db, "test": test_db}
                    torch.save(embeddings, os.path.join(root, "glove.pth"))
                    with open(os.path.join(root, "db.pickle"), "wb") as handle:
                        pickle.dump(pickled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                self.w2ind = pickled_data["w2ind"]
                db = pickled_data[split]
                self.sent = db["sentence"]
                self.targets = db["sentiment_label"]
                self.transforms = transforms
                self.return_items = return_items

            def preprocess(self, root):
                # load all the data
                db_sent = pd.read_csv(os.path.join(root, "datasetSentences.txt"), sep="\t")
                db_dict = pd.read_csv(
                    os.path.join(root, "dictionary.txt"), sep="|", header=None, names=["sentence", "phrase ids"]
                )
                db_split = pd.read_csv(os.path.join(root, "datasetSplit.txt"), sep=",")
                sent_labels = pd.read_csv(os.path.join(root, "sentiment_labels.txt"), sep="|")
                dataset = pd.merge(pd.merge(pd.merge(db_sent, db_split), db_dict), sent_labels)

                def label(db_name, sent_label):
                    if db_name == "sst1":
                        if sent_label <= 0.2:
                            return 0  # very negative
                        elif sent_label <= 0.4:
                            return 1  # negative
                        elif sent_label <= 0.6:
                            return 2  # neutral
                        elif sent_label <= 0.8:
                            return 3  # positive
                        elif sent_label <= 1:
                            return 4  # very positive
                    else:
                        if sent_label <= 0.4:
                            return 0  # negative
                        elif sent_label > 0.6:
                            return 1  # positive
                        else:
                            return -1  # drop neutral

                def filter_text(s):
                    # make lower case
                    s = s.lower()
                    # remove spaces between ' and alphabets
                    s = re.sub(r"\s*(')\s*", r"\1", s)
                    s = re.sub(r"\s(')", r"\1", s)
                    # remove punctuations
                    s = re.sub(r"[^\w\s]", "", s)
                    # split text
                    s = s.split(" ")
                    return s

                # convert labels
                dataset["sentiment_label"] = dataset["sentiment values"].apply(
                    lambda x: label("sst-1" if "sst-1" in root else "sst-2", x)
                )
                # remove all the nuetral labels
                dataset = dataset[dataset["sentiment_label"] != -1]
                # some text filtration
                dataset["sentence"] = dataset["sentence"].apply(lambda s: filter_text(s))

                # open glove embeddings and set word to indx mapping
                words = []
                embeddings = []
                for line in open(os.path.join(root, "glove.6B.300d.txt"), "r", encoding="utf-8"):
                    line = line.split(" ")
                    word = line[0]
                    embed = np.array(list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:]))))
                    embeddings.append(embed)
                    words.append(word)

                # create word to indx mapping
                w2ind = {k: v + 1 for v, k in enumerate(words)}
                w2ind["<unk>"] = len(w2ind) + 1
                w2ind["<pad>"] = 0

                # for <unk> and <pad>
                random_init = np.random.uniform(-0.5 / embed.shape[0], 0.5 / embed.shape[0], size=(300,))
                embeddings.insert(0, random_init)
                embeddings.append(random_init)
                embeddings = torch.FloatTensor(np.array(embeddings))

                # get train, val and test splits
                train_db = dataset[dataset["splitset_label"] == 1][["sentence", "sentiment_label"]]
                val_db = dataset[dataset["splitset_label"] == 2][["sentence", "sentiment_label"]]
                test_db = dataset[dataset["splitset_label"] == 3][["sentence", "sentiment_label"]]

                train_db = train_db.reset_index(drop=True)
                val_db = val_db.reset_index(drop=True)
                test_db = test_db.reset_index(drop=True)

                return embeddings, w2ind, train_db, val_db, test_db

            def __getitem__(self, indx):
                sent = self.sent[indx]
                target = self.targets[indx]
                print("-------------")
                # Perform transformations
                data = {}
                for key, transform in self.transforms.items():
                    transformed_sent = transform(sent)
                    print(transformed_sent)
                    data[key] = torch.LongTensor(
                        [self.w2ind.get(word, self.w2ind["<unk>"]) for word in transformed_sent]
                    )
                data["target"] = target
                print("-------------")
                exit()
                return {key: data[key] for key in self.return_items}

            def __len__(self):
                return len(self.sent)

        return SSTDataset(root=dataroot, split=split, transforms=transforms, return_items=return_items)


class NeighborDataset:
    def __init__(self, input_dataset, neighbor_indices):
        self.input_dataset = input_dataset
        if "input" not in list(self.input_dataset.transforms.keys()):
            raise ValueError("input key not found in transforms")
        self.input_dataset.return_items = ["input", "target"]  # sanity check
        self.nbr_indices = np.load(neighbor_indices)
        self.return_items = ["anchor", "neighbor", "target"]

    def __getitem__(self, i):
        # Get anchor and choose one of the possible neighbors
        anchor = self.input_dataset[i]
        pos_nbrs = self.nbr_indices[i]
        nbr_idx = random.choices(pos_nbrs, k=1)[0]
        nbr = self.input_dataset[nbr_idx]
        return {
            "anchor": anchor["input"],
            "neighbor": nbr["input"],
            "target": anchor["target"],
        }

    def __len__(self):
        return len(self.input_dataset)


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


class SentPadCollate:
    def __init__(self, pad_indx, return_items):
        self.pad_indx = pad_indx
        self.return_items = return_items

    def __call__(self, batch):
        collate_batch = {}
        for key in self.return_items:
            current_batch = [item[key] for item in batch]
            if key == "target":
                collate_batch["target"] = torch.LongTensor(current_batch)
            else:
                collate_batch[key] = torch.nn.utils.rnn.pad_sequence(
                    current_batch, batch_first=True, padding_value=self.pad_indx
                )
        return collate_batch


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
