from utils.common_config import get_model
import yaml
import torch
import numpy as np

p = yaml.safe_load(open("/home/sneezygiraffe/Unsupervised-Classification/configs/pretext/simclr_cifar10.yml", "r"))
model = get_model(p)

ckpt = torch.load("simclr_cifar-10.pth.tar")
model.load_state_dict(ckpt)

import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

from torchvision import datasets
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
dataset_helper = {
    "cifar10": datasets.CIFAR10
}

def get_dataset(params, mode, standard_transform, augment_transform):
    dataset_params = params.copy()
    name = dataset_params.pop("name")
    base = dataset_helper[name]

    class CustomDataset(base):
        def __init__(self, root, train=True, download=False, standard_transform=None, augment_transform=None):
            super().__init__(root=root, train=train, download=download)
            self.standard_transform = standard_transform
            self.augment_transform = augment_transform
        def __getitem__(self, indx):
            img, target = self.data[indx], self.targets[indx]
            img = Image.fromarray(img)
            out = []
            if self.standard_transform is not None:
                out.append(self.standard_transform(img))
            if self.augment_transform is not None:
                out.append(self.augment_transform(img))
            out.append(target)
            return out

    return CustomDataset(**dataset_params, train=mode=="train", standard_transform=standard_transform, augment_transform=augment_transform)

inv_normalize = None

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_transform(transform_params):
    transform = []
    transform_names = list(transform_params.keys())
    if "color_jitter" in transform_names:
        p = transform_params["color_jitter"].pop("p")
        color_jitter = transforms.ColorJitter(**transform_params["color_jitter"])
        random_color_jitter = transforms.Lambda(lambda x: color_jitter(x) if random.random()<p else x)
        transform.append(random_color_jitter)
    if "random_gray" in transform_names:
        random_gray = transforms.RandomGrayscale(**transform_params["random_gray"])
        transform.append(random_gray)
    if "gaussian_blur" in transform_names:
        gaussian_blur = GaussianBlur()
        random_gaussian_blur = transforms.Lambda(lambda x: gaussian_blur(x) if random.random()<p else x)
        transform.append(random_gaussian_blur)
    if "random_crop" in transform_names:
        random_crop = transforms.RandomResizedCrop(**transform_params["random_crop"])
        transform.append(random_crop)
    if "center_crop" in transform_names:
        center_crop = transforms.CenterCrop(**transform_params["center_crop"])
        transform.append(center_crop)
    if "resize" in transform_names:
        resize = transforms.Resize(**transform_params["resize"])
        transform.append(resize)
    if "horizontal_flip" in transform_names:
        random_horizontal_flip = transforms.RandomHorizontalFlip()
        transform.append(random_horizontal_flip)
    transform.append(transforms.ToTensor())
    if "normalize" in transform_names:
        transform.append(transforms.Normalize(**transform_params["normalize"]))
    global inv_normalize
    mean = transform_params["normalize"]["mean"]
    std = transform_params["normalize"]["std"]
    inv_normalize = transforms.Normalize(mean=[-mean[i]/std[i] for i in range(3)], std=[1/std[i] for i in range(3)])
    return transforms.Compose(transform)

device = torch.device("cuda")
classifier = Classifier(128, 10).to(device)
optimizer =  optim.Adam(classifier.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4)

config = yaml.safe_load(open("simclr.yaml", "r"))
standard_transform = get_transform(config["standard_transform"])
val_transform = get_transform(config["val_transform"])

train_dataset = get_dataset(config["dataset"], mode="train", standard_transform=standard_transform, augment_transform=None)
val_dataset = get_dataset(config["dataset"], mode="val", standard_transform=val_transform, augment_transform=None)
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
model = model.to(device)
total_epochs = 20
train_fvs = []
train_labels = []
model.eval()
for ep in range(total_epochs):
	mean_correct = []
	classifier.train()
	for i, (img, labels) in enumerate(train_dataloader):
	    try:
	        fv, labels = train_fvs[i].to(device), train_labels[i].to(device)
	    except:
	        img, labels = img.to(device), labels.to(device)
	        with torch.no_grad():
	            fv = model(img)
	        train_fvs.append(fv.detach().cpu())
	        train_labels.append(labels.detach().cpu())
	    
	    out = classifier(fv)
	    loss = F.nll_loss(out, labels.long())
	    pred_cls = out.max(1)[1]
	    correct = pred_cls.eq(labels.long().data).cpu().sum()
	    mean_correct.append(correct.item()/img.size()[0])
	    loss.backward()
	    optimizer.step()
	if ep % 1 == 0:
	    print(f"  > [{ep}/{total_epochs}] classifier train accuracy: {round(np.mean(mean_correct),4)}")

	classifier.eval()
	mean_correct = []
	test_fvs = []
	for img, labels in val_dataloader:
		img, labels = img.to(device), labels.to(device)
		with torch.no_grad():
		    fv = model(img)
		    out = classifier(fv)
		pred_cls = out.max(1)[1]
		correct = pred_cls.eq(labels.long().data).cpu().sum()
		mean_correct.append(correct.item()/img.size()[0])

	test_acc = np.mean(mean_correct)
	print(f"  > classifier val accuracy: {round(test_acc,4)}")




