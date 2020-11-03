
"""
SimCLR and Clustering models

@author: Nishant Prabhu
"""

import os
import numpy as np 
from sklearn import metrics
from torchvision import models
from logging_utils import Scalar

import torch
import torch.nn as nn
import torch.nn.functional as F


backbones = {
    'resnet18': {'model': models.resnet18(pretrained=False), 'out_features': 512},
    'resnet34': {'model': models.resnet34(pretrained=False), 'out_features': 512},
    'resnet50': {'model': models.resnet50(pretrained=False), 'out_features': 2048}
}

class Backbone(torch.nn.Module):

    def __init__(self, name):
        super(Backbone, self).__init__()

        base = backbones[name]['model']
        conv_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn_1 = torch.nn.BatchNorm2d(64)
        maxpool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        flatten = torch.nn.Flatten()

        layers = list(base.children())
        self.backbone = torch.nn.Sequential(
            conv_1, 
            bn_1,
            torch.nn.ReLU(), 
            maxpool_1,
            *layers[4:len(layers)-1],
            flatten
        )

    def forward(self, x):
        return self.backbone(x)


class Head(torch.nn.Module):

    def __init__(self, in_features, n_clusters):
        super(Head, self).__init__()

        self.head = torch.nn.Sequential(
                torch.nn.Linear(in_features, n_clusters),
                torch.nn.Softmax(dim=-1)
        )
        self.best_head = 0
        self.loss = Scalar()

    def forward(self, x):
        out = self.head(x)
        return out


class ClusteringModel(torch.nn.Module):

    def __init__(self, name, n_clusters, n_heads, feature_dim=128, pretrained_model=None):
        super(ClusteringModel, self).__init__()
        
        self.n_heads = n_heads
        self.backbone = ContrastiveModel(dataset=name, head='mlp', features_dim=feature_dim)
        self.heads = torch.nn.ModuleList([Head(in_features=feature_dim, n_clusters=n_clusters) for _ in range(n_heads)])    

        if pretrained_model is not None:
            print("\n[INFO] Loading {} into backbone".format(pretrained_model.split('/')[-1]))
            self.backbone.load_state_dict(torch.load(pretrained_model))
        
        self.branches = torch.nn.ModuleList([torch.nn.Sequential(self.backbone, self.heads[i]) for i in range(n_heads)])


    def forward(self, x, forward_pass='full'):

        if forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [self.heads[i](x) for i in range(self.n_heads)]         

        elif forward_pass == 'full':
            out = [self.branches[i](x) for i in range(self.n_heads)]   
            
        elif 'branch_' in forward_pass:
            try:
                idx = int(forward_pass.split('_')[1])
            except:
                raise ValueError('Branch specification error')  

            out = self.branches[idx](x)   

        return out


    def backpropagate(self, losses):
        """
        losses must be a list-like object of len = n_heads
        """
        for i, l in enumerate(losses):
            self.heads[i].loss.update(l.item())
            l.backward()


# Resnet model for SimCLR for CIFAR dataset
# Adapted from https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/models/resnet_cifar.py

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return {'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 'dim': 512}

def resnet50():
    backbone = models.__dict__['resnet50']()
    backbone.fc = nn.Identity()
    return {'backbone': backbone, 'dim': 2048}


# Contrastive model for SimCLR

class ContrastiveModel(nn.Module):

    """
    Takes in a 4d tensor (image) and returns a (1, 128) vector
    consisting for SimCLR embeddings

    """

    def __init__(self, dataset, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()

        if dataset in ['cifar10', 'cifar100', 'stl10']:
            backbone = resnet18()
        else:
            backbone = resnet50()

        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(), 
                nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim=1)
        return features
