
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
from resnets import resnet18, resnet50


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
        features = self.contrastive_head(self.backbone_net(x))
        features = F.normalize(features, dim=1)
        return features



class ClusteringModel(torch.nn.Module):

    def __init__(self, backbone, n_clusters):
        super(ClusteringModel, self).__init__()

        self.backbone = backbone['backbone']
        self.cluster_head = torch.nn.Linear(backbone['dim'], n_clusters)

    def forward(self, x):
        x = self.backbone(x)
        out = self.cluster_head(x)
        return out
