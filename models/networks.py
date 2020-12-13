
"""
Models for SimCLR, SCAN and linear evaluation.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import models


BACKBONES = {
    'resnet18': {'net': models.resnet18, 'dim': 512},
    'resnet50': {'net': models.resnet50, 'dim': 2048},
    'resnet101': {'net': models.resnet101, 'dim': 2048} 
}


class Encoder(nn.Module):

    def __init__(self, name='resnet18', pretrained=False, zero_init_residual=False):
        super(Encoder, self).__init__()
        assert name in BACKBONES.keys(), 'name should be one of (resnet18, resnet50, resnet101)'

        # Initial layers
        conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu1 = nn.ReLU(inplace=True)

        # Backbone
        resnet = BACKBONES[name]['net'](pretrained=pretrained)
        layers = list(resnet.children())
        self.backbone = nn.Sequential(conv0, bn1, relu1, *layers[4:len(layers)-1])
        self.backbone_dim = BACKBONES[name]['dim']

        # Initializations for untrained resnets
        if not pretrained:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Reference: https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.backbone.modules():
                    if isinstance(m, models.resnet.Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    if isinstance(m, models.resnet.BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out

def init_weights(m):
    """
    Weight initializations for a layer.
    """
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

class ProjectionHead(nn.Module):

    def __init__(self, in_dim=512, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.W1 = nn.Linear(in_dim, in_dim)
        self.BN1 = nn.BatchNorm1d(in_dim)
        self.ReLU = nn.ReLU()
        self.W2 = nn.Linear(in_dim, out_dim)
        self.BN2 = nn.BatchNorm1d(out_dim)
        self.apply(init_weights)

    def forward(self, x):
        out = self.BN2(self.W2(self.ReLU(self.BN1(self.W1(x)))))
        out = F.normalize(out, p=2, dim=1)
        return out


class ClassificationHead(nn.Module):

    def __init__(self, in_dim=512, n_classes=10):
        super(ClassificationHead, self).__init__()
        self.W1 = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.W1(x)


class ClusteringHead(nn.Module):

    def __init__(self, in_dim=512, n_clusters=10, heads=1):
        super(ClusteringHead, self).__init__()
        self.W = nn.ModuleList([nn.Linear(in_dim, n_clusters) for _ in range(heads)])

    def forward(self, x):
        return [w(x) for w in self.W]

