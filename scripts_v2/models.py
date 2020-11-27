
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
from resnets import resnet18, resnet50


base_models = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50
}


class SimCLR(torch.nn.Module):

    def __init__(self, name="", feature_dim=128):
        super(SimCLR, self).__init__()
        assert name in base_models.keys(), 'name should be one of resnet18, resnet34, resnet50'
        
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        resnet = base_models[name](pretrained=False)
        backbone = list(resnet.children())

        self.backbone = nn.Sequential(conv1, bn1, nn.ReLU(), maxpool, *backbone[4:len(backbone)-1])

        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.flatten(out)
        out = self.head(out)
        out = F.normalize(out, dim=1)
        return out


class Classifier(torch.nn.Module):

    """ For evaluation purposes only """

    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)


class ClusteringModel(torch.nn.Module):

    def __init__(self, backbone, backbone_dim, num_clusters):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone
        self.cluster_head = nn.Linear(backbone_dim, num_clusters)

    def forward(self, x):
        return self.cluster_head(self.backbone(x))


class ContrastiveModel(torch.nn.Module):

    def __init__(self, data_name, head='mlp', feature_dim=128):
        super(ContrastiveModel, self).__init__()
        if data_name in ['cifar10', 'cifar100', 'stl10']:
            self.backbone = resnet18()['backbone']
            self.backbone_dim = 512
        else:
            self.backbone = resnet50()['backbone']
            self.backbone_dim = 2048

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, feature_dim)
        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(),
                nn.Linear(self.backbone_dim, feature_dim)
            )

    def forward(self, x):
        out = self.contrastive_head(self.backbone(x))
        out = F.normalize(out, dim=1)
        return out