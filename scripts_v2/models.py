
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
from resnets import resnet18, resnet50


BACKBONES = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50
}


class SimCLR(nn.Module):

    def __init__(self, name="resnet18", pretrained=False, zero_init_residual=False):
        super(SimCLR, self).__init__()
        assert name in list(BACKBONES.keys())
        # start layers
        conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        BN1 = nn.BatchNorm2d(64)
        ReLU1 = nn.ReLU(inplace=True)
        
        # backbone
        resnet = BACKBONES[name](pretrained=pretrained)
        backbone = list(resnet.children())
        self.backbone = nn.Sequential(conv0, BN1, ReLU1, *backbone[4:len(backbone)-1])

        # initialise weights if not pretrained
        if pretrained == False:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
            # reference: https://arxiv.org/abs/1706.02677
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

    def forward(self, x, forward_pass='full'):
        if forward_pass == 'full':
            out = self.contrastive_head(self.backbone(x))
            out = F.normalize(out, dim=1)
            return out
        elif forward_pass == 'backbone':
            out = self.backbone(x)
            out = F.normalize(out, dim=1)
            return out