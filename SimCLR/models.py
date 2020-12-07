
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models 

BACKBONES = {
    'resnet18': {'net': models.resnet18, 'dim': 512},
    'resnet34': {'net': models.resnet34, 'dim': 512},
    'resnet50': {'net': models.resnet50, 'dim': 2048} 
}


class Encoder(nn.Module):

    def __init__(self, backbone='resnet18', pretrained=False, zero_init_residual=False):
        super(Encoder, self).__init__()
        assert backbone in BACKBONES.keys(), 'backbone should be one of (resnet18, resnet34, resnet50)'

        conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu1 = nn.ReLU(inplace=True)

        # Backbone
        resnet = BACKBONES[backbone]['net'](pretrained=pretrained)
        layers = list(resnet.children())
        self.backbone = nn.Sequential(conv0, bn1, relu1, *layers[4: len(layers)-1])
        self.backbone_dim = BACKBONES[backbone]['dim']

        if not pretrained:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.backbone.modules():
                    if isinstance(m, models.resnet.Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    if isinstance(m, models.resnet.BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, normalize=False):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        if normalize:
            out = F.normalize(out, p=2, dim=1)
        return out


class ProjectionHead(nn.Module):

    def __init__(self, in_dim=512, out_dim=128, linear=False):
        super(ProjectionHead, self).__init__()
        self.fc_1 = nn.Linear(in_dim, in_dim)
        self.bn_1 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(in_dim, out_dim)
        self.bn_2 = nn.BatchNorm1d(out_dim)
        if linear:
            self.head = nn.Sequential(self.fc_2, self.bn_2)
        else:
            self.head = nn.Sequential(self.fc_1, self.bn_1, self.relu, self.fc_2, self.bn_2)

    def forward(self, x):
        return self.head(x)


class LinearClassifier(nn.Module):

    def __init__(self, in_dim=512, out_dim=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

