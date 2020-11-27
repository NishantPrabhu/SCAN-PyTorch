import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import datasets
import utils
import losses
from tqdm import tqdm
import numpy as np

BACKBONES = {
    "resnet18": models.resnet18
}

class Encoder(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, zero_init_residual=False):
        super(Encoder, self).__init__()
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
        out = self.backbone[0:8](x)
        out = torch.flatten(out, 1)
        return out

# initialize weights to 0
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

# projection head
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.W1 = nn.Linear(in_dim, in_dim)
        self.BN1 = nn.BatchNorm1d(in_dim)
        self.ReLU = nn.ReLU()
        self.W2 = nn.Linear(in_dim, out_dim)
        self.BN2 = nn.BatchNorm1d(out_dim)
    
    def forward(self, x):
        return self.BN2(self.W2(self.ReLU(self.BN1(self.W1(x)))))      
    
# classification head
class ClassificationHead(nn.Module):
    def __init__(self, in_dim=512, n_classes=10):
        super(ClassificationHead, self).__init__()
        self.W1 = nn.Linear(in_dim, n_classes)
    
    def forward(self, x):
        return self.W1(x)
    
# simclr model
class SimCLR():
    def __init__(self, config):
        self.config = config
        # setup device (not multi-gpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"found device {torch.cuda.get_device_name(0)}")
        
        # define model
        self.enc = Encoder(**config["encoder"]).to(self.device)
        self.proj_head = ProjectionHead(**config["proj_head"]).to(self.device)
        self.proj_head.apply(init_weights)
        
        # optimizer, scheduler, criterion
        self.lr = config["simclr_optim"]["lr"]
        self.optim = utils.get_optim(config=config["simclr_optim"], parameters=list(self.enc.parameters())+list(self.proj_head.parameters()))
        self.scheduler, self.warmup_epochs = utils.get_scheduler(config={**config["simclr_scheduler"], "epochs": config["epochs"]}, optim=self.optim)
        self.criterion = losses.SimclrCriterion(config["batch_size"], **config["criterion"])
        
        self.val_best = 0
        
    def train_one_step(self, data):
        img_i, img_j = data["i"].to(self.device), data["j"].to(self.device)
        z_i = self.proj_head(self.enc(img_i))
        z_j = self.proj_head(self.enc(img_j))
        
        self.optim.zero_grad()
        loss = self.criterion(z_i, z_j)
        loss.backward()
        self.optim.step()
        
        return {"loss": loss.item()}
    
    def validate(self, train_loader, val_loader):
        cls_head = ClassificationHead(**self.config["cls_head"]).to(self.device)
        optim = utils.get_optim(config=self.config["cls_optim"], parameters=cls_head.parameters())
        scheduler, _ = utils.get_scheduler(config={**self.config["cls_scheduler"], "epochs": self.config["val_epochs"]}, optim=optim)
        criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(total=self.config["val_epochs"])
        patience = 0
        best = 0
        for epoch in range(self.config["val_epochs"]):
            loss_cntr, acc_cntr = [], []
            for data in train_loader:
                img, target = data["train"].to(self.device), data["target"].to(self.device)
                with torch.no_grad():
                    h = self.enc(img)
                pred = cls_head(h)
                loss = criterion(pred, target)
                loss.backward()
                optim.step()
                
                pred_label = pred.argmax(1)
                acc = (pred_label==target).sum()/target.shape[0]
                
                loss_cntr.append(loss.item())
                acc_cntr.append(acc.item())
            pbar.set_description(f"cls head train epoch: {epoch}, loss: {round(np.mean(loss_cntr), 4)}, acc: {round(np.mean(acc_cntr), 4)}")
            if np.mean(acc_cntr) > best:
                best = np.mean(acc_cntr)
                patience = 0
            else:
                patience += 1
            if patience > 10:
                break
            scheduler.step()
            pbar.update(1)
        pbar.close()
        train_acc = np.mean(acc_cntr)
        
        acc_cntr = []
        pbar = tqdm(total=len(val_loader))
        for data in val_loader:
            img, target = data["val"].to(self.device), data["target"].to(self.device)
            with torch.no_grad():
                h = self.enc(img)
                pred = cls_head(h)
            pred_label = pred.argmax(1)
            acc = (pred_label==target).sum()/target.shape[0]
            acc_cntr.append(acc.item())
            pbar.update(1)
        pbar.set_description(f"cls head test acc: {round(np.mean(acc_cntr), 4)}")
        test_acc = np.mean(acc_cntr)
        pbar.close()
        if test_acc > self.val_best:
            self.val_best = test_acc
            return {"train_acc": train_acc, "test_acc": test_acc}, True
        else:
            return {"train_acc": train_acc, "test_acc": test_acc}, False
    
    def save(self, file_name):
        torch.save(self.enc.state_dict(), file_name)