import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import datasets
import utils
import losses
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_distances

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
        return F.log_softmax(self.W1(x), -1)
    
# simclr model
class SimCLR():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nfound device {torch.cuda.get_device_name(0)}\n")
        
        # model - encoder and projection head
        self.enc = Encoder(**config["enc"]).to(self.device)
        utils.print_network(model=self.enc, name="Encoder")
        self.proj_head = ProjectionHead(**config["proj_head"]).to(self.device)
        utils.print_network(model=self.proj_head, name="Project Head")
        self.proj_head.apply(init_weights)
        
        # optimizer, scheduler, criterion
        self.lr = config["simclr_optim"]["lr"]
        self.optim = utils.get_optim(config=config["simclr_optim"], parameters=list(self.enc.parameters())+list(self.proj_head.parameters()))
        self.lr_scheduler, self.warmup_epochs = utils.get_scheduler(config={**config["simclr_scheduler"], "epochs": config["epochs"]}, optim=self.optim)
        self.criterion = losses.SimclrCriterion(config["batch_size"], **config["criterion"])
    
    # train one step (called from trainer class)
    def train_one_step(self, data):
        
        img_i, img_j = data["i"].to(self.device), data["j"].to(self.device)
        z_i = self.proj_head(self.enc(img_i))
        z_j = self.proj_head(self.enc(img_j))
        
        self.optim.zero_grad()
        loss = self.criterion(z_i, z_j)
        loss.backward()
        self.optim.step()
        
        return {"loss": loss.item()}
    
    # calculate acc
    @staticmethod
    def calculate_acc(z, targets, topk=20):
        indices = np.argsort(cosine_distances(z,z), axis=1)[:,0:topk+1]
        anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
        neighbor_targets = np.take(targets, indices[:,1:], axis=0)
        accuracy = np.mean(neighbor_targets == anchor_targets)
        return accuracy

    # acc based validation (quick)
    def validate(self, epoch, val_loader):
        # calculate val acc
        pbar = tqdm(total=len(val_loader), desc=f"val epoch - {epoch}")
        f_vecs, labels = [], []
        for data in val_loader:
            img, target = data["val_img"].to(self.device), data["target"]
            with torch.no_grad():
                z = self.proj_head(self.enc(img))
            z = F.normalize(z, p=2, dim=-1)
            f_vecs.extend(z.cpu().detach().numpy())
            labels.extend(target.numpy())
            pbar.update(1)
        f_vecs, labels = np.array(f_vecs), np.array(labels)
        val_acc = SimCLR.calculate_acc(f_vecs, labels)
        pbar.set_description(f"valid epoch: {epoch} acc: {round(val_acc, 4)}")
        pbar.close()
        return {"acc": val_acc}

    def linear_eval(self, train_loader, val_loader, output_dir):
        # initialize classification head
        cls_head = ClassificationHead(**self.config["cls_head"]).to(self.device)
        optim = utils.get_optim(config=self.config["cls_optim"], parameters=cls_head.parameters())
        scheduler, _ = utils.get_scheduler(config={**self.config["cls_scheduler"], "epochs": self.config["linear_eval_epochs"]}, optim=optim)
        
        # train and validate freezing encoder
        best_cls = 0
        pbar = tqdm(total=self.config["linear_eval_epochs"])
        for epoch in range(self.config["linear_eval_epochs"]):
            # train
            loss_cntr, acc_cntr = [], []
            for data in train_loader:
                img, target = data["train_img"].to(self.device), data["target"].to(self.device)
                with torch.no_grad():
                    h = self.enc(img)
                pred = cls_head(h)
                loss = F.nll_loss(pred, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pred_label = pred.argmax(1)
                acc = (pred_label==target).sum()/target.shape[0]
                loss_cntr.append(loss.item())
                acc_cntr.append(acc.item())
            train_loss, train_acc = np.mean(loss_cntr), np.mean(acc_cntr)
            # validate
            loss_cntr, acc_cntr = [], []
            for data in val_loader:
                img, target = data["val_img"].to(self.device), data["target"].to(self.device)
                with torch.no_grad():
                    h = self.enc(img)
                pred = cls_head(h)
                loss = F.nll_loss(pred, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pred_label = pred.argmax(1)
                acc = (pred_label==target).sum()/target.shape[0]
                loss_cntr.append(loss.item())
                acc_cntr.append(acc.item())
            val_loss, val_acc = np.mean(loss_cntr), np.mean(acc_cntr)
            
            pbar.set_description(f"Epoch - {epoch} train: loss - {round(train_loss,2)} acc - {round(train_acc,2)} val: loss - {round(val_loss,2)} acc - {round(val_acc,2)}")
            pbar.update(1)
            scheduler.step()
            
            if val_acc > best_cls:
                best_cls = val_acc
                torch.save(cls_head.state_dict(), os.path.join(output_dir, "cls_head.ckpt"))
        pbar.close()
        return {"train acc": train_acc, "best val acc": best_cls}
    
    # save encoder and projection head
    def save(self, output_dir, prefix):
        torch.save(self.enc.state_dict(), os.path.join(output_dir, f"{prefix}_encoder.ckpt"))
        torch.save(self.proj_head.state_dict(), os.path.join(output_dir, f"{prefix}_proj_head.ckpt"))