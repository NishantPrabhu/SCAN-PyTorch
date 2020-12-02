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
from scipy.optimize import linear_sum_assignment

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
    
# Clustering head
class ClusteringHead(nn.Module):
    def __init__(self, in_dim=512, n_clusters=10, n_heads=1):
        super(ClusteringHead, self).__init__()
        self.W = nn.ModuleList([nn.Linear(in_dim, n_clusters) for _ in range(n_heads)])
    
    def forward(self, x):
        return [w(x) for w in self.W]

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
        indices = (np.argsort(cosine_distances(z,z), axis=1)[:,0:topk+1]).astype(np.int32)
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
    
    def find_neighbours(self, data_loader, img_key, f_name, topk=20):
        pbar = tqdm(total=len(data_loader), desc=f"building feature vectors")
        f_vecs = []
        for data in data_loader:
            img = data[img_key].to(self.device)
            with torch.no_grad():
                z = self.proj_head(self.enc(img))
            z = F.normalize(z, p=2, dim=-1)
            f_vecs.extend(z.cpu().detach().numpy())
            pbar.update(1)
        pbar.close()
        f_vecs = np.array(f_vecs)
        neighbour_indices = []
        pbar = tqdm(total=len(f_vecs)//100, desc=f"finding neighbours")
        ind = 0
        while ind<len(f_vecs):
            n = np.argsort(cosine_distances(f_vecs[ind:ind+100], f_vecs), axis=1)[:,1:topk+1]
            neighbour_indices.extend(n.astype(np.int32))
            pbar.update(1)
            ind += 100
        pbar.close()
        neighbour_indices = np.array(neighbour_indices)
        np.save(f_name, neighbour_indices)

def hungarian_match(preds, targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((preds == c1) * (targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

# scan model
class SCAN():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nfound device {torch.cuda.get_device_name(0)}\n")

        self.enc = Encoder(**config["enc"]).to(self.device)
        utils.print_network(model=self.enc, name="Encoder")
        self.cluster_head = ClusteringHead(**config["cluster_head"]).to(self.device)
        utils.print_network(model=self.cluster_head, name="Clustering Head")
        
        self.lr = config["scan_optim"]["lr"]
        self.optim = utils.get_optim(config=config["scan_optim"], parameters=list(self.enc.parameters())+list(self.cluster_head.parameters()))
        self.lr_scheduler, self.warmup_epochs = utils.get_scheduler(config={**config["scan_scheduler"], "epochs": config["epochs"]}, optim=self.optim)
        self.criterion = losses.ScanCriterion(**config["criterion"])
    
    def train_one_step(self, data):
        
        anchor_img, neighbour_img = data["anchor_img"].to(self.device), data["neighbour_img"].to(self.device)
        
        anchor_out = self.cluster_head(self.enc(anchor_img))
        neighbour_out = self.cluster_head(self.enc(neighbour_img))
        
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchor, neighbour in zip(anchor_out, neighbour_out):
            t_l, c_l, e_l = self.criterion(anchor, neighbour)
            total_loss.append(t_l)
            consistency_loss.append(c_l)
            entropy_loss.append(e_l)

        loss = torch.sum(torch.stack(total_loss, dim=0))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        metrics = {}
        for i in range(len(total_loss)):
            metrics[f"h{i}_t_loss"] = total_loss[i].item()
            metrics[f"h{i}_ent_loss"] = entropy_loss[i].item()
            metrics[f"h{i}_consis_loss"] = consistency_loss[i].item()
        
        return metrics
    
    def validate(self, epoch, val_loader):
        loss_cntr = {}
        total_loss_cntr = []
        pred_labels = []
        target_labels = []
        pbar = tqdm(total=len(val_loader), desc=f"valid - {epoch}")
        for indx, data in enumerate(val_loader):
            anchor_img, neighbour_img = data["anchor_img"].to(self.device), data["neighbour_img"].to(self.device)
            with torch.no_grad():
                anchor_out = self.cluster_head(self.enc(anchor_img))
                neighbour_out = self.cluster_head(self.enc(neighbour_img))

            pred_labels.extend(np.concatenate([o.argmax(dim=1).unsqueeze(1).detach().cpu().numpy() for o in anchor_out], axis=1))
            target_labels.extend(data["target"].numpy())
            
            total_loss, consistency_loss, entropy_loss = [], [], []
            for anchor, neighbour in zip(anchor_out, neighbour_out):
                t_l, c_l, e_l = self.criterion(anchor, neighbour)
                total_loss.append(t_l.item())
                consistency_loss.append(c_l.item())
                entropy_loss.append(e_l.item())
            
            for i in range(len(total_loss)):
                if indx == 0:
                    loss_cntr[f"h{i}_t_loss"] = [total_loss[i]]
                    loss_cntr[f"h{i}_ent_loss"] = [entropy_loss[i]]
                    loss_cntr[f"h{i}_consis_loss"] = [consistency_loss[i]]
                else:
                    loss_cntr[f"h{i}_t_loss"].append(total_loss[i])
                    loss_cntr[f"h{i}_ent_loss"].append(entropy_loss[i])
                    loss_cntr[f"h{i}_consis_loss"].append(consistency_loss[i])
            total_loss_cntr.append(np.array(total_loss).reshape(1,-1))
            pbar.update(1)
        pbar.close()
        loss_cntr = {key: np.mean(value) for key, value in loss_cntr.items()}
        total_loss_cntr = np.mean(np.concatenate(total_loss_cntr, axis=0), axis=0)
        best_head_indx = np.argmin(total_loss_cntr) 
        
        pred_labels = np.array(pred_labels)[:,best_head_indx]
        target_labels = np.array(target_labels)
        match = hungarian_match(pred_labels, target_labels, len(np.unique(pred_labels)), len(np.unique(target_labels))) 
        print(match)
        remapped_preds = np.zeros(len(pred_labels))
        for pred_i, target_i in match:
            remapped_preds[pred_labels == int(pred_i)] = int(target_i)
        
        cls_acc = {}
        for i in np.unique(remapped_preds):
            indx = remapped_preds==i
            cls_acc[f"class_{i}_acc"] = (remapped_preds[indx] == target_labels[indx]).sum() / len(remapped_preds[indx])
        log = f""
        for key, value in {**loss_cntr, **cls_acc}.items():
            log += f"{key} - {round(value,2)} "
        print(log)
        return {**loss_cntr, "acc": np.mean(list(cls_acc.values()))}

    # save encoder and clustering head
    def save(self, output_dir, prefix):
        torch.save(self.enc.state_dict(), os.path.join(output_dir, f"{prefix}_encoder.ckpt"))
        torch.save(self.cluster_head.state_dict(), os.path.join(output_dir, f"{prefix}_cluster_head.ckpt"))

# selflabel model
class SelfLabel():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nfound device {torch.cuda.get_device_name(0)}\n")

        self.enc = Encoder(**config["enc"]).to(self.device)
        utils.print_network(model=self.enc, name="Encoder")
        self.cluster_head = ClusteringHead(**config["cluster_head"]).to(self.device)
        utils.print_network(model=self.cluster_head, name="Clustering Head")
        
        self.lr = config["selflabel_optim"]["lr"]
        self.optim = utils.get_optim(config=config["selflabel_optim"], parameters=list(self.enc.parameters())+list(self.cluster_head.parameters()))
        self.lr_scheduler, self.warmup_epochs = utils.get_scheduler(config={**config["selflabel_scheduler"], "epochs": config["epochs"]}, optim=self.optim)
        self.criterion = losses.SelflabelCriterion(**config["criterion"])
    
    def train_one_step(self, data):
        
        anchor, anchor_aug = data["anchor"].to(self.device), data["anchor_aug"].to(self.device)
        
        anchor = self.cluster_head(self.enc(anchor))[0]
        anchor_aug = self.cluster_head(self.enc(anchor_aug))[0]
        
        loss = self.criterion(anchor, anchor_aug)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return {"loss": loss.item()}
    
    def validate(self, epoch, val_loader):
        pred_labels = []
        target_labels = []
        pbar = tqdm(total=len(val_loader), desc=f"valid - {epoch}")
        for indx, data in enumerate(val_loader):
            img = data["img"].to(self.device)
            with torch.no_grad():
                img_out = self.cluster_head(self.enc(img))[0]

            pred_labels.extend(img_out.argmax(dim=1).cpu().detach().numpy())
            target_labels.extend(data["target"].numpy())
            pbar.update(1)
        pbar.close()
        
        pred_labels = np.array(pred_labels)
        target_labels = np.array(target_labels)
        match = hungarian_match(pred_labels, target_labels, len(np.unique(pred_labels)), len(np.unique(target_labels))) 
        print(match)
        remapped_preds = np.zeros(len(pred_labels))
        for pred_i, target_i in match:
            remapped_preds[pred_labels == int(pred_i)] = int(target_i)
        
        cls_acc = {}
        for i in np.unique(remapped_preds):
            indx = remapped_preds==i
            cls_acc[f"class_{i}_acc"] = (remapped_preds[indx] == target_labels[indx]).sum() / len(remapped_preds[indx])
        log = f""
        for key, value in cls_acc.items():
            log += f"{key} - {round(value,2)} "
        print(log)
        return {"acc": np.mean(list(cls_acc.values()))}

    # save encoder and clustering head
    def save(self, output_dir, prefix):
        torch.save(self.enc.state_dict(), os.path.join(output_dir, f"{prefix}_encoder.ckpt"))
        torch.save(self.cluster_head.state_dict(), os.path.join(output_dir, f"{prefix}_cluster_head.ckpt"))