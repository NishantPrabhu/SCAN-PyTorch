
"""
Training functions and classes.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
from . import networks
from utils import common, train_utils, losses, eval_utils
import torch
import os
import torch.nn.functional as F
import numpy as np
from data.datasets import DATASET_HELPER

# =================================================================================================
# Simple architecture for Contrastive Learning of Visual Representations (SimCLR)
# =================================================================================================

class SimCLR:

    def __init__(self, config, device, output_dir):
        
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Models 
        self.encoder = networks.Encoder(**config['encoder']).to(self.device)
        common.print_network(self.encoder, 'Encoder')
        self.proj_head = networks.ProjectionHead(**config['proj_head']).to(self.device)
        common.print_network(self.proj_head, 'Projection Head')

        # Optimizer, scheduler and criterion
        self.optim = train_utils.get_optimizer(
            config = config['simclr_optim'], 
            params = list(self.encoder.parameters())+list(self.proj_head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**config['simclr_lr_scheduler'], 'epochs': config['epochs']},
            optimizer = self.optim
        )
        self.lr = config['simclr_optim']['lr']
        self.criterion = losses.SimclrLoss(**config['simclr_criterion'])
        
        self.best = 0
        
    def load_ckpt(self):
        
        ckpt = torch.load(os.path.join(self.output_dir, "last.ckpt"))
        self.encoder.load_state_dict(ckpt["enc"])
        self.proj_head.load_state_dict(ckpt["proj_head"])
        self.optim.load_state_dict(ckpt["optim"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        return ckpt["epoch"]

    def train_one_step(self, data):
        """ Trains model on one batch of data """

        img_i, img_j = data['i'].to(self.device), data['j'].to(self.device)
        zi = self.proj_head(self.encoder(img_i))
        zj = self.proj_head(self.encoder(img_j))

        self.optim.zero_grad()
        loss = self.criterion(zi, zj)
        loss.backward()
        self.optim.step()
        return {'loss': loss.item()}

    def save_ckpt(self, epoch):
        ckpt = {
            "enc": self.encoder.state_dict(),
            "proj_head": self.proj_head.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": epoch
        }
        if self.lr_scheduler:
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, "last.ckpt"))
    
    def save_model(self, fname):
        model_save = {
            "enc": self.encoder.state_dict()
        }
        torch.save(model_save, os.path.join(self.output_dir, fname))
    
    @torch.no_grad()
    def return_fvecs(self, data_loader):
        """ Build feature vectors """

        fvecs, gt = [], []

        for indx, data in enumerate(data_loader):
            img, target = data['img'].to(self.device), data['target']
            z = self.proj_head(self.encoder(img))
            z = F.normalize(z, p=2, dim=-1)
            fvecs.extend(z.detach().cpu().numpy())
            gt.extend(target.numpy())
            common.progress_bar(progress=indx/len(data_loader))
        common.progress_bar(progress=1)

        fvecs, gt = np.array(fvecs), np.array(gt)
        return fvecs, gt
    
    def validate(self, val_loader):
        # validate 
        
        fvecs, gt = self.return_fvecs(val_loader)
        neighbour_indices = eval_utils.find_neighbors(fvecs, topk=self.config["n_neighbors"])
        acc = eval_utils.compute_neighbour_acc(gt, neighbour_indices, topk=self.config["n_neighbors"])
        
        if acc >= self.best:
            self.save_model("best.pth")
            self.best = acc
        
        return {'neighbour acc': acc}

    def linear_eval(self, train_loader, val_loader):
        """ Evaluation of SimCLR vectors with linear classifier """

        clf_head = networks.ClassificationHead(
            in_dim=self.encoder.backbone_dim, 
            n_classes=DATASET_HELPER[self.config['dataset']]['classes']
        ).to(self.device)
        clf_optim = train_utils.get_optimizer(
            config={**self.config['clf_optimizer']}, 
            params=clf_head.parameters()
        )
        clf_scheduler, _ = train_utils.get_scheduler(
            config={**self.config['clf_scheduler'], 'epochs': self.config['linear_eval_epochs']}, 
            optimizer=clf_optim
        )
        criterion = losses.SupervisedLoss()

        best = 0
        train_metrics = common.AverageMeter()
        val_metrics = common.AverageMeter()
        for epoch in range(1, self.config['linear_eval_epochs']+1):   
            train_metrics.reset()
            val_metrics.reset()

            # Training 
            clf_head.train()
            epoch
            for batch in train_loader:
                img, target = batch['img'].to(self.device), batch['target']
                with torch.no_grad():    
                    h = self.encoder(img)
                
                # Loss and update
                out = clf_head(h)
                loss = criterion(out, target.to(self.device))
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

                # Accuracy
                pred = out.argmax(dim=1).cpu()
                correct = pred.eq(target.view_as(pred)).sum().item()
                train_metrics.add({"train acc": correct/len(batch), "train loss": loss.item()})

            # Validation
            clf_head.eval()
            for batch in val_loader:
                img, target = batch['img'].to(self.device), batch['target']
                with torch.no_grad():
                    h = self.encoder(img)
                
                # Loss
                out = clf_head(h).cpu()
                loss = criterion(out, target)
                
                # Accuracy
                pred = out.argmax(dim=1)
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_metrics.add({"val acc": correct/len(batch), "val loss": loss.item()})

            train_log = train_metrics.return_msg()
            val_log = val_metrics.return_msg()
            
            common.progress_bar(progress=epoch/self.config["linear_eval_epochs"], status=train_log+val_log)
            # Save model if better
            if  val_metrics.return_metrics()["val acc"] > best:
                torch.save(clf_head.state_dict(), os.path.join(self.output_dir, 'best_clf_head.pth'))
                best = val_metrics.return_metrics()["val acc"]
            
            if clf_scheduler is not None:
                clf_scheduler.step()

        return {'linear eval acc': best}

    def build_neighbors(self, data_loader, fname):
        fvecs, _ = self.return_fvecs(data_loader)
        neighbour_indices = eval_utils.find_neighbors(fvecs, topk=self.config["n_neighbors"])
        np.save(os.path.join(self.output_dir, fname), neighbour_indices)

# =============================================================================================
# Clustering model for SCAN
# =============================================================================================

class ClusteringModel:

    def __init__(self, config, device, output_dir):
        
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Models 
        save = torch.load(self.config["simclr_save"])
        self.encoder = networks.Encoder(**config['encoder']).to(self.device)
        self.encoder.load_state_dict(save["enc"])
        common.print_network(self.encoder, 'Encoder')
        self.cluster_head = networks.ClusteringHead(**config['cluster_head']).to(self.device)
        common.print_network(self.cluster_head, 'Clustering Head')

        # Optimizer, scheduler and criterion
        self.optim = train_utils.get_optimizer(
            config = config['cluster_optim'], 
            params = list(self.encoder.parameters())+list(self.cluster_head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**config['cluster_lr_scheduler'], 'epochs': config['epochs']},
            optimizer = self.optim
        )
        self.lr = config['cluster_optim']['lr']
        self.criterion = losses.ClusterLoss(**config['cluster_criterion'])
        
        self.best = 0
        self.best_head_indx = None
    
    def load_ckpt(self):
        
        ckpt = torch.load(os.path.join(self.output_dir, "last.ckpt"))
        self.encoder.load_state_dict(ckpt["enc"])
        self.cluster_head.load_state_dict(ckpt["cluster_head"])
        self.optim.load_state_dict(ckpt["optim"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        return ckpt["epoch"]
    
    def train_one_step(self, data):
        """ Trains model on one batch of data """

        anchor, neighbor = data['anchor'].to(self.device), data['neighbor'].to(self.device)
        a_out = self.cluster_head(self.encoder(anchor))
        n_out = self.cluster_head(self.encoder(neighbor))
        
        # compute losses for each head
        t_loss, c_loss, e_loss = [], [], []
        for a, n in zip(a_out, n_out):
            t, c, e = self.criterion(a, n)
            t_loss.append(t)
            c_loss.append(c)
            e_loss.append(e)

        loss = torch.sum(torch.stack(t_loss, dim=0))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        metrics = {}
        for i in range(len(t_loss)):
            metrics[f'h{i} total loss'] = t_loss[i].item()
            metrics[f'h{i} consistency loss'] = c_loss[i].item()
            metrics[f'h{i} entropy loss'] = e_loss[i].item()

        return metrics

    def save_ckpt(self, epoch):
        ckpt = {
            "enc": self.encoder.state_dict(),
            "cluster_head": self.cluster_head.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": epoch
        }
        if self.lr_scheduler:
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, "last.ckpt"))
    
    def save_model(self, fname):
        W = torch.nn.ModuleList(self.cluster_head.W[self.best_head_indx: self.best_head_indx+1])
        model_save = {
            "enc": self.encoder.state_dict(),
            "cluster_head": W.state_dict()
        }
        torch.save(model_save, os.path.join(self.output_dir, fname))

    @torch.no_grad()
    def validate(self, val_loader):  
        """ Computes metrics to assess quality of clustering """
        
        loss_cntr = common.AverageMeter()
        pred, gt = [], []

        for indx, data in enumerate(val_loader):
            anchor, neighbor = data['anchor'].to(self.device), data['neighbor'].to(self.device)
            
            with torch.no_grad():
                a_out = self.cluster_head(self.encoder(anchor))
                n_out = self.cluster_head(self.encoder(neighbor))

            pred.extend(np.concatenate([o.argmax(dim=1).unsqueeze(1).detach().cpu().numpy() for o in a_out], axis=1))
            gt.extend(data['target'].numpy())

            # Compute all losses and collect
            for i, (a, n) in enumerate(zip(a_out, n_out)):
                t, c, e = self.criterion(a, n)
                loss_cntr.add(
                    {
                        f"h{i} total loss": t.item(),
                        f'h{i} consistency loss': c.item(),
                        f'h{i} entropy loss': e.item()
                    }
                )
            common.progress_bar(progress=indx/len(val_loader))

        loss = loss_cntr.return_metrics()
        total_loss = {key: value for key, value in loss.items() if "total loss" in key}
        self.best_head_indx = np.argmin(list(total_loss.values()))
        
        # Hungarian matching accuracy for best head
        pred = np.array(pred)[:, self.best_head_indx]
        gt = np.array(gt)
        cls_map = eval_utils.hungarian_match(pred, gt, len(np.unique(pred)), len(np.unique(gt)))
        
        remapped_pred = np.zeros(len(pred))
        for pred_c, target_c in cls_map:
            remapped_pred[pred == int(pred_c)] = int(target_c)

        acc = {}
        for i in np.unique(remapped_pred):
            indx = remapped_pred == i
            acc[f"cls {i} acc"] = (remapped_pred[indx] == gt[indx]).sum()/len(remapped_pred[indx])
        acc["acc"] = np.mean(list(acc.values()))
        
        return {**loss, **acc}

# =============================================================================================
# Self labelling model for SCAN
# =============================================================================================

class Selflabel:

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\n{}[INFO] Device found: {}{}".format(pallete['yellow'], torch.cuda.get_device_name(0), pallete['end']))

        # Models 
        self.encoder = models.Encoder(**config['encoder'])
        setup_utils.print_network(self.encoder, 'Encoder')
        self.head = models.ClusteringHead(**config['head'])
        setup_utils.print_network(self.head, 'Clustering Head')

        # Load best model
        try:
            data_name, enc_name = self.config['dataset']['name'], self.config['encoder']['name']
            best_encoder = torch.load(os.path.join(self.output_dir, 'scan/{}/{}_best_encoder.ckpt'.format(
                data_name, enc_name
            )))
            best_head = torch.load(os.path.join(self.output_dir, 'scan/{}/{}_best_head.ckpt'.format(
                data_name, enc_name
            )))
            self.encoder.load_state_dict(best_encoder)
            self.head.load_state_dict(best_head)
        except:
            print("\n{}[WARN] Could not load clustering checkpoint! Starting from random initialization!{}".format(
                pallete['red'], pallete['end']
            ))

        # Optimizer, scheduler and loss function
        self.optim = setup_utils.get_optimizer(
            config = self.config['optimizer'],
            params = list(self.encoder.parameters()) + list(self.head.parameters())
        )
        self.lr_scheduler = setup_utils.get_scheduler(
            config = self.config['scheduler'],
            optimizer = self.optim
        )
        self.criterion = losses.SelflabelLoss(**self.config['criterion'])
        self.lr = self.config['optimizer']['lr']


    def train_one_step(self, data):
        """ Trains model on one batch of data """

        anchor, anchor_aug = data['anchor'].to(self.device), data['anchor_aug'].to(self.device)

        # NO grad on anchor
        with torch.no_grad():
            anchor_logits = self.head(self.encoder(anchor))[0]
        aug_logits = self.head(self.encoder(anchor_aug))[0]

        loss = self.criterion(anchor_logits, aug_logits)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'loss': loss.item()}


    def validate(self, epoch, val_loader):
        """ Assesses predicition quality """

        pred_labels, target_labels = [], []
        pbar = tqdm(total=len(val_loader), desc='Val epoch {:4d}'.format(epoch))

        for idx, data in enumerate(val_loader):
            img = data['img'].to(self.device)
            with torch.no_grad():
                out = self.head(self.encoder(img))[0]
            
            pred_labels.extend(img.argmax(dim=1).cpu().detach().numpy())
            target_labels.extend(data['target'].numpy())
            pbar.update(1)
        pbar.close()

        pred_labels, target_labels = np.array(pred_labels), np.array(target_labels)
        match = hungarian_match(pred_labels, target_labels, len(np.unique(pred_labels)), len(np.unique(target_labels)))
        print("\nHungarian match: {}".format(match))

        remapped_preds = np.zeros(len(pred_labels))
        for pred_i, target_i in match:
            remapped_preds[pred_labels == int(pred_i)] = int(target_i)

        cls_acc = {}
        for i in np.unique(remapped_preds):
            indx = remapped_preds == i
            cls_acc[int(i)] = (remapped_preds[indx] == target_labels[indx]).sum()/len(remapped_preds[indx])
        
        # Print relevant stuff 
        print("\nValidation epoch {}".format(epoch))
        for k, v in loss_dict.items():
            print("\t{}: {:.4f}".format(k, v))
        
        print("\nHungarian accuracy")
        for k, v in cls_acc.items():
            print("\tClass {} - {:.4f}".format(k, v))

        # Log class accuracy in wandb as histogram
        data = [[k, v] for k, v in cls_acc.items()]
        table = wandb.Table(data=data, columns=['cluster', 'accuracy'])
        wandb.log({'cluster_accuracy_chart': wandb.plot.bar(table, 'cluster', 'accuracy', title='Cluster-wise accuracy')})

        return {"acc": np.mean(list(cls_acc.values()))}


    def save(self, epoch):
        data_name = self.config['dataset']['name']
        enc_name = self.config['encoder']['name']
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'head': self.head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.lr_scheduler.statie_dict() if self.lr_scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, 'selflabel/{}/{}_epoch_{}'.format(
            data_name, enc_name, epoch
        )))


