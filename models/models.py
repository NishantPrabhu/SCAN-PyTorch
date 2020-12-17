
'''
Training functions and classes.

Authors: Mukund Varma T, Nishant Prabhu
'''

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
        self.proj_head = networks.ProjectionHead(in_dim=self.encoder.backbone_dim, **config['proj_head']).to(self.device)
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
        
        ckpt = torch.load(os.path.join(self.output_dir, 'last.ckpt'))
        self.encoder.load_state_dict(ckpt['enc'])
        self.proj_head.load_state_dict(ckpt['proj_head'])
        self.optim.load_state_dict(ckpt['optim'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        return ckpt['epoch']

    def train_one_step(self, data):
        ''' Trains model on one batch of data '''

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
            'enc': self.encoder.state_dict(),
            'proj_head': self.proj_head.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': epoch
        }
        if self.lr_scheduler:
            ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, 'last.ckpt'))
    
    def save_model(self, fname):
        model_save = {
            'enc': self.encoder.state_dict()
        }
        torch.save(model_save, os.path.join(self.output_dir, fname))
    
    @torch.no_grad()
    def return_fvecs(self, data_loader):
        ''' Build feature vectors '''

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
        neighbour_indices = eval_utils.find_neighbors(fvecs, topk=self.config['n_neighbors'])
        acc = eval_utils.compute_neighbour_acc(gt, neighbour_indices, topk=self.config['n_neighbors'])
        
        if acc >= self.best:
            self.save_model('best.pth')
            self.best = acc
        
        return {'neighbour acc': acc}

    def linear_eval(self, train_loader, val_loader):
        ''' Evaluation of SimCLR vectors with linear classifier '''

        clf_head = networks.ClassificationHead(
            in_dim=self.encoder.backbone_dim, 
            n_classes=DATASET_HELPER[self.config['dataset']]['classes']
        ).to(self.device)
        clf_optim = train_utils.get_optimizer(
            config={**self.config['clf_optim']}, 
            params=clf_head.parameters()
        )
        clf_scheduler, _ = train_utils.get_scheduler(
            config={**self.config['clf_lr_scheduler'], 'epochs': self.config['linear_eval_epochs']}, 
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
                train_metrics.add({'train acc': correct/len(pred), 'train loss': loss.item()})

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
                val_metrics.add({'val acc': correct/len(pred), 'val loss': loss.item()})

            train_log = train_metrics.return_msg()
            val_log = val_metrics.return_msg()
            
            common.progress_bar(progress=epoch/self.config['linear_eval_epochs'], status=train_log+val_log)
            # Save model if better
            if  val_metrics.return_metrics()['val acc'] > best:
                torch.save(clf_head.state_dict(), os.path.join(self.output_dir, 'best_clf_head.pth'))
                best = val_metrics.return_metrics()['val acc']
            
            if clf_scheduler is not None:
                clf_scheduler.step()

        return {'linear eval acc': best}

    def build_neighbors(self, data_loader, fname):
        fvecs, _ = self.return_fvecs(data_loader)
        neighbour_indices = eval_utils.find_neighbors(fvecs, topk=self.config['n_neighbors'])
        np.save(os.path.join(self.output_dir, fname), neighbour_indices)


# =============================================================================================
# RotNet: Unsupervised representation learning by predicting image rotations
# =============================================================================================

class RotNet:

    def __init__(self, config, device, output_dir):

        self.config = config 
        self.device = device 
        self.output_dir = output_dir

        # Models
        self.encoder = networks.Encoder(**config['encoder']).to(self.device)
        common.print_network(self.encoder, 'Encoder')
        self.cls_head = networks.RotnetClassifier(in_dim=self.encoder.backbone_dim, **config['rotnet']).to(self.device)
        common.print_network(self.cls_head, 'Classification Head')

        # Optimizer, scheduler and loss function
        self.optim = train_utils.get_optimizer(
            config = config['rotnet_optim'], 
            params = list(self.encoder.parameters())+list(self.cls_head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**config['rotnet_lr_scheduler'], 'epochs': config['epochs']},
            optimizer = self.optim
        )
        self.lr = config['rotnet_optim']['lr']
        self.criterion = losses.SupervisedLoss()

        # Accuracy monitor for model checkpointing
        self.best = 0


    def load_ckpt(self):
        ckpt = torch.load(os.path.join(self.output_dir, 'last.ckpt'))
        self.encoder.load_state_dict(ckpt['enc'])
        self.cls_head.load_state_dict(ckpt['head'])
        self.optim.load_state_dict(ckpt['optim'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        return ckpt['epoch']


    def save_ckpt(self, epoch):
        ckpt = {
            'enc': self.encoder.state_dict(),
            'head': self.cls_head.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': epoch
        }
        if self.lr_scheduler is not None:
            ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, 'last.ckpt'))
    
    def save_model(self, fname):
        model_save = {
            'enc': self.encoder.state_dict()
        }
        torch.save(model_save, os.path.join(self.output_dir, fname))

    def train_one_step(self, data):
        ''' Trains model on one batch of data '''

        img, labels = data['img'].to(self.device), data['target'].to(self.device)
        out = self.cls_head(self.encoder(img))
        loss = self.criterion(out, labels)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'loss': loss.item()}

    @torch.no_grad()
    def return_fvecs(self, data_loader):
        ''' Build feature vectors '''

        fvecs, gt = [], []

        for indx, data in enumerate(data_loader):
            img, target = data['img'].to(self.device), data['target']
            z = self.encoder(img)
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
        neighbour_indices = eval_utils.find_neighbors(fvecs, topk=self.config['n_neighbors'])
        acc = eval_utils.compute_neighbour_acc(gt, neighbour_indices, topk=self.config['n_neighbors'])
        
        if acc >= self.best:
            self.save_model('best.pth')
            self.best = acc
        
        return {'neighbour acc': acc}

    def linear_eval(self, train_loader, val_loader):
        ''' Evaluation of SimCLR vectors with linear classifier '''

        clf_head = networks.ClassificationHead(
            in_dim=self.encoder.backbone_dim, 
            n_classes=DATASET_HELPER[self.config['dataset']]['classes']
        ).to(self.device)
        clf_optim = train_utils.get_optimizer(
            config={**self.config['clf_optim']}, 
            params=clf_head.parameters()
        )
        clf_scheduler, _ = train_utils.get_scheduler(
            config={**self.config['clf_lr_scheduler'], 'epochs': self.config['linear_eval_epochs']}, 
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
                train_metrics.add({'train acc': correct/len(pred), 'train loss': loss.item()})

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
                val_metrics.add({'val acc': correct/len(pred), 'val loss': loss.item()})

            train_log = train_metrics.return_msg()
            val_log = val_metrics.return_msg()
            
            common.progress_bar(progress=epoch/self.config['linear_eval_epochs'], status=train_log+val_log)
            # Save model if better
            if  val_metrics.return_metrics()['val acc'] > best:
                torch.save(clf_head.state_dict(), os.path.join(self.output_dir, 'best_clf_head.pth'))
                best = val_metrics.return_metrics()['val acc']
            
            if clf_scheduler is not None:
                clf_scheduler.step()
            break

        return {'linear eval acc': best}

    def build_neighbors(self, data_loader, fname):
        fvecs, _ = self.return_fvecs(data_loader)
        neighbour_indices = eval_utils.find_neighbors(fvecs, topk=self.config['n_neighbors'])
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
        save = torch.load(self.config['simclr_save'])
        self.encoder = networks.Encoder(**config['encoder']).to(self.device)
        self.encoder.load_state_dict(save['enc'])
        common.print_network(self.encoder, 'Encoder')
        self.cluster_head = networks.ClusteringHead(in_dim=self.encoder.backbone_dim, **config['cluster_head']).to(self.device)
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
        
        ckpt = torch.load(os.path.join(self.output_dir, 'last.ckpt'))
        self.encoder.load_state_dict(ckpt['enc'])
        self.cluster_head.load_state_dict(ckpt['cluster_head'])
        self.optim.load_state_dict(ckpt['optim'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        return ckpt['epoch']
    
    def train_one_step(self, data):
        ''' Trains model on one batch of data '''

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
            'enc': self.encoder.state_dict(),
            'cluster_head': self.cluster_head.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': epoch
        }
        if self.lr_scheduler:
            ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, 'last.ckpt'))
    
    def save_model(self, fname):
        w = list(self.cluster_head.W[self.best_head_indx].state_dict().values())
        w_keys = ['W.0.weight', 'W.0.bias']
        model_save = {
            'enc': self.encoder.state_dict(),
            'cluster_head': dict(zip(w_keys, w))
        }
        torch.save(model_save, os.path.join(self.output_dir, fname))

    @torch.no_grad()
    def validate(self, val_loader):  
        ''' Computes metrics to assess quality of clustering '''
        
        total_loss = common.AverageMeter()
        probs, gt = [], []

        for indx, data in enumerate(val_loader):
            anchor, neighbor = data['anchor'].to(self.device), data['neighbor'].to(self.device)
            
            with torch.no_grad():
                a_out = self.cluster_head(self.encoder(anchor))
                n_out = self.cluster_head(self.encoder(neighbor))

            probs.append(torch.cat([o.unsqueeze(0).detach().cpu() for o in a_out], dim=0))
            gt.extend(data['target'])

            # Compute all losses and collect
            for i, (a, n) in enumerate(zip(a_out, n_out)):
                t, _, _ = self.criterion(a, n)
                total_loss.add({f'h{i} total loss': t.item()})
            common.progress_bar(progress=indx/len(val_loader))
        common.progress_bar(progress=1)
        
        total_loss = total_loss.return_metrics()
        self.best_head_indx = np.argmin(list(total_loss.values()))
        
        # eval
        probs = torch.cat(probs, dim=1)[self.best_head_indx]
        gt = torch.tensor(gt)
        cluster_score = eval_utils.eval_clusters(probs, gt)
        
        if cluster_score['acc'] >= self.best:
            self.save_model('best.pth')
            self.best = cluster_score['acc']

        return {**cluster_score}

# =============================================================================================
# Self labelling model for SCAN
# =============================================================================================

class SelfLabel:

    def __init__(self, config, device, output_dir):
        
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Models 
        save = torch.load(self.config['cluster_save'])
        self.encoder = networks.Encoder(**config['encoder']).to(self.device)
        self.encoder.load_state_dict(save['enc'])
        common.print_network(self.encoder, 'Encoder')
        self.cluster_head = networks.ClusteringHead(in_dim=self.encoder.backbone_dim, **config['cluster_head']).to(self.device)
        self.cluster_head.load_state_dict(save['cluster_head'])
        common.print_network(self.cluster_head, 'Clustering Head')

        # Optimizer, scheduler and criterion
        self.optim = train_utils.get_optimizer(
            config = config['selflabel_optim'], 
            params = list(self.encoder.parameters())+list(self.cluster_head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**config['selflabel_lr_scheduler'], 'epochs': config['epochs']},
            optimizer = self.optim
        )
        self.lr = config['selflabel_optim']['lr']
        self.criterion = losses.SelflabelLoss(**config['selflabel_criterion'])
        
        self.best = 0
    
    def load_ckpt(self):
        
        ckpt = torch.load(os.path.join(self.output_dir, 'last.ckpt'))
        self.encoder.load_state_dict(ckpt['enc'])
        self.cluster_head.load_state_dict(ckpt['cluster_head'])
        self.optim.load_state_dict(ckpt['optim'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        return ckpt['epoch']

    def train_one_step(self, data):
        ''' Trains model on one batch of data '''

        img, img_aug = data['img'].to(self.device), data['img_aug'].to(self.device)

        # NO grad on anchor
        with torch.no_grad():
            img_out = self.cluster_head(self.encoder(img))[0]
        img_aug_out = self.cluster_head(self.encoder(img_aug))[0]

        loss = self.criterion(img_out, img_aug_out)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'loss': loss.item()}

    def save_ckpt(self, epoch):
        ckpt = {
            'enc': self.encoder.state_dict(),
            'cluster_head': self.cluster_head.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': epoch
        }
        if self.lr_scheduler:
            ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, 'last.ckpt'))
    
    def save_model(self, fname):
        model_save = {
            'enc': self.encoder.state_dict(),
            'cluster_head': self.cluster_head.state_dict()
        }
        torch.save(model_save, os.path.join(self.output_dir, fname))

    @torch.no_grad()
    def validate(self, val_loader):
        ''' Assesses prediction quality '''

        probs, gt = [], []
        for indx, data in enumerate(val_loader):
            img = data['img'].to(self.device)
            with torch.no_grad():
                out = self.cluster_head(self.encoder(img))
            
            probs.append(torch.cat([o.unsqueeze(0).detach().cpu() for o in out], dim=0))
            gt.extend(data['target'])
            common.progress_bar(progress=indx/len(val_loader))
        common.progress_bar(progress=1)
        probs, gt = torch.cat(probs, dim=1)[0], torch.tensor(gt)
        cluster_score = eval_utils.eval_clusters(probs, gt)
        
        if cluster_score['acc'] >= self.best:
            self.save_model('best.pth')
            self.best = cluster_score['acc']

        return {**cluster_score}

