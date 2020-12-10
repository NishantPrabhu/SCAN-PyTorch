
"""
Training functions and classes.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import train_utils 
import data_utils
import models 
import losses
from tqdm import tqdm 
import numpy as np 
import wandb
import faiss


def init_weights(m):
    """
    Weight initializations for a layer.
    """
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class SimCLR:

    def __init__(self, config):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\n[INFO] Device found: {}".format(torch.cuda.get_device_name(0)))

        # Models 
        self.encoder = models.Encoder(**config['encoder']).to(self.device)
        train_utils.print_network(self.encoder, 'Encoder')
        self.proj_head = models.ProjectionHead(**config['projection_head']).to(self.device)
        train_utils.print_network(self.proj_head, 'Projection Head')
        self.proj_head.apply(init_weights)

        # Optimizer, scheduler and criterion
        self.optim = train_utils.get_optimizer(
            config=config['simclr_optim'], 
            params=list(self.encoder.parameters)+list(self.proj_head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config={**config['simclr_scheduler'], 'epochs': config['epochs']},
            optimizer=self.optim
        )
        self.criterion = losses.SimclrLoss(config['batch_size'], **config['criterion'])


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


    @staticmethod
    def calculate_accuracy(z, targets, topk=20):
        """ Computes accuracy of mined neighbors """

        # Mine neighbors
        index = faiss.IndexFlatIP(z.size(1))
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(z)
        _, indices = index.search(z, topk+1)

        # Compute accuracy 
        anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
        neighbor_targets = np.take(targets, indices[:, 1:], axis=0)
        accuracy = np.mean(anchor_targets == neighbor_targets)
        return accuracy


    def validate(self, epoch, val_loader):
        """ Niehgbor mining accuracies wrapper function """

        pbar = tqdm(total=len(val_loader), desc='Val epoch {}'.format(epoch))
        fvecs, labels = [], []

        for data in val_loader:
            img, target = data['img'].to(self.device), data['target']
            with torch.no_grad():
                z = self.proj_head(self.encoder(img))
            z = F.normalize(z, p=2, dim=-1)
            fvecs.append(z.detach().cpu().numpy())
            labels.append(target.numpy())
            pbar.update(1)

        fvecs, labels = np.array(fvecs), np.array(labels)
        acc = SimCLR.calculate_accuracy(fvecs, labels, topk=20)

        pbar.set_description('[Val epoch] {} - [Accuracy] {:.4f}'.format(epoch, acc))
        pbar.close()
        return {'val_acc': acc}


    def linear_eval(self, train_loader, val_loader):
        """ Evaluation of SimCLR vectors with linear classifier """

        clf_head = models.ClassificationHead(in_dim=self.encoder.backbone_dim, n_classes=self.config['dataset']['n_classes'])
        clf_optim = train_utils.get_optimizer(**self.config['clf_optim'], params=clf_head.parameters())
        clf_scheduler, _ = train_utils.get_scheduler(**self.config['clf_scheduler'], optimizer=clf_optim)

        # Train classifier with frozen encoder
        # Feature vectors are extracted from encoder only, not the projection head 
        best_acc = 0
        train_loss, train_acc = [], []
        val_loss, val_acc = [], []

        for epoch in range(self.config['linear_eval_epochs']):         
            pbar = tqdm(total=len(train_loader)+len(val_loader), decs='Epoch {}'.format(epoch+1))

            # Training 
            clf_head.train()
            for batch in train_loader:
                img, target = batch['img'].to(self.device), batch['target']
                with torch.no_grad():    
                    h = self.encoder(img)
                
                # Loss and update
                pred = clf_head(h)
                loss = F.nll_loss(pred, target, reduction='mean')
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

                # Accuracy
                correct = pred.eq(target.view_as(pred)).sum().item()
                train_acc.append(correct)
                train_loss.append(loss.item())
                pbar.update(1)
            train_loss, train_acc = np.mean(train_loss), np.mean(train_acc)

            # Validation
            clf_head.eval()
            for batch in val_loader:
                img, target = batch['img'].to(self.device), batch['target']
                with torch.no_grad():
                    h = self.encoder(img)
                # Loss
                pred = clf_head(h)
                loss = F.nll_loss(pred, target, reduction='mean')
                # Accuracy
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc.append(correct)
                val_loss.append(loss.item())
                pbar.update(1)
            val_loss, val_acc = np.mean(val_loss), np.mean(val_acc)

            pbar.set_description('[Epoch] {:3d} [Train loss] {:.4f} [Train acc] {:.4f} [Val loss] {:.4f} [Val acc] {:.4f}'.format(
                epoch+1, train_loss, train_acc, val_loss, val_acc
            ))
            pbar.close()

        


                







    


