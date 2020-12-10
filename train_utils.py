
"""
Training functions and classes.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.optimize import linear_sum_assignment
import setup_utils 
import data_utils
import models 
import losses
from tqdm import tqdm 
from termcolor import cprint
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


def hungarian_match(preds, targets, preds_k, targets_k):
    """ 
    Lowest error matching between predicted and target classes.
    """
    # Based on implementation from IIC
    num_samples = targets.shape[0]

    assert preds_k == targets_k, 'Different number of unique classes in preds and targets'
    k = preds_k 
    num_correct = np.zeros((k, k))

    for c1 in range(k):
        for c2 in range(k):
            votes = int(((preds == c1) * (targets == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # Return as list of tuples (out_c, gt_c)
    res = [(out_c, gt_c) for out_c, gt_c in match]
    return res


# =================================================================================================
# Simple architecture for Contrastive Learning of Visual Representations (SimCLR)
# =================================================================================================

class SimCLR:

    def __init__(self, config, output_dir):

        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cprint("\n[INFO] Device found: {}".format(torch.cuda.get_device_name(0)), 'yellow')

        # Models 
        self.encoder = models.Encoder(**config['simclr_encoder']).to(self.device)
        setup_utils.print_network(self.encoder, 'Encoder')
        self.proj_head = models.ProjectionHead(**config['simclr_projection_head']).to(self.device)
        setup_utils.print_network(self.proj_head, 'Projection Head')
        self.proj_head.apply(init_weights)

        # Optimizer, scheduler and criterion
        self.optim = setup_utils.get_optimizer(
            config=config['simclr_optim'], 
            params=list(self.encoder.parameters)+list(self.proj_head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = setup_utils.get_scheduler(
            config={**config['simclr_scheduler'], 'epochs': config['epochs']},
            optimizer=self.optim
        )
        self.criterion = losses.SimclrLoss(config['batch_size'], **config['simclr_criterion'])


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
        index = faiss.IndexFlatIP(z.shape[1])
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

        data_name = self.config['dataset']['name']
        enc_name = self.config['encoder']['name']

        clf_head = models.ClassificationHead(in_dim=self.encoder.backbone_dim, n_classes=self.config['dataset']['n_classes'])
        clf_optim = setup_utils.get_optimizer(**self.config['clf_optim'], params=clf_head.parameters())
        clf_scheduler, _ = setup_utils.get_scheduler(**self.config['clf_scheduler'], optimizer=clf_optim)
        done_epochs = 0

        # If a checkpoint exists, load it
        ckpt_path = os.path.join(self.output_dir, 'simclr/{}/{}_linear_eval.ckpt'.format(data_name, enc_name))
        if os.path.exists(ckpt_path):
            cprint("\n[INFO] Resuming training from {}_linear_eval.ckpt".format(enc_name), 'yellow')
            ckpt = torch.load(ckpt_path)
            done_epochs = ckpt['epoch']
            clf_head.load_state_dict(ckpt['clf_head'])
            clf_optim.load_state_dict(ckpt['clf_optim'])
            clf_scheduler.load_state_dict(ckpt['clf_scheduler'])

        # Train classifier with frozen encoder
        # Feature vectors are extracted from encoder only, not the projection head 
        best_acc = 0
        train_loss, train_acc = [], []
        val_loss, val_acc = [], []

        for epoch in range(self.config['linear_eval_epochs'] - done_epochs):         
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

            # Save model if better
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(clf_head.state_dict(), \
                    os.path.join(self.output_dir, 'simclr/{}/{}_best_clf_head.ckpt'.format(data_name, enc_name)))

            # Save model
            current_state = {
                'epoch': epoch,
                'clf_head': clf_head.state_dict(),
                'clf_optim': clf_optim.state_dict(),
                'clf_scheduler': clf_scheduler.state_dict()
            }
            torch.save(current_state, os.path.join(self.output_dir, 'simclr/{}/{}_linear_eval.ckpt'.format(data_name, enc_name)))
            
            # Update pbar
            pbar.set_description('[Epoch] {:3d} [Train loss] {:.4f} [Train acc] {:.4f} [Val loss] {:.4f} [Val acc] {:.4f}'.format(
                epoch+1, train_loss, train_acc, val_loss, val_acc
            ))
            pbar.close()

        
    def find_neighbors(self, data_loader, img_key, topk=20):
        """ Mine neighbors with trained encoder and projection head """

        # Generate vectors
        pbar = tqdm(total=len(data_loader), desc='Building feature vectors')
        fvecs = []
        for batch in data_loader:
            img = batch[img_key].to(self.device)
            with torch.no_grad():
                z = self.proj_head(self.encoder(img))
            fvecs.extend(z.cpu().detach().numpy())
            pbar.update(1)
        pbar.close()
        fvecs = np.array(fvecs)

        # Mine neighbors and save
        index = faiss.IndexFlatIP(dim=fvecs.shape[1])
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(fvecs)
        _, indices = index.search(fvecs, topk+1)
        np.save(os.path.join(self.output_dir, 'simclr/{}/{}_neighbors.npy'), indices)


    def save(self, epoch):
        """ Save the model, optimizer and scheduler """

        data_name = self.config['dataset']['name']
        enc_name = self.config['simclr_encoder']['name']
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'proj_head': self.proj_head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, 'simclr/{}/{}_epoch_{}'.format(
            data_name, enc_name, epoch
        )))


# =============================================================================================
# Clustering model for SCAN
# =============================================================================================

class SCAN:

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\n[INFO] Found device: {}\n".format(torch.cuda.get_device_name(0)))

        # Models
        self.encoder = models.Encoder(**self.config['scan_encoder']).to(self.device)
        setup_utils.print_network(self.encoder, name='Encoder')
        self.cluster_head = models.ClusteringHead(**self.config['scan_clustering_head']).to(self.device)
        setup_utils.print_network(self.cluster_head, name='Clustering Head')
        
        # Load SimCLR checkpoint into encoder   
        try:
            best_encoder = torch.load(os.path.join(self.output_dir, 'simclr/{}/{}_best_encoder.ckpt'))
            self.encoder.load_state_dict(best_encoder)
        except:
            cprint("\n[WARN] Could not load SimCLR encoder! Starting with random initialization!", 'red')

        # Optimizer, scheduler and loss function
        self.optim = setup_utils.get_optimizer(
            config = self.config['scan_optimizer'], 
            params = list(self.encoder.parameters())+list(self.cluster_head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = setup_utils.get_scheduler(
            config = self.config['scan_scheduler'],
            optimizer = self.optim
        )
        self.criterion = losses.ScanLoss(**self.config['scan_criterion'])

    
    def train_one_step(self, data):
        """ Trains model on one batch of data """

        anchor_img, neighbor_img = data['anchor_img'].to(self.device), data['neighbor_img'].to(self.device)
        anchor_out = self.cluster_head(self.encoder(anchor_img))
        neighbor_out = self.cluster_head(self.encoder(neighbor_img))

        total_losses, consis_losses, entr_losses = [], [], []
        for anchor, neighbor in zip(anchor_out, neighbor_out):
            total_loss, consis_loss, entr_loss = self.criterion(anchor, neighbor)
            total_losses.append(total_loss)
            consis_losses.append(consis_loss)
            entr_losses.append(entr_loss)

        loss = torch.sum(torch.stack(total_losses, dim=0))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        metrics = {}
        for i in range(len(total_losses)):
            metrics[f'h{i}_total_loss'] = total_losses[i].item()
            metrics[f'h{i}_consis_loss'] = consis_losses[i].item()
            metrics[f'h{i}_entr_loss'] = entr_losses[i].item()

        return metrics


    def validate(self, epoch, val_loader):  
        """ Computes metrics to assess quality of clustering """

        loss_dict = {}
        total_loss_counter = []
        pred_labels, target_labels = [], []
        pbar = tqdm(total=len(val_loader), desc="Val epoch {}".format(epoch))

        for idx, batch in enumerate(val_loader):
            anchor_img, neighbor_img = data['anchor_img'].to(self.device), data['neighbor_img'].to(self.device)
            
            # Freeze model and generate predictions
            with torch.no_grad():
                anchor_out = self.cluster_head(self.encoder(anchor_img))
                neighbor_out = self.cluster_head(self.encoder(neighbor_img))

            pred_labels.extend(np.concatenate([o.argmax(dim=1).unsqueeze(1).detach().cpu().numpy() for o in anchor_out], axis=1))
            target_labels.extend(data['target'].numpy())

            # Compute all losses and collect
            total_loss, consistency_loss, entropy_loss = [], [], []
            for anchor, neighbor in zip(anchor_out, neighbor_out):
                tl, cl, el = self.criterion(anchor, neighbor)
                total_loss.append(tl)
                consistency_loss.append(cl)
                entropy_loss.append(el)

            # Collect loss for each head
            for i in range(len(total_loss)):
                if idx == 0:
                    loss_dict[f'h{i}_total_loss'] = [total_loss[i]]
                    loss_dict[f'h{i}_consis_loss'] = [consistency_loss[i]]
                    loss_dict[f'h{i}_entr_loss'] = [entropy_loss[i]]
                else:
                    loss_dict[f'h{i}_total_loss'].append(total_loss[i])
                    loss_dict[f'h{i}_consis_loss'].append(consistency_loss[i])
                    loss_dict[f'h{i}_entr_loss'].append(entropy_loss[i])

            total_loss_counter.append(np.array(total_loss).reshape(1, -1))
            pbar.update(1)
        pbar.close()

        loss_dict = {k: np.mean(v) for k, v in loss_dict.items()}
        total_loss_counter = np.mean(np.concatenate(total_loss_counter, axis=0), axis=0)
        best_head_index = np.argmin(total_loss_counter)

        # Hungarian matching accuracy for best head
        pred_labels = np.array(pred_labels)[:, best_head_index]
        target_labels = np.array(target_labels)
        match = hungarian_match(pred_labels, target_labels, len(np.unique(pred_labels)), len(np.unique(target_labels)))
        print("\nHungarian match: {}".format(match))
        
        remapped_preds = np.zeros(len(pred_labels))
        for pred_i, target_i in match:
            remapped_preds[pred_labels = int(pred_i)] == int(target_i)

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

        return {**loss_dict, "acc": np.mean(list(cls_acc.values()))}


    def save(self, epoch):
        data_name = self.config['dataset']['name']
        enc_name = self.config['scan_encoder']['name']
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'cluster_head': self.cluster_head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, 'scan/{}/{}_epoch_{}'.format(
            data_name, enc_name, epoch
        )))


# =============================================================================================
# Self labelling model for SCAN
# =============================================================================================

class Selflabel:

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cprint("\n[INFO] Device found: {}".format(torch.cuda.get_device_name(0)), 'yellow')

        # Models 
        self.encoder = models.Encoder(**config['selflabel_encoder'])
        setup_utils.print_network(self.encoder, 'Encoder')
        self.cluster_head = models.ClusteringHead(**config['selflabel_cluster_head'])
        setup_utils.print_network(self.cluster_head, 'Clustering Head')

        # Load best model
        try:
            data_name, enc_name = self.config['dataset']['name'], self.config['selflabel_encoder']['name']
            best_encoder = torch.load(os.path.join(self.output_dir, 'scan/{}/{}_best_encoder.ckpt'.format(
                data_name, enc_name
            )))
            best_cluster_head = torch.load(os.path.join(self.output_dir, 'scan/{}/{}_best_cluster_head.ckpt'.format(
                data_name, enc_name
            )))
            self.encoder.load_state_dict(best_encoder)
            self.cluster_head.load_state_dict(best_cluster_head)
        except:
            cprint("\n[WARN] Could not load clustering checkpoint! Starting from random initialization!", 'red')

        # Optimizer, scheduler and loss function
        self.optim = setup_utils.get_optimizer(
            config = self.config['selflabel_optim'],
            params = list(self.encoder.parameters()) + list(self.cluster_head.parameters())
        )
        self.lr_scheduler = setup_utils.get_scheduler(
            config = self.config['selflabel_scheduler'],
            optimizer = self.optim
        )
