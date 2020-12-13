
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

# def hungarian_match(preds, targets, preds_k, targets_k):
#     """ 
#     Lowest error matching between predicted and target classes.
#     """
#     # Based on implementation from IIC
#     num_samples = targets.shape[0]

#     assert preds_k == targets_k, 'Different number of unique classes in preds and targets'
#     k = preds_k 
#     num_correct = np.zeros((k, k))

#     for c1 in range(k):
#         for c2 in range(k):
#             votes = int(((preds == c1) * (targets == c2)).sum())
#             num_correct[c1, c2] = votes

#     match = linear_sum_assignment(num_samples - num_correct)
#     match = np.array(list(zip(*match)))

#     # Return as list of tuples (out_c, gt_c)
#     res = [(out_c, gt_c) for out_c, gt_c in match]
#     return res


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
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch
        }
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
        neighbour_indices = eval_utils.find_neighbours(fvecs, topk=self.config["n_neighbours"])
        acc = eval_utils.compute_neighbour_acc(gt, neighbour_indices, topk=self.config["n_neighbours"])
        
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

    def build_neighbours(self, data_loader, fname):
        fvecs, _ = self.return_fvecs(data_loader)
        neighbour_indices = eval_utils.find_neighbours(fvecs, topk=self.config["n_neighbours"])
        np.save(os.path.join(self.output_dir, fname), neighbour_indices)

# =============================================================================================
# Clustering model for SCAN
# =============================================================================================

class SCAN:

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\n{}[INFO] Found device: {}\n{}".format(pallete['yellow'], torch.cuda.get_device_name(0), pallete['end']))

        # Models
        self.encoder = models.Encoder(**self.config['encoder']).to(self.device)
        setup_utils.print_network(self.encoder, name='Encoder')
        self.head = models.ClusteringHead(**self.config['head']).to(self.device)
        setup_utils.print_network(self.head, name='Clustering Head')

        # Loss meters to keep track of loss for each head
        self.loss_meters = [data_utils.Scalar() for _ in range(self.config['head']['heads'])]
        
        # Load SimCLR checkpoint into encoder   
        try:
            best_encoder = torch.load(os.path.join(self.output_dir, 'simclr/{}/{}_best_encoder.ckpt'))
            self.encoder.load_state_dict(best_encoder)
        except:
            print("\n{}[WARN] Could not load SimCLR encoder! Starting with random initialization!{}".format(pallete['red'], pallete['end']))

        # Optimizer, scheduler and loss function
        self.optim = setup_utils.get_optimizer(
            config = self.config['optimizer'], 
            params = list(self.encoder.parameters())+list(self.head.parameters())
        )
        self.lr_scheduler, self.warmup_epochs = setup_utils.get_scheduler(
            config = self.config['scheduler'],
            optimizer = self.optim
        )
        self.criterion = losses.ScanLoss(**self.config['criterion'])
        self.lr = self.config['optimizer']['lr']

    
    def train_one_step(self, data):
        """ Trains model on one batch of data """

        anchor_img, neighbor_img = data['anchor_img'].to(self.device), data['neighbor_img'].to(self.device)
        anchor_out = self.head(self.encoder(anchor_img))
        neighbor_out = self.head(self.encoder(neighbor_img))

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
        pbar = tqdm(total=len(val_loader), desc="Val epoch {:4d}".format(epoch))

        for idx, batch in enumerate(val_loader):
            anchor_img, neighbor_img = data['anchor_img'].to(self.device), data['neighbor_img'].to(self.device)
            
            # Freeze model and generate predictions
            with torch.no_grad():
                anchor_out = self.head(self.encoder(anchor_img))
                neighbor_out = self.head(self.encoder(neighbor_img))

            pred_labels.extend(np.concatenate([o.argmax(dim=1).unsqueeze(1).detach().cpu().numpy() for o in anchor_out], axis=1))
            target_labels.extend(data['target'].numpy())

            # Compute all losses and collect
            total_loss, consistency_loss, entropy_loss = [], [], []
            for i, (anchor, neighbor) in enumerate(zip(anchor_out, neighbor_out)):
                tl, cl, el = self.criterion(anchor, neighbor)
                self.loss_meters[i].update(tl)
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

        return {**loss_dict, "acc": np.mean(list(cls_acc.values()))}


    def save(self, epoch):
        """ Save the model """

        data_name = self.config['dataset']['name']
        enc_name = self.config['encoder']['name']
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'head': self.head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, 'scan/{}/{}_epoch_{}'.format(
            data_name, enc_name, epoch
        )))


    def find_best_head(self):
        """ Returns the head with lowest average total loss """

        return np.argmin([m.mean for m in self.loss_meters])


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


