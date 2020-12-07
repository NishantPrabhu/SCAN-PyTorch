
import torch 
import torch.nn as nn
import torch.nn.functional as F
from models import Encoder, ProjectionHead, LinearClassifier
from train_utils import get_lr_scheduler, get_optimizer
from losses import SimCLRLoss
from tqdm import tqdm
from termcolor import cprint
import numpy as np
import faiss 
import pickle
import wandb


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class SimCLR:

    def __init__(self, config):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device found: {}".format(torch.cuda.get_device_name(0)))

        # Models 
        self.encoder = Encoder(**config['encoder']).to(self.device)
        self.proj_head = ProjectionHead(**config['proj_head']).to(self.device)
        self.proj_head.apply(init_weights)

        # Optimizer, scheduler and criterion
        self.optim = get_optimizer(
            config = {**self.config['simclr_optim'], 'max_epoch': self.config['epochs']},
            params = list(self.encoder.parameters())+list(self.proj_head.parameters())
        )
        self.lr_scheduler = get_lr_scheduler(
            config = {**self.config['simclr_scheduler'], 'total_epochs': self.config['epochs']},
            optimizer = self.optim
        )
        self.criterion = SimCLRLoss(config['batch_size'], **config['criterion'])

        # Wandb
        wandb.init(name='simclr-lars', project='simclr-for-scan')


    def train_step(self, data):

        img_i, img_j = data['i'].to(self.device), data['j'].to(self.device)        
        z_i = self.proj_head(self.encoder(img_i))
        z_j = self.proj_head(self.encoder(img_j))

        self.optim.zero_grad()
        loss = self.criterion(z_i, z_j)
        loss.backward()
        self.optim.step()
        return loss.item()


    @staticmethod
    def mining_accuracy(z, targets, k=20):

        # Find closest neighbors
        index = faiss.IndexFlatIP(128)
        index.add(z)
        _, indices = index.search(z, k+1)
        
        # Compute accuracy
        anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
        neighbor_targets = np.take(targets, indices[:, 1:], axis=0)
        accuracy = (anchor_targets == neighbor_targets).mean()
        return accuracy 


    def neighbor_mining_accuracy(self, val_loader):

        fvecs, labels = [], []

        for batch in tqdm(val_loader):
            img, target = batch['img'].to(self.device), batch['target']
            with torch.no_grad():
                z = self.proj_head(self.encoder(img))
            z = F.normalize(z, p=2, dim=-1)
            fvecs.extend(z.detach().cpu().numpy())
            labels.extend(target.numpy())

        fvecs, labels = np.array(fvecs), np.array(labels)
        val_acc = SimCLR.mining_accuracy(fvecs, labels)
        return val_acc


    def linear_evaluation(self, train_loader, val_loader):
        
        clf_head = LinearClassifier(self.encoder.backbone_dim, out_dim=self.config['dataset']['num_classes']).to(self.device)
        optim = get_optimizer(
            config = {**self.config['clf_optim']}, 
            params = clf_head.parameters()
        )
        scheduler = get_lr_scheduler(
            config = {**self.config['clf_scheduler'], 'total_epochs': self.config['epochs']},
            optimizer = optim
        )

        best_acc = 0

        for epoch in range(self.config['linear_eval_epochs']):
            train_correct = 0
            val_correct = 0

            for batch in train_loader:
                img, targets = batch['img'].to(self.device), batch['target'].to(self.device)
                with torch.no_grad():
                    h = self.encoder(img)
                clf_out = clf_head(h)
                loss = F.nll_loss(clf_out, targets, reduction='mean')
                optim.zero_grad()
                loss.backward()
                optim.step()
                preds = clf_out.argmax(dim=1).detach().cpu()
                train_correct += preds.eq(targets.cpu().view_as(preds)).sum().item()
            
            train_acc = train_correct / len(train_loader.dataset)
            scheduler.step(epoch)
            
            for batch in val_loader:
                img, targets = batch['img'].to(self.device), batch['target']
                with torch.no_grad():
                    h = self.encoder(img)
                preds = clf_head(h).argmax(dim=1).detach().cpu()
                val_correct += preds.eq(targets.view_as(preds)).sum().item()

            val_acc = val_correct / len(val_loader.dataset)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(clf_head.state_dict(), '../saved_data/models/best_clf_head.ckpt')
                torch.save(optim.state_dict(), '../saved_data/models/best_clf_optim.ckpt')

            print("Epoch {:3d} - [Train accuracy] {:.4f} - [Val accuracy] {:.4f}".format(epoch+1, train_acc, val_acc))

        return {'train_acc': train_acc, 'val_acc': val_acc}


    def mine_neighbors(self, data_loader, img_key='img', k=20):
        
        fvecs = []
        for batch in tqdm(data_loader, leave=False):
            img = batch[img_key].to(self.device)
            with torch.no_grad():
                z = self.proj_head(self.encoder(img))
            z = F.normalize(z, p=2, dim=-1)
            fvecs.extend(z.cpu().detach().numpy())
        fvecs = np.array(fvecs)

        index = faiss.IndexFlatIP(128)
        index.add(fvecs)
        _, indices = index.search(fvecs, k+1)

        with open('../saved_data/other/{}_neighbors.pkl'.format(self.config['dataset']['name']), 'wb') as f:
            pickle.dump(indices, f)


    def train(self, simclr_loader, val_loader):

        best_acc = 0
        for epoch in range(self.config['epochs']):
            cprint('\nEpoch {}/{}'.format(epoch+1, self.config['epochs']), 'green')
            cprint('---------------------------------------------------------', 'green')
            
            loss_list = []
            for i, batch in enumerate(simclr_loader):
                loss = self.train_step(batch)
                loss_list.append(loss)
                wandb.log({'Contrastive loss': loss, 'Epoch': epoch+1})

                if i % int(0.1 * len(simclr_loader)) == 0:
                    print("[Batch] {:4d}/{} - [Constrastive loss] {:.4f}".format(
                        i, len(simclr_loader), loss
                    ))
            
            print("Epoch {} - [Average loss] {:.4f}".format(epoch+1, np.mean(loss_list)))
            wandb.log({'Average contrastive loss': np.mean(loss_list), 'Epoch': epoch+1})

            if (epoch+1) % self.config['eval_every'] == 0:
                mining_acc = self.neighbor_mining_accuracy(val_loader)
                print("\nEpoch {} - [Neighbor mining accuracy] {:.4f}".format(epoch+1, mining_acc))
                wandb.log({'Neighbor mining accuracy': mining_acc})

                if mining_acc > best_acc:
                    print("[INFO] Saving data, accuracy increased from {:.4f} -> {:.4f}".format(best_acc, mining_acc))
                    best_acc = mining_acc
                    torch.save(self.encoder.state_dict(), '../saved_data/models/simclr_encoder.ckpt')
                    torch.save(self.proj_head.state_dict(), '../saved_data/models/simclr_proj_head.ckpt')
                    torch.save(self.optim.state_dict(), '../saved_data/models/simclr_optimizer.ckpt')

        return self.encoder, self.proj_head
                    





    
        




        









    