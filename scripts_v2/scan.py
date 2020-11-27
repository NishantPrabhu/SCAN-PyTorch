
import torch 
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F 
import torch.optim as optim

import wandb
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from models import SimCLR, ClusteringModel, ContrastiveModel
from losses import SCANLoss, ConfidenceBasedCELoss
from data_utils import NeighborsDataset, MemoryBank, Scalar


# Metadata
meta = {
    'cifar10': {'data': datasets.CIFAR10, 'n_classes': 10},
    'cifar100': {'data': datasets.CIFAR100, 'n_classes': 100},
    'stl10': {'data': datasets.STL10, 'n_classes': 10}
}

norms = {
    'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
    'stl10': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
}

backbone_dims = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048
}


class SCAN:

    def __init__(self, data_name='cifar10', n_neighbors=5, batch_size=128, learning_rate=1e-04, entropy_weight=2, threshold=0.9):
        
        self.name = data_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_neighbors = n_neighbors
        cprint('\n[INFO] Device found: {}'.format(self.device), 'yellow')
        
        # Dataset
        tensor_transform = T.Compose([
            T.ToTensor(), 
            T.Normalize(norms[self.name]['mean'], norms[self.name]['std'])
        ])
        self.dataset = meta[self.name]['data'](root='../data/{}'.format(self.name), train=True, 
                                               transform=tensor_transform, download=True)
        
        # Model and optimizer
        cprint('\n[INFO] Initializing model and optimizer', 'yellow')
        # self.simclr = SimCLR(name='resnet18', feature_dim=128).to(self.device)
        # self.simclr.load_state_dict(torch.load('../saved_data/models/simclr_cifar10.pth.tar', map_location=self.device))
        
        self.simclr = ContrastiveModel(self.name, head='mlp', feature_dim=128).to(self.device)
        self.simclr.load_state_dict(torch.load('../saved_data/pretrained/simclr_cifar-10.pth.tar', map_location=self.device))
        self.model = ClusteringModel(self.simclr.backbone, self.simclr.backbone_dim, meta[self.name]['n_classes']).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-04)

        # Memory bank and generate initial neighbor_idx
        self.memory = MemoryBank(size=len(self.dataset), dim=self.simclr.backbone_dim, num_classes=meta[self.name]['n_classes'])
        self.update_simclr_vectors()
        neighbor_idx, acc = self.memory.mine_nearest_neighbors(n_neighbors, True)
        print("Initial neighbor mining accuracy: {:.4f}".format(acc))

        # Generate NeighborsDataset instance and dataloader for SCAN
        self.neighbors_dset = NeighborsDataset(self.name, neighbor_idx)
        self.dataloader = torch.utils.data.DataLoader(self.neighbors_dset, batch_size=batch_size, shuffle=True, num_workers=8)

        # Loss function
        self.clustering_loss = SCANLoss(entropy_weight=entropy_weight)
        self.self_labelling_loss = ConfidenceBasedCELoss(threshold=threshold, apply_class_balancing=True)
        
        # wandb initialization
        cprint('\n[INFO] Initializing wandb...', 'yellow')
        wandb.init(name='test-run-simple', project='scan-v2')


    def clustering_train_step(self):

        loss_meter = Scalar()
        for i, batch in enumerate(self.dataloader):

            # Compute loss
            anchor, neighbor = batch['anchor'].to(self.device), batch['neighbor'].to(self.device)
            anchor_logits = self.model(anchor)
            neighbor_logits = self.model(neighbor)
            loss, consistency_loss, entropy_loss = self.clustering_loss(anchor_logits, neighbor_logits)
            
            # Backprop and update
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_meter.update(loss.item())
            wandb.log({
                'Total loss': loss.item(),
                'Consistency loss': consistency_loss.item(),
                'Entropy loss': entropy_loss.item() 
            })

            if i % int(0.1 * len(self.dataloader)) == 0:
                print("Batch {:4d}/{} - [Consistency] {:.4f} - [Entropy] {:.4f}".format(
                    i, len(self.dataloader), consistency_loss, entropy_loss
                ))
                
            
    def check_clustering_quality(self):

        preds, gt = [], []
        with torch.no_grad():
            for batch in self.dataloader:
                anchor = batch['anchor'].to(self.device)
                output = self.model(anchor).argmax(dim=-1)        
                preds.append(output)
                gt.append(batch['label'])

        preds = torch.cat(preds, dim=0).cpu().numpy()
        gt = torch.cat(gt, dim=0).numpy()
        ratio_dict = {}

        for l in np.unique(preds):
            labels = gt[np.where(preds == l)[0]]
            max_count_label = sorted(labels, key=labels.tolist().count)[-1]
            max_frac = labels.tolist().count(max_count_label)/len(labels)
            ratio_dict.update({l: (max_count_label, max_frac)})

        return ratio_dict


    def update_simclr_vectors(self):

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        for imgs, labels in tqdm(dataloader, leave=False):
            imgs = imgs.to(self.device)
            fv = self.simclr.backbone(imgs).detach().cpu()
            self.memory.update(fv, labels)


    def self_labelling_train_step(self):

        loss_meter = Scalar()
        acc_meter = Scalar()
        conf_counts = {}

        for i, batch in enumerate(self.dataloader):
            
            # Compute self-labelling loss
            anchor, neighbor = batch['anchor'].to(self.device), batch['neighbor'].to(self.device)
            anchor_logits = self.model(anchor)
            neighbor_logits = self.model(neighbor)
            loss, acc, masked_target = self.self_labelling_loss(anchor_logits, neighbor_logits)

            # Backprop and update model
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_meter.update(loss.item())
            acc_meter.update(acc)
            wandb.log({'Self labelling CE loss': loss.item()})

            # Check how many confident samples are there
            uniq, count = torch.unique(masked_target, return_counts=True)
            uniq, count = uniq.cpu().numpy(), count.cpu().numpy()

            for u, c in zip(uniq, count):
                if u in conf_counts.keys():
                    conf_counts[u] += c
                else:
                    conf_counts[u] = c

            if i % int(0.1 * len(self.dataloader)) == 0:
                print("Batch {:4d}/{} - [Masked CE Loss] {:.4f} - [Accuracy] {:.4f}".format(
                    i, len(self.dataloader), loss.item(), acc
                ))

        return acc_meter.mean, conf_counts


    def train(self, clustering_epochs=200, self_labelling_epochs=100):

        cprint('\n[INFO] Beginning clustering training...', 'yellow')

        for epoch in range(clustering_epochs):
            cprint('\nEpoch {}/{}'.format(epoch+1, clustering_epochs), 'green')
            cprint('-----------------------------------------------------', 'green')

            # Perform training step
            self.clustering_train_step()

            # Every 10 epochs, check clustering quality and update simclr vectors
            # And the save the model and optimizer too
            if (epoch+1) % 10 == 0:
                ratio_dict = self.check_clustering_quality()
                print("\nClustering quality check...")
                for k, v in ratio_dict.items():
                    print("{} - {:.4f}".format(k, v[1]))

                # Update simclr vectors
                self.update_simclr_vectors()
                neighbor_idx, acc = self.memory.mine_nearest_neighbors(self.n_neighbors, True)
                print("\nNeighbor mining accuracy: {:.4f}".format(float(acc)))
                self.neighbors_dset = NeighborsDataset(self.name, neighbor_idx)
                self.dataloader = torch.utils.data.DataLoader(self.neighbors_dset, batch_size=self.batch_size, 
                                                              shuffle=True, num_workers=8)
                wandb.log({'Neighbor mining accuracy': acc})

                # Save model and optim
                torch.save(self.model.state_dict(), '../saved_data/models/clustering_model')
                torch.save(self.opt.state_dict(), '../saved_data/models/optimizer')
                print("[INFO] Saved data at epoch {}".format(epoch+1))


        cprint('\n[INFO] Beginning self-labelling...', 'yellow')

        for epoch in range(self_labelling_epochs):
            cprint("\nEpoch {}/{}".format(epoch+1, self_labelling_epochs), 'green')
            cprint('-----------------------------------------------------', 'green')

            # Perform training step
            mean_acc, conf_counts = self.self_labelling_train_step()

            print("\nEpoch {} - [Mean accuracy] {:.4f}".format(epoch+1, mean_acc))
            print("Confident sample counts:", conf_counts)

            # Save models every 10 epochs
            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), '../saved_data/models/self_labelling_model')
                torch.save(self.opt.state_dict(), '../saved_data/models/optimizer')