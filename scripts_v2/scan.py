
import torch 
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F 
import torch.optim as optim

import wandb
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from models import SimCLR, ClusteringModel, ContrastiveModel, Classifier
from losses import SCANLoss, ConfidenceBasedCELoss
from data_utils import NeighborsDataset, AugmentedDataset, GenDataset
from data_utils import MemoryBank, Scalar


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

    def __init__(self, data_name='cifar10', encoder='scan', n_neighbors=5, batch_size=128, sl_batch_size=1024, learning_rate=1e-04, 
                 entropy_weight=2, threshold=0.9):
        
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
        
        if encoder == 'mukund':
            self.simclr = SimCLR(name='resnet18').to(self.device)
            self.simclr.load_state_dict(torch.load('../saved_data/pretrained/mukund_simclr_cifar-10.ckpt', map_location=self.device))
            self.model = ClusteringModel(self.simclr, 512, meta[self.name]['n_classes']).to(self.device)
            self.opt = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-04)

        elif encoder == 'scan':    
            self.simclr = ContrastiveModel(self.name, head='mlp', feature_dim=128).to(self.device)
            self.simclr.load_state_dict(torch.load('../saved_data/pretrained/simclr_cifar-10.pth.tar', map_location=self.device))
            self.model = ClusteringModel(self.simclr.backbone, self.simclr.backbone_dim, meta[self.name]['n_classes']).to(self.device)
            self.opt = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-04)

        # Loading saved clustering model
        # self.model.load_state_dict(torch.load('../saved_data/models/clustering_model', map_location=self.device))
        # self.opt.load_state_dict(torch.load('../saved_data/models/optimizer', map_location=torch.device('cpu')))
        self.classifier = Classifier(feature_dim=512, num_classes=10)
        self.clf_opt = optim.Adam(self.classifier.parameters(), lr=1e-04, betas=(0.9, 0.999), eps=1e-10)

        # Memory bank and generate initial neighbor_idx
        self.memory = MemoryBank(size=len(self.dataset), dim=512, num_classes=meta[self.name]['n_classes'])
        self.update_simclr_vectors()
        neighbor_idx, acc = self.memory.mine_nearest_neighbors(n_neighbors, True)
        print("Initial neighbor mining accuracy: {:.4f}".format(acc))
        self.neighbor_idx = neighbor_idx

        # Generate NeighborsDataset instance and dataloader for SCAN clustering
        self.neighbors_dset = NeighborsDataset(self.name, self.neighbor_idx)
        self.dataloader = torch.utils.data.DataLoader(self.neighbors_dset, batch_size=batch_size, shuffle=True, num_workers=8)

        # Generate AugmentedDataset instance and dataloader for SCAN self-labelling
        self.augmented_dset = AugmentedDataset(self.name)
        self.augmented_loader = torch.utils.data.DataLoader(self.augmented_dset, batch_size=sl_batch_size, shuffle=True, num_workers=8)

        # Validation data
        val_dset = meta[data_name]['data'](root='../data/{}'.format(data_name), train=False, transform=tensor_transform, download=True)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=8)

        # Loss function
        self.clustering_loss = SCANLoss(entropy_weight=entropy_weight)
        self.self_labelling_loss = ConfidenceBasedCELoss(threshold=threshold, apply_class_balancing=True)
        
        # wandb initialization
        cprint('\n[INFO] Initializing wandb...', 'yellow')
        wandb.init(name='test-run-simple', project='scan-v2')


    def clustering_train_step(self):

        loss_meter = Scalar()
        L, CL, EL = [], [], []

        for i, batch in enumerate(self.dataloader):

            # Compute loss
            anchor, neighbor = batch['anchor'].to(self.device), batch['neighbor'].to(self.device)
            anchor_logits = self.model(anchor)
            neighbor_logits = self.model(neighbor)
            loss, consistency_loss, entropy_loss = self.clustering_loss(anchor_logits, neighbor_logits)
            L.append(loss.item())
            CL.append(consistency_loss.item())
            EL.append(entropy_loss.item())
            
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

        return L, CL, EL
                
            
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
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, leave=False):
                imgs = imgs.to(self.device)
                fv = self.model.backbone(imgs).cpu()
                self.memory.update(fv, labels)


    def self_labelling_train_step(self):

        loss_meter = Scalar()
        acc_meter = Scalar()
        conf_counts = {}

        for i, batch in enumerate(self.augmented_loader):
            
            # Compute self-labelling loss
            anchor, augment = batch['weak'].to(self.device), batch['strong'].to(self.device)
            with torch.no_grad():
                anchor_logits = self.model(anchor)
            augment_logits = self.model(augment)
            loss, acc, masked_target = self.self_labelling_loss(anchor_logits, augment_logits)

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

            if i % int(0.1 * len(self.augmented_loader)) == 0:
                print("Batch {:4d}/{} - [Masked CE Loss] {:.4f} - [Accuracy] {:.4f}".format(
                    i, len(self.augmented_loader), loss.item(), acc
                ))

        wandb.log({'Self labelling accuracy': acc_meter.mean})
        return acc_meter.mean, conf_counts


    def validate(self):
        train_fv, train_labels = [], []
        val_fv, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.train_loader, leave=False):
                img, gt = batch 
                img = img.to(self.device)
                train_fv.append(self.model.backbone(img).cpu())
                train_labels.append(gt)

            for batch in tqdm(self.train_loader, leave=False):
                img, gt = batch
                img = img.to(self.device)
                val_fv.append(self.model.backbone(img).cpu())
                val_labels.append(gt)

        train_fv, train_labels = torch.cat(train_fv, dim=0), torch.cat(train_labels, dim=0)
        val_fv, val_labels = torch.cat(val_fv, dim=0), torch.cat(val_labels, dim=0)

        trainset, valset = GenDataset(train_fv, train_labels), GenDataset(val_fv, val_labels)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=8)

        # Train classifier
        train_correct = 0
        for img, labels in tqdm(train_loader, leave=False):
            out = self.classifier(img)
            loss = F.nll_loss(out, labels, reduction='mean')
            self.clf_opt.zero_grad()
            loss.backward()
            self.clf_opt.step()

            preds = out.argmax(dim=1)
            train_correct += preds.eq(labels.view_as(preds)).sum().item()
            
        # Validate
        val_correct = 0
        with torch.no_grad():
            for img, labels in tqdm(val_loader, leave=False):
                preds = self.classifier(img).argmax(dim=1)
                val_correct += preds.eq(labels.view_as(preds)).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)  
        return train_acc, val_acc


    def train(self, clustering_epochs=200, self_labelling_epochs=100):

        cprint('\n[INFO] Beginning clustering training...', 'yellow')
        L, CL, EL = [], [], []
        hung_acc = []
        # train_accs, val_accs = [], []

        for epoch in range(clustering_epochs):
            cprint('\nEpoch {}/{}'.format(epoch+1, clustering_epochs), 'green')
            cprint('-----------------------------------------------------------------', 'green')

            # Perform training step
            loss, cons_loss, ent_loss = self.clustering_train_step()
            L.extend(loss)
            CL.extend(cons_loss)
            EL.extend(ent_loss)

            # # Validate
            # train_acc, val_acc = self.validate()
            # train_accs.append(train_acc)
            # val_accs.append(val_acc)
            # print("\nTraining accuracy: {:.4f} - Val accuracy {:.4f}".format(train_acc, val_acc))

            # Every 10 epochs, check clustering quality
            # And the save the model and optimizer too
            if (epoch+1) % 2 == 0:
                ratio_dict = self.check_clustering_quality()
                print("\nClustering quality check...")
                acc_vals = []
                for k, v in ratio_dict.items():
                    acc_vals.append(v[1])
                    print("\t{} - ({} - {:.4f})".format(k, v[0], v[1]))
                print("\nAverage hungarian accuracy: {:.4f}".format(sum(acc_vals)/10))
                hung_acc.append(sum(acc_vals)/10)

                # Save model and optim
                torch.save(self.model.state_dict(), '../saved_data/models/clustering_model')
                torch.save(self.opt.state_dict(), '../saved_data/models/optimizer')
                print("\n[INFO] Saved data at epoch {}".format(epoch+1))

        return L, CL, EL, hung_acc


        # cprint('\n[INFO] Beginning self-labelling...', 'yellow')
        # confident_samples = {}
        # best_acc = 0
        
        # for epoch in range(self_labelling_epochs):
        #     cprint("\nEpoch {}/{}".format(epoch+1, self_labelling_epochs), 'green')
        #     cprint('-----------------------------------------------------------------', 'green')

        #     # Perform training step
        #     mean_acc, conf_counts = self.self_labelling_train_step()
        #     for k, v in conf_counts.items():
        #         if k in confident_samples.keys():
        #             confident_samples[k] += v
        #         else:
        #             confident_samples[k] = v

        #     print("\nEpoch {} - [Mean accuracy] {:.4f}".format(epoch+1, mean_acc))
        #     print("Confident sample counts:", confident_samples)

        #     if (epoch+1) % 10 == 0:
        #         ratio_dict = self.check_clustering_quality()
        #         print("\nClustering quality check...")
        #         for k, v in ratio_dict.items():
        #             print("\t{} - {:.4f}".format(k, v[1]))

        #     # Save models every 10 epochs
        #     if mean_acc > best_acc:
        #         print("\n[INFO] Saving model at epoch {}, Accuracy improved from {:.4f} -> {:.4f}".format(
        #             epoch+1, best_acc, mean_acc
        #         ))
        #         best_acc = mean_acc
        #         torch.save(self.model.state_dict(), '../saved_data/models/self_labelling_model')
        #         torch.save(self.opt.state_dict(), '../saved_data/models/optimizer')
        #     else:
        #         print("\n[INFO] Current accuracy {:.4f} lower than best: {:.4f}".format(mean_acc, best_acc))
