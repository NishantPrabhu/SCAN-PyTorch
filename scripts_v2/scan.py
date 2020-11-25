
"""
Main SCAN class and some other stuff

@author: Nishant Prabhu

"""

# Dependencies
import os
import wandb
import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from termcolor import cprint
from torchvision import transforms as T
import torch.nn.functional as F

from logging_utils import Scalar
from models import ClusteringModel, ContrastiveModel
from augment import Augment, Cutout, TensorAugment
from losses import SCANLoss, ConfidenceBasedCE
from train_utils import get_train_predictions, get_val_predictions
from data_utils import generate_data_loaders, generate_embeddings
from sklearn.utils.class_weight import compute_class_weight


cluster_map = {
    'cifar10': 10,
    'cifar100': 100,
    'stl10': 10
}

simclr_pretrained = {
    'cifar10': '../simclr/pretrained_models/simclr_cifar-10.pth.tar',
    'cifar100': '../simclr/pretrained_models/simclr_cifar-20.pth.tar',
    'stl10': '../simclr/pretrained_models/simclr_stl-10.pth.tar'
}


class SCAN():

    def __init__(self, dataset, n_neighbors, transforms, batch_size, learning_rate=1e-04, entropy_weight=5, threshold=0.9):

        self.save_path = './'
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.aug_func = Augment(n=4)
        self.tensor_aug = TensorAugment()
        self.image_transform = transforms['standard']
        self.augment_transform = transforms['augment']
        self.n_clusters = cluster_map[dataset]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thresh = threshold

        # Loss functions
        self.scan_loss = SCANLoss(entropy_weight)
        self.self_labelling_loss = ConfidenceBasedCE(threshold, True)

        # Contrastive model for SimCLR
        self.encoder = ContrastiveModel(dataset, head='mlp', features_dim=128).to(self.device)
        self.encoder.load_state_dict(torch.load(simclr_pretrained[dataset]))

        # Intialize data loaders
        cprint("\n[INFO] Initializing data loaders", 'yellow')
        self.train_loader, self.val_loader = generate_data_loaders(
            name = dataset,
            batch_size = batch_size,
            n_neighbors = n_neighbors,
            transforms = transforms,
            embedding_net = self.encoder,
            augment_fun = self.aug_func,
        )

        # Initialize model and optimizer
        cprint("\n[INFO] Initializing clustering model", 'yellow')
        backbone = {'backbone': self.encoder.backbone, 'dim': self.encoder.backbone_dim}
        self.model = ClusteringModel(backbone, self.n_clusters).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-04)

        print("Complete!")

        # Wandb initialization
        cprint("\n[INFO] Initializing wandb", 'yellow')
        wandb.init(name='scan_' + dataset, project='scan-unsupervised-classification')


    def find_best_head(self):
        """
        Finds the head with minimum loss and its loss
        criterion options: 'loss', 'votes'

        """
        losses = [self.model.heads[i].loss.mean for i in range(self.n_heads)]
        return np.argmin(losses), min(losses)


    def evaluate_model(self):
        """
        Some evaluation method, don't know what to call

        """
        predictions, ground_truth = [], []

        for batch in self.val_loader:
            images, labels = batch
            preds = self.model(images.to(self.device)).argmax(dim=-1).detach().cpu()
            predictions.append(preds)
            ground_truth.append(labels)

        pred = torch.cat(predictions, dim=0)
        gt = torch.cat(ground_truth, dim=0)
        assert pred.shape == gt.shape, "Evaluation pred shape: {}, gt shape: {}".format(pred.shape, gt.shape)

        count_dict = {}
        for i in torch.unique(pred):
            gt_slice = gt[torch.where(pred == i)[0]].numpy()
            most_freq = sorted(gt_slice, key=gt_slice.tolist().count)[-1]
            max_frac = gt_slice.tolist().count(most_freq) / len(gt_slice)
            try:
                count_dict.update({i.item(): max_frac})
            except Exception as e:
                count_dict.update({i: max_frac})

        cprint("\nEvaluation results", 'green')
        for k, v in count_dict.items():
            print("\t{} - {:.4f}".format(k, v))

        return count_dict


    def train_clustering(self, epochs=100, save_frequency=10):
        """
            Main training function.

            Args:
                epochs <int>
                    Number of learning runs over the entire dataset
                entropy_weight <float>
                    Weightage of entropy loss copmared to similarity loss
                save_frequency <int>
                    Number of epochs after which model is checkpointed

            Returns:
                Trained model
        """

        cprint("\n[INFO] Beginning training", 'yellow')

        # Iterate over epochs
        for epoch in range(epochs):

            cprint("\nEPOCH {}/{}".format(epoch+1, epochs), 'green')
            cprint("------------------------------------------", 'green')
            loss_meter = Scalar()

            for i, batch in enumerate(self.train_loader):

                out = get_train_predictions(self.model, batch, self.device)
                total_losses, sim_losses, ent_losses = self.scan_loss(out['anchor_logits'], out['neighbor_logits'])

                # Backpropagate and update model
                self.optim.zero_grad()
                total_losses.backward()
                self.optim.step()
                loss_meter.update(total_losses.item())

                if i % int(0.1 * len(self.train_loader)) == 0:
                    print("[Batch] {:4d}/{} - [Entropy] {:.4f} - [Consistency] {:.4f}".format(
                        i, len(self.train_loader), ent_losses, sim_losses
                    ))

                # Log on wandb
                wandb.log({
                    'Similarity loss': sim_losses,
                    'Entropy loss': ent_losses
                })

            # Logging
            wandb.log({'Mean loss': loss_meter.mean})

            # Summarize epoch
            cprint("Epoch {} - Average total loss: {:.4f}".format(epoch+1, loss_meter.mean), 'green')

            # Every 10th epoch, evaluate
            if (epoch+1) % 10 == 0:
                _ = self.evaluate_model()

            # Save models
            if (epoch+1) % save_frequency == 0:
                torch.save(self.model.state_dict(), '../saved_data/models/clustering_model')
                torch.save(self.optim.state_dict(), '../saved_data/models/optimizer')
                print("\n[INFO] Saved model at epoch {}\n".format(epoch+1))

        return self.model


    def train_self_labelling(self, epochs=100, save_frequency=10):
        """
        Self-labelling function. To be performed after clustering training is complete.
        Will be done only with the head with lowest loss

        """
        # Load the new model into self.model
        torch.cuda.empty_cache()
        self.model.load_state_dict(torch.load('../saved_data/models/clustering_model'))

        # Confident samples based on model's predictions
        # This will be a dictionary in which keys are pseudo-labels and values
        # are a list containing tensors of confident examples with that psuedo-label
        cprint("\n[INFO] Beginning self-labelling", 'yellow')

        self.confident_samples = {}
        for epoch in range(epochs):
            cprint("\nEPOCH {}/{}".format(epoch+1, epochs), 'green')
            cprint("------------------------------------------------------------", 'green')

            accuracy_meter = Scalar()
            loss_meter = Scalar()

            for i, batch in enumerate(self.train_loader):
                out = get_train_predictions(self.model, batch, self.device)
                image_logits, neighbor_logits = out['anchor_logits'], out['neighbor_logits']
                image_probs = F.softmax(image_logits, dim=1)
                x_locs, y_locs = torch.where(image_probs > self.thresh)     # 2d tensor with probabilities

                if len(x_locs) == 0 or len(y_locs) == 0:
                    continue

                for x, y in zip(x_locs, y_locs):
                    if y not in self.confident_samples.keys():
                        self.confident_samples[y.item()] = [(batch['image'][x].unsqueeze(0).detach().cpu(), image_probs[x, y].item())]
                    else:
                        self.confident_samples[y.item()].append((x.detach().cpu(), image_probs[x, y].item()))

                # Supervised training for one epoch
                self.optim.zero_grad()
                loss, acc = self.self_labelling_loss(image_logits, neighbor_logits)
                loss.backward()
                self.optim.step()

                accuracy_meter.update(acc)
                loss_meter.update(loss)

                # Output
                if i % int(0.1 * len(self.train_loader)) == 0:
                    print("[Batch] {:4d}/{} - [Accuracy] {:.4f} - [CE Loss] {:.4f}".format(
                        i, len(self.train_loader), acc, loss))

                    print({k: len(v) for k, v in self.confident_samples.items()})

                wandb.log({'Self-labelling cross entropy': loss})

            # Epoch stats
            cprint("\nEpoch {:3d}/{} - [Average loss] {:.4f} - [Average accuracy] {:.4f}".format(
                epoch+1, epochs, loss_meter.mean, accuracy_meter.mean), 'green')
            cprint("\n=======================================================================", 'green')

            wandb.log({'Average self-labelling accuracy': accuracy_meter.mean})

            if (epoch+1) % save_frequency == 0:
                torch.save(self.model.state_dict(), '../saved_data/models/self_labelling_model')
                torch.save(self.optim.state_dict(), '../saved_data/models/optimizer')
                print("\n[INFO] Saved model at epoch {}\n".format(epoch+1))

        return self.model, self.confident_samples
