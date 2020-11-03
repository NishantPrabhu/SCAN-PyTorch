
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
from evaluation import similarity_loss, entropy_loss, hungarian_evaluate
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

    def __init__(self, dataset, n_heads, n_neighbors, transforms, batch_size, learning_rate=1e-04, entropy_weight=5):

        self.save_path = './'
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.aug_func = Augment(n=4)
        self.cutout_func = Cutout(length=5, n_holes=1)
        self.tensor_aug = TensorAugment()
        self.image_transform = transforms['standard']
        self.augment_transform = transforms['augment']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = cluster_map[dataset]

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
            cutout_fun = self.cutout_func
        )

        # Initialize model and optimizer
        if not os.path.exists('../saved_data/clustering_model'):

            cprint("\n[INFO] Initializing clustering model", 'yellow')
            self.n_heads = n_heads
            self.model = ClusteringModel(dataset, self.n_clusters, self.n_heads,
                feature_dim=128, pretrained_model=simclr_pretrained[dataset]).to(self.device)
            self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)

        else:

            latest_model = '../saved_data/clustering_model'
            latest_optim = '../saved_data/optimizer'
            self.model = ClusteringModel(dataset, self.n_clusters, self.n_heads,
                feature_dim=128, pretrained_model=simclr_pretrained[dataset]).to(self.device)
            self.model.load_state_dict(torch.load(latest_model, map_location=self.device))
            self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.optim.load_state_dict(torch.load(latest_optim))

        print("Complete!")

        # Wandb initialization
        cprint("\n[INFO] Initializing wandb", 'yellow')
        wandb.init(
            name='scan_' + dataset,
            project='scan-unsupervised-classification'
        )


    def compute_loss(self, image_out, neighbor_out, combined_out):
        """
        Loss computation: similarity loss, entropy loss and total loss are returned

        """
        sim_losses, ent_losses, total_losses = [], [], []

        for i in range(self.n_heads):
            sim_loss = similarity_loss(image_out[i], neighbor_out[i])
            ent_loss = entropy_loss(combined_out[i])
            total_loss = sim_loss - self.entropy_weight * (1. / (ent_loss + 1e-06))

            sim_losses.append(sim_loss)
            ent_losses.append(ent_loss)
            total_losses.append(total_loss)

        return sim_losses, ent_losses, total_losses


    def find_best_head(self, criterion='loss'):
        """
        Finds the head with minimum loss and its loss
        criterion options: 'loss', 'votes'

        """
        if criterion == 'loss':
            losses = [self.model.heads[i].loss.mean for i in range(self.n_heads)]
            return np.argmin(losses), min(losses)
        elif criterion == 'votes':
            votes = [self.model.heads[i].best_head for i in range(self.n_heads)]
            return np.argmax(votes), max(votes)


    def evaluate_model(self, epoch, best_head):
        """
        Hungarian evaluation routine with validation data

        """
        self.model.eval()
        probs, preds, labels = [], [], []

        for batch in self.val_loader:
            out = get_val_predictions(self.model, batch, self.device)
            probs.extend(out['probs'])
            preds.extend(out['preds'])
            labels.extend(out['labels'])

        all_predictions = {'probs': probs, 'preds': preds, 'labels': labels}
        results = hungarian_evaluate(
            subhead_index=best_head,
            all_predictions=all_predictions,
            class_names=None,
            confusion_matrix=True,
            confusion_matrix_kwargs={'title': '', 'name': 'confusion_matrix_{}.jpeg'.format(epoch+1)}
        )

        return results


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

            for i, batch in enumerate(self.train_loader):

                # Generate outputs
                out = get_train_predictions(self.model, batch, self.device)

                # Compute similarity loss and entropy loss
                _, _, total_losses = self.compute_loss(out['image_probs'], out['neighbor_probs'], out['probs'])

                # Update loss on loss_meter
                for j in range(self.n_heads):
                    self.model.heads[j].loss.update(total_losses[j].item())

                # Backpropagate and update model
                self.optim.zero_grad()
                self.model.backpropagate(total_losses)
                self.optim.step()

                # Intra-epoch logging
                min_loss_head, min_loss = self.find_best_head()
                self.model.heads[min_loss_head].best_head += 1

                if i % int(0.1 * len(self.train_loader)) == 0:
                    print("[Batch] {:4d}/{} - [Minimum Total Loss] {:.4f} - [Winner Head] {}".format(
                        i, len(self.train_loader), min_loss, min_loss_head))

                # Log on wandb
                wandb.log({'Minimum loss': min_loss})


            # Find head with lowest final total loss
            losses = [self.model.heads[i].loss.mean for i in range(self.n_heads)]
            mean_total_loss = np.mean(losses)
            head_ranks = np.argsort(losses)

            # Clear kooda from CUDA
            torch.cuda.empty_cache()

            # Validate model
            # results = self.evaluate_model(epoch=epoch+1, best_head=head_ranks[0])
            # accuracy, ari, nmi = results['accuracy'], results['ARI'], results['NMI']

            # Logging
            wandb.log({
                'Mean loss': mean_total_loss,
                # 'Hungarian accuracy': accuracy,
                # 'Adjusted rand index': ari,
                # 'Normalized mutual information': nmi
            })

            # Summarize epoch
            cprint("\nEPOCH {}".format(epoch+1), 'green')
            cprint("\tAverage total loss                {:.4f}".format(mean_total_loss), 'green')
            # cprint("\tHungarian accuracy                {:.4f}".format(accuracy), 'green')
            # cprint("\tAdjusted rand score               {:.4f}".format(ari), 'green')
            # cprint("\tNormalized mutual information     {:.4f}".format(nmi), 'green')
            cprint("\n=======================================================", 'green')

            # Save models
            if (epoch+1) % save_frequency == 0:
                torch.save(self.model.state_dict(), '../saved_data/models/clustering_model')
                torch.save(self.optim.state_dict(), '../saved_data/models/optimizer')
                print("\n[INFO] Saved model at epoch {}\n".format(epoch+1))

        return self.model


    def train_supervised(self, best_head):
        """
        Helper function for self-labelling.

        Confident samples for each label are augmented once
        and the model is trained in supervised manner with the
        pseudo-labels that it generated

        """
        X, y = [], []

        for k, v in self.confident_samples.items():
            X.extend([im[0] for im in v])
            y.extend([k]*len(v))

        weights = [1./y.count(a) if a in y else 1. for a in np.arange(self.n_clusters)]
        weights = torch.FloatTensor(weights).to(self.device)

        X = torch.cat(X, dim=0).to(self.device)
        y = torch.LongTensor(y).to(self.device)

        # Augment X amd generate new data
        X_aug = self.tensor_aug(X).to(self.device)

        # Train model with this X and y
        self.model.train()
        image_probs = self.model(X, forward_pass='branch_{}'.format(best_head))
        augment_probs = self.model(X_aug, forward_pass='branch_{}'.format(best_head))
        ce_loss = F.cross_entropy(image_probs, y) + F.cross_entropy(augment_probs, y)

        self.optim.zero_grad()
        ce_loss.backward()
        self.optim.step()

        # Compute accuracy
        image_preds = image_probs.argmax(dim=-1)
        augment_preds = augment_probs.argmax(dim=-1)
        correct = image_preds.eq(y.view_as(image_preds)).sum().item() + \
                    augment_preds.eq(y.view_as(augment_preds)).sum().item()
        accuracy = correct/(2*X.size(0))

        return accuracy, ce_loss.item()


    def train_self_labelling(self, threshold=0.99, epochs=100, save_frequency=10):
        """
        Self-labelling function. To be performed after clustering training is complete.
        Will be done only with the head with lowest loss

        """
        # Load the new model into self.model
        self.model.load_state_dict(torch.load('../saved_data/models/clustering_model'))

        # Confident samples based on model's predictions
        # This will be a dictionary in which keys are pseudo-labels and values
        # are a list containing tensors of confident examples with that psuedo-label
        cprint("\n[INFO] Beginning self-labelling", 'yellow')

        self.confident_samples = {}

        # Find best head
        best_head, _ = self.find_best_head(criterion='loss')

        # Iterate
        for epoch in range(epochs):

            cprint("\nEPOCH {}/{}".format(epoch+1, epochs), 'green')
            cprint("------------------------------------------------------------", 'green')

            accuracy_meter = Scalar()
            loss_meter = Scalar()

            for i, batch in enumerate(self.train_loader):
                out = get_train_predictions(self.model, batch, self.device)
                image_probs = out['image_probs'][best_head]
                x_locs, y_locs = torch.where(image_probs > threshold)     # 2d tensor with probabilities

                if len(x_locs) == 0 or len(y_locs) == 0:
                    continue

                for x, y in zip(x_locs, y_locs):
                    if y not in self.confident_samples.keys():
                        self.confident_samples[y.item()] = [(batch['image'][x].unsqueeze(0).detach().cpu(),
                                                            image_probs[x, y].item())]
                    else:
                        self.confident_samples[y.item()].append((x.detach().cpu(), image_probs[x, y].item()))

                # Supervised training for one epoch
                acc, loss = self.train_supervised(best_head)
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

        return self.model, best_head, self.confident_samples
