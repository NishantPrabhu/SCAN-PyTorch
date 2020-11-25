
"""
Helper functions for model training

Author: Nishant Prabhu

"""

import torch
import numpy as np


# def get_feature_vectors(model, batch):
#     """
#     Passes data through backbone and returns feature vectors
#
#     """
#     image, neighbor, label = batch.values()
#     image_fs = model(image, forward_pass='backbone')            # Shape -> (batch_size, 128)
#     neighbor_fs = model(neighbor, forward_pass='backbone')      # Shape -> (batch_size, 128)
#     combined_ = torch.cat((image_fs, neighbor_fs), dim=0)       # Shape -> (2*batch_size, 128)
#
#     return {'image': image_fs, 'neighbor': neighbor_fs, 'combined': combined_}


def get_train_predictions(model, batch, device):
    """
    Passes data through full model and returns predictions,
    and probabilities of each head. Meant for training data only

    """
    image, neighbor, label, _ = batch.values()
    image, neighbor = image.to(device), neighbor.to(device)
    image_logits = model(image, forward_pass='full')                  # Shape -> (batch_size, n_clusters)
    neighbor_logits = model(neighbor, forward_pass='full')            # Shape -> (batch_size, n_clusters)

    return {
        'anchor_logits': image_logits,
        'neighbor_logits': neighbor_logits,
        'labels': label.to(device)
    }


def get_val_predictions(model, batch, device):
    """
    Passes data through full model and returns predictions,
    and probabilities of each head. Meant for validation data only

    """
    image, label = batch[0].to(device), batch[1]
    image_logits = model(image, forward_pass='full')                          # Shape -> (batch_size, n_clusters)
    image_logits = [im.cpu() for im in image_probs]

    return {
        'logits': image_logits,
        'labels': label
    }
