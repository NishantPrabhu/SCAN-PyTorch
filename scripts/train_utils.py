
"""
Helper functions for model training

Author: Nishant Prabhu

"""

import torch 
import numpy as np 


def get_feature_vectors(model, batch):
    """
    Passes data through backbone and returns feature vectors

    """
    image, neighbor, label = batch.values()
    image_fs = model(image, forward_pass='backbone')            # Shape -> (batch_size, 128)
    neighbor_fs = model(neighbor, forward_pass='backbone')      # Shape -> (batch_size, 128)
    combined_ = torch.cat((image_fs, neighbor_fs), dim=0)       # Shape -> (2*batch_size, 128)

    return {'image': image_fs, 'neighbor': neighbor_fs, 'combined': combined_}


def get_train_predictions(model, batch, device):
    """
    Passes data through full model and returns predictions,
    and probabilities of each head. Meant for training data only

    """
    image, neighbor, label, _ = batch.values()
    image, neighbor = image.to(device), neighbor.to(device)
    image_probs = model(image, forward_pass='full')                  # Shape -> (batch_size, n_clusters)
    neighbor_probs = model(neighbor, forward_pass='full')            # Shape -> (batch_size, n_clusters)
    combined_probs = [torch.cat([i, j], dim=0) for i, j in zip(image_probs, neighbor_probs)]             
    
    image_preds = torch.cat([p.argmax(dim=-1) for p in image_probs], dim=0)         # Shape -> (batch_size,)
    neighbor_preds = torch.cat([p.argmax(dim=-1) for p in neighbor_probs], dim=0)   # Shape -> (batch_size,)
    combined_preds = torch.cat((image_preds, neighbor_preds), dim=0)                # Shape -> (2*batch_size,)

    return {
        'image_probs': image_probs,
        'neighbor_probs': neighbor_probs,
        'probs': combined_probs,
        'image_preds': image_preds,
        'neighbor_preds': neighbor_preds,
        'preds': combined_preds,
        'labels': label.to(device) 
    }


def get_val_predictions(model, batch, device):
    """
    Passes data through full model and returns predictions,
    and probabilities of each head. Meant for validation data only

    """
    image, label = batch[0].to(device), batch[1]
    image_probs = model(image, forward_pass='full')                          # Shape -> (batch_size, n_clusters)
    image_probs = [im.cpu() for im in image_probs]
    image_preds = [i.argmax(dim=-1).cpu() for i in image_probs]              # Shape -> (batch_size,)

    return {
        'probs': image_probs,
        'preds': image_preds,
        'labels': label 
    }