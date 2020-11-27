
"""
Main script

"""

import torch 
from scan import SCAN 


if __name__ == '__main__':

    # Parameters
    data_name = 'cifar10'
    n_neighbors = 20
    batch_size = 128
    learning_rate = 1e-04
    entropy_weight = 5
    threshold = 0.9
    clustering_epochs = 200
    self_labelling_epochs = 100

    # Initialize SCAN
    scan_model = SCAN(data_name, n_neighbors, batch_size, learning_rate, entropy_weight, threshold)

    # Train
    scan_model.train(clustering_epochs, self_labelling_epochs)