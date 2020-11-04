
"""
Main script to call all other functions

Author: Nishant Prabhu
"""

import torch
from torchvision import transforms as T
from scan import SCAN
import argparse
import pickle


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default='cifar10', type=str, help='name of dataset to work with')
    ap.add_argument("-o", "--heads", default=10, type=int, help='number of heads in model')
    ap.add_argument("-n", "--neighbors", default=20, type=int, help='number of nearest neighbors to mine')
    ap.add_argument("-b", "--batch-size", default=1000, type=int, help='batch size')
    ap.add_argument("-l", "--lr", default=1e-04, type=float, help='model learning rate')
    ap.add_argument("-w", "--entropy-weight", default=5.0, type=float, help='weight of entropy loss')
    ap.add_argument("-c", "--clustering-epochs", default=100, type=int, help='epochs for clustering task')
    ap.add_argument("-s", "--self-labelling-epochs", default=100, type=int, help='epochs for self labelling task')
    ap.add_argument("-f", "--save-frequency", default=10, type=int, help='frequency of saving trained model')
    ap.add_argument("-t", "--threshold", default=0.99, type=float, help='threshold probability for self labelling')
    args = vars(ap.parse_args())

    # Normalization constants
    norm_const = {
        'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]},
        'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.267, 0.256, 0.276]},
        'stl10': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    }

    # Define image transforms
    standard_transform = T.Compose([
        T.Resize(size=32),
        T.CenterCrop(size=32),
        T.ToTensor(),
        T.Normalize(norm_const[args['dataset']]['mean'], norm_const[args['dataset']]['std'])
    ])

    augment_transform = T.Compose([
        T.Resize(size=32),
        T.CenterCrop(size=32),
        T.ToTensor(),
        T.Normalize(norm_const[args['dataset']]['mean'], norm_const[args['dataset']]['std'])
    ])

    transforms = {'standard': standard_transform, 'augment': augment_transform}

    # Initialize SCAN model
    scan_model = SCAN(
        dataset = args['dataset'],
        n_heads = args['heads'],
        n_neighbors = args['neighbors'],
        transforms = transforms,
        batch_size = args['batch_size'],
        learning_rate = args['lr'],
        entropy_weight = args['entropy_weight']
    )

    # Train for clustering
    trained_model = scan_model.train_clustering(epochs=args['clustering_epochs'], save_frequency=args['save_frequency'])
    torch.save(trained_model.state_dict(), '../saved_data/models/clustering_model')

    # Clear CUDA of everything
    torch.cuda.empty_cache()

    # Self labelling
    trained_model, best_head, conf_samples = \
        scan_model.train_self_labelling(epochs=args['self_labelling_epochs'], threshold=args['threshold'])

    torch.save(trained_model.state_dict(), '../saved_data/models/self_labelling_model')

    with open('../saved_data/other/best_head.pkl', 'wb') as f:
        pickle.dump(best_head, f)

    with open('../saved_data/other/confident_samples.pkl', 'wb') as f:
        pickle.dump(conf_samples, f)
