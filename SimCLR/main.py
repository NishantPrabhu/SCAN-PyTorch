
import torch
import yaml
import random
import argparse
import data_utils
import numpy as np 
from simclr import SimCLR
from termcolor import cprint
from torch.utils.data import DataLoader, WeightedRandomSampler 


def init_experiment(config_file, seed=42):
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Other 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    config = yaml.safe_load(open(config_file, 'r'))
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simclr')
    parser.add_argument('-c', '--config', required=True, help='path to config file')
    args = vars(parser.parse_args())
    config = init_experiment(args['config'], seed=42)
    
    # Transforms
    cprint('\n[INFO] Acquiring transforms ...', 'yellow')
    train_transform = data_utils.get_transform(config['train_transform'])
    val_transform = data_utils.get_transform(config['val_transform'])
    aug_transform = data_utils.get_transform(config['aug_transform'])
    print("Done!")

    # Datasets and loaders    
    cprint('\n[INFO] Initializing data loaders ...', 'yellow')
    loader_params = {'batch_size': config['batch_size'], 'num_workers': config['num_workers']}
    simclr_dataset = data_utils.get_dataset(
        config = config, 
        split = 'train', 
        transforms = {'i': aug_transform, 'j': aug_transform},
        return_items = {'i', 'j'}
    )
    sample_weights = data_utils.sample_weights(simclr_dataset.targets)
    simclr_loader = DataLoader(
        simclr_dataset, 
        sampler=WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True),
        drop_last=True,
        **loader_params
    )

    train_dataset = data_utils.get_dataset(config=config, split='train', transforms={'img': train_transform}, return_items={'img', 'target'})
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_params)

    val_dataset = data_utils.get_dataset(config=config, split='val', transforms={'img': val_transform}, return_items={'img', 'target'})
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)

    # Begin training
    cprint('\n[INFO] Beginning training ...', 'yellow')
    simclr = SimCLR(config)
    last_encoder, last_proj_head = simclr.train(simclr_loader, val_loader)

    torch.save(last_encoder.state_dict(), '../saved_data/models/last_simclr_encoder.ckpt')
    torch.save(last_proj_head.state_dict(), '../saved_data/models/last_simclr_proj_head.ckpt')

    # Linear evaluation
    cprint('\n[INFO] Training complete. Linear evalution ...', 'yellow')
    metrics = simclr.linear_evaluation(train_loader, val_loader)
    print("Training accuracy: {:.4f}".format(metrics['train_acc']))
    print("Validation accuracy: {:.4f}".format(metrics['val_acc']))

    # Mine neighbors and save
    cprint('\n[INFO] Mining neighbors and saving ...', 'yellow')
    simclr.mine_neighbors(train_loader, img_key='img', k=20)




