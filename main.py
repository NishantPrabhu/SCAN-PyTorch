
""" 
Script to use all other modules and run the thing.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import os
import torch  
import argparse
import train_utils
import setup_utils
import data_utils
import augmentations
import numpy as np 
from datetime import datetime as dt



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to configuration file')
    parser.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str, help='Output directory')
    args = vars(parser.parse_args())
    
    # Initialize experiment
    config, output_dir = setup_utils.init_experiment(args, seed=420)

    # Generate transforms
    data_transforms = {
        'train_transform': augmentations.get_transform(config['train_transform']),
        'val_transform': augmentations.get_transform(config['val_transform']),
        'aug_transform': augmentations.get_transform(config['aug_transform'])
    }

    # Generate datasets
    train_dset = data_utils.get_dataset(
        config = config, 
        split = 'train', 
        transforms = {k: data_transforms[v] for k, v in config['data_transforms']['train'].items()}, 
        return_items = config['train_items']
    )
    val_dset = data_utils.get_dataset(
        config = config, 
        split = 'val', 
        transforms = {k: data_transforms[v] for k, v in config['data_transforms']['val'].items()}, 
        return_items = config['val_items']
    )
    main_dset = data_utils.get_dataset(
        config = config, 
        split = 'train', 
        transforms = {k: data_transforms[v] for k, v in config['data_transforms']['main'].items()}, 
        return_items = config['main_items']
    )

    main_loader = data_utils.get_dataloader(config, main_dset, weigh=True, shuffle=True)
    val_loader = data_utils.get_dataloader(config, val_dset, weigh=False, shuffle=False)
    train_loader = data_utils.get_dataloader(config, train_dset, weigh=False, shuffle=False)


    # Begin training
    trainer = train_utils.Trainer(config, output_dir)
    print("\n[INFO] Beginning training ...")

    if trainer.task == 'simclr':
    
        # Train
        trainer.train(main_loader, val_loader)
        
        # Linear 
        print('\n[INFO] Linear evaluation ...')
        val_acc = trainer.model.linear_eval(train_loader, val_loader)

        # Neighbor mining
        print("\n[INFO] Mining neighbors ...")
        trainer.model.find_neighbors(train_loader, 'img', 'train_neighbors', topk=20)
        trainer.model.find_neighbors(val_loader, 'img', 'val_neighbors', topk=20)


    elif trainer.task == 'scan':

        # Load neighbors
        train_neighbors = np.load(os.path.join(output_dir, 'simclr/{}/{}_train_neighbors.npy'.format(
            config['dataset']['name'], config['encoder']['name']
        )))
        val_neighbors = np.load(os.path.join(output_dir, 'simclr/{}/{}_val_neighbors.npy'.format(
            config['dataset']['name'], config['encoder']['name']
        )))

        # Generate datasets and loaders specific to SCAN
        scan_train_dset = data_utils.NeighborDataset(train_dset, train_neighbors)
        scan_val_dset = data_utils.NeighborDataset(val_dset, val_neighbors)
        scan_train_loader = data_utils.get_dataloader(config, scan_train_dset, weigh=False, shuffle=True)
        scan_val_loader = data_utils.get_dataloader(config, scan_val_dset, weigh=False, shuffle=False)

        # Train
        trainer.train(scan_train_loader, scan_val_loader)


    elif trainer.task == 'selflabel':

        # Train
        trainer.train(main_loader, val_loader)

    else:
        raise NotImplementedError("No task was found in the configuration file")


