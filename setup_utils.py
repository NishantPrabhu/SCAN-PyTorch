
"""
Helper functions for training and optimization.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import random 
import numpy as np
import os 
import torch 
import yaml 
import logging 
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler


def open_config(file_path):
    """
    Opens a configuration file.
    """
    config = yaml.safe_load(open(file_path, 'r'))
    return config


def init_experiment(args, seed=420):
    """
    Instantiates output directories, logging files
    and random seeds.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Some other stuff
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    # Setup logging directory
    output_dir = args['output']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reset logger and setup
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.FileHandler(os.path.join(output_dir, 'trainlogs.txt'))]
    )

    print('\n[INFO] Logging at {}'.format(output_dir))

    # Open config file and write
    config = open_config(args['config'])
    with open(os.path.join(output_dir, 'hyperparameters.txt'), 'w') as logs:
        logs.write(yaml.dump(config))

    return config, output_dir


def print_network(model, name=''):
    """
    Pretty prints the model.
    """
    print(name.rjust(35))
    print('-'*70)
    print('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param'))
    print('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2] + '.' + param.split('.')[-1]
        if p_name[:2] != 'BN' and p_name[:2] != 'bn':    # Not printing batch norm layers
            print(
                '{:>25} {:>27} {:>15}'.format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
                )
            )
    print('-'*70 + '\n')


def get_optimizer(config, params):
    """
    Initializes an optimizer with provided configuration.
    """
    name = config.get('name', 'sgd')
    if name == 'sgd':
        return optim.SGD(params=params, lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9, nesterov=True)
    elif name == 'adam':
        return optim.Adam(params=params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError("Invalid optimizer {}".format(name))


def get_scheduler(config, optimizer):
    """
    Initializes a scheduler with provided configuration.
    """
    name = config.get('name', None)
    warmup_epochs = config.get('warmup_epochs', 0)

    if warmup_epochs > 0:
        for group in optimizer.param_groups:
            group['lr'] = 1e-12/warmup_epochs * group['lr']

    if name is not None:
        if name == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, config['epochs']-warmup_epochs, eta_min=0.0, last_epoch=-1)
        else:
            raise NotImplementedError('Invalid scheduler {}'.format(name))
        return scheduler, warmup_epochs

    else:
        return None, warmup_epochs

