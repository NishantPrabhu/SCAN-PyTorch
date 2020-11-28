import random
import numpy as np
import torch
import os
import yaml
import logging
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def open_config(file_name):
    # load config file
    config = yaml.safe_load(open(file_name, "r"))
    return config

def init_expt(args, seed=420):
    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some other stuff
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # setup logging directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # reset logger and setup
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.FileHandler(os.path.join(output_dir, 'trainlogs.txt')), logging.StreamHandler()]
    )

    logging.info(f"Logging at {output_dir}")
    
    # open config file and write
    config = open_config(args.config)
    with open(os.path.join(output_dir, 'hyperparams.txt'), 'w') as logs:
        logs.write(yaml.dump(config))
    
    return config, output_dir

# pretty print network
def print_network(model, name=""):
    logging.info(name.rjust(35))
    logging.info('-'*70)
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param'))
    logging.info('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        if p_name[:2] != 'BN' and p_name[:2] != 'bn': # not printing batchnorm layers
            logging.info(
                '{:>25} {:>27} {:>15}'.format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
                )
            )
    logging.info('-'*70)

# get optimizer
def get_optim(config, parameters):
    name = config.get("name", "sgd")
    if name == "sgd":
        return optim.SGD(parameters, lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, nesterov=True)
    else:
        raise NotImplementedError(f"{name} not setup")

# get lr scheduler
def get_scheduler(config, optim):
    name = config.get("name", None)
    warmup_epochs = config.get("warmup_epochs", 0)
    if warmup_epochs>0:
        for param_grp in optim.param_groups:
            param_grp["lr"] = 1e-12/warmup_epochs*param_grp["lr"]
    if name:
        if name == "cosin":
            scheduler = lr_scheduler.CosineAnnealingLR(optim, config["epochs"]-warmup_epochs, eta_min=0.0, last_epoch=-1)
        else:
            raise NotImplementedError(f"{name} not setup")
        return scheduler, warmup_epochs
    else:
        None, warmup_epochs
    