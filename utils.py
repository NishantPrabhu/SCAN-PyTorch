import yaml
import random
import torch
import numpy as np
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def open_config(file_name):
    # load config file
    config = yaml.safe_load(open(file_name, "r"))
    return config

def init_expt(output_dir, seed=420):
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"logging at {output_dir}")

def get_optim(config, parameters):
    name = config.get("name", "adam")
    if name == "sgd":
        return optim.SGD(parameters, lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, nesterov=True)
    else:
        raise NotImplementedError(f"{name} not setup")

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
    