import argparse
from datetime import datetime
import utils
import models
import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import wandb
import logging
import numpy as np
import torch

def train_one_epoch(epoch, train_loader, model, output_dir):
    pbar = tqdm(total=len(train_loader), desc=f"train epoch - {epoch}")
    epoch_metrics = {}
    for indx, data in enumerate(train_loader):
        step = epoch*len(train_loader) + indx
        metrics = model.train_one_step(data)
        for key, value in metrics.items():
            if indx == 0:
                epoch_metrics[key] = [value]
            else:
                epoch_metrics[key].append(value)
        wandb.log({**metrics, "train step": step})
        pbar.update(1)
    
    # logs
    log = f"train epoch - {epoch} "
    for key, value in epoch_metrics.items():
        log += f"{key} - {round(np.mean(value), 4)} "
    pbar.set_description(log)
    logging.info(log)
    wandb.log({"lr": model.optim.param_groups[0]["lr"], "epoch": epoch})
    pbar.close()

    # step learning rate
    if epoch+1 <= model.warmup_epochs:
        model.optim.param_groups[0]["lr"] = (epoch+1)/model.warmup_epochs*model.lr
    elif model.lr_scheduler is not None:
        model.lr_scheduler.step()

    model.save(output_dir, "last")

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(description="simclr training script")
    arg_parser.add_argument("-c", "--config", required=True, type=str, help="path to config file")
    arg_parser.add_argument("-o", "--output_dir", default=datetime.now().strftime('%Y-%m-%d_%H-%M'), type=str, help="output directory")
    args = arg_parser.parse_args()
    
    # start experiment
    config, output_dir = utils.init_expt(args)
    if "simclr" in args.config:
        wandb.init("simclr")
        simclr = models.SimCLR(config) 
        
        # image transforms
        train_transform = datasets.get_transform(config["train_transform"])
        valid_transform = datasets.get_transform(config["valid_transform"])
        aug_transform = datasets.get_transform(config["aug_transform"])
        
        # datasets and dataloaders
        simclr_dataset = datasets.get_dataset(config=config, split="train", transforms={"i": aug_transform, "j": aug_transform}, return_items={"i", "j"})
        sample_weights = datasets.sample_weights(simclr_dataset.targets)
        simclr_loader = DataLoader(simclr_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True), drop_last=True)

        train_dataset = datasets.get_dataset(config=config, split="train", transforms={"train_img": train_transform}, return_items={"train_img", "target"})
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True)

        val_dataset = datasets.get_dataset(config=config, split="val", transforms={"val_img": valid_transform}, return_items={"val_img", "target"})
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=False)
        
        best = 0
        print("\n")
        for epoch in range(config["epochs"]):
            train_one_epoch(epoch, simclr_loader, simclr, output_dir)
            if epoch%config["valid_every"] == 0:
                metrics = simclr.validate(epoch, val_loader)
                wandb.log({**metrics, "val epoch": epoch})
                log = f"valid epoch - {epoch} "
                for key, value in metrics.items():
                    log += f"{key} - {round(value, 4)} "
                logging.info(log)
                if metrics["acc"] > best:
                    best = metrics["acc"]
                    simclr.save(output_dir, "best")
        print("\nTraining complete. Performing Linear evaluation")
        # load the best encoder model
        simclr.enc.load_state_dict(torch.load(os.path.join(output_dir, "best_encoder.ckpt")))
        metrics = simclr.linear_eval(train_loader, val_loader, output_dir)
        log = f"\nLinear Evaluation:"
        for key, value in metrics.items():
            log += f" {key} - {round(value, 4)}"
        logging.info(log)
        print("\nTraining Complete. Exitting")
    
    else:
        raise NotImplementedError()    