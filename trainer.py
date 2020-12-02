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
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=False)

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
        print("\ntraining complete. performing linear evaluation")
        # load the best encoder model
        simclr.enc.load_state_dict(torch.load(os.path.join(output_dir, "best_encoder.ckpt")))
        metrics = simclr.linear_eval(train_loader, val_loader, output_dir)
        log = f"\nlinear evaluation:"
        for key, value in metrics.items():
            log += f" {key} - {round(value, 4)}"
        logging.info(log)
        
        print(f"finding nearest neighbours")        
        # find neighbours and save for later
        simclr.proj_head.load_state_dict(torch.load(os.path.join(output_dir, "best_proj_head.ckpt")))
        simclr.find_neighbours(data_loader=train_loader, img_key="train_img", f_name=os.path.join(output_dir, "train_neighbours.npy"), topk=20)
        
        simclr.find_neighbours(data_loader=val_loader, img_key="val_img", f_name=os.path.join(output_dir, "val_neighbours.npy"), topk=20)

    elif "scan" in args.config:
        wandb.init("scan")
        
        scan = models.SCAN(config=config)
        scan.enc.load_state_dict(torch.load(os.path.join(config["simclr_save"], "best_encoder.ckpt")))
        
        train_neighbours = np.load(os.path.join(config["simclr_save"], "train_neighbours.npy"))
        val_neighbours = np.load(os.path.join(config["simclr_save"], "val_neighbours.npy"))
        
        # image transforms
        aug_transform = datasets.get_transform(config["aug_transform"])
        val_transform = datasets.get_transform(config["valid_transform"])
        
        train_img_dataset = datasets.get_dataset(config=config, split="train", transforms={"img": aug_transform}, return_items={"img", "target"})
        scan_dataset = datasets.NeighbourDataset(img_dataset=train_img_dataset, neighbour_indices=train_neighbours)
        scan_loader = DataLoader(scan_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True, drop_last=True)
        
        val_img_dataset = datasets.get_dataset(config=config, split="val", transforms={"img": val_transform}, return_items={"img", "target"})
        val_dataset = datasets.NeighbourDataset(img_dataset=val_img_dataset, neighbour_indices=val_neighbours)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=False, drop_last=True)
        
        best = 0
        print("\n")
        for epoch in range(config["epochs"]+1):
            train_one_epoch(epoch, scan_loader, scan, output_dir)
            if epoch%config["valid_every"] == 0:
                # find the best head and evaluate
                metrics = scan.validate(epoch, val_loader)
                wandb.log({**metrics, "val epoch": epoch})
                log = f"valid epoch - {epoch} "
                for key, value in metrics.items():
                    log += f"{key} - {round(value, 4)} "
                logging.info(log)
                if metrics["acc"] > best:
                    best = metrics["acc"]
                    scan.save(output_dir, "best")

        print("\ntraining complete.")
    elif "selflabel" in args.config:
        wandb.init("selflabel")
        
        selflabel = models.SelfLabel(config=config)
        selflabel.enc.load_state_dict(torch.load(os.path.join(config["clustering_save"], "best_encoder.ckpt")))
        selflabel.cluster_head.load_state_dict(torch.load(os.path.join(config["clustering_save"], "best_cluster_head.ckpt")))
        
        # image transforms
        aug_transform = datasets.get_transform(config["aug_transform"])
        std_transform = datasets.get_transform(config["std_transform"])
        
        selflabel_dataset = datasets.get_dataset(config=config, split="train", transforms={"anchor": std_transform, "anchor_aug": aug_transform}, return_items={"anchor", "anchor_aug", "target"})
        selflabel_loader = DataLoader(selflabel_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True, drop_last=False)
        
        val_dataset = datasets.get_dataset(config=config, split="val", transforms={"img": std_transform}, return_items={"img", "target"})
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=False, drop_last=True)
        
        best = 0
        print("\n")
        for epoch in range(config["epochs"]+1):
            train_one_epoch(epoch, selflabel_loader, selflabel, output_dir)
            if epoch%config["valid_every"] == 0:
                # find the best head and evaluate
                metrics = selflabel.validate(epoch, val_loader)
                wandb.log({**metrics, "val epoch": epoch})
                log = f"valid epoch - {epoch} "
                for key, value in metrics.items():
                    log += f"{key} - {round(value, 4)} "
                logging.info(log)
                if metrics["acc"] > best:
                    best = metrics["acc"]
                    selflabel.save(output_dir, "best")

        print("\ntraining complete.")
    else:
        raise NotImplementedError()    