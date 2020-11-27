import utils
from datetime import datetime
import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import models
from tqdm import tqdm
import os
import wandb

class Trainer():
    def __init__(self, args):
        # basic stuff
        self.config = utils.open_config(args.config)
        self.output_dir = args.output
        utils.init_expt(self.output_dir)
        
        # model
        if "simclr" in args.config:
            self.model = models.SimCLR(self.config)
        else:
            raise NotImplementedError()
        
        # get transforms
        train_transform = datasets.get_transform(self.config["train_transform"])
        valid_transform = datasets.get_transform(self.config["valid_transform"])
        aug_transform = datasets.get_transform(self.config["aug_transform"])
        
        # datasets and dataloaders
        simclr_dataset = datasets.get_dataset(config=self.config, split="train", transforms={"i": aug_transform, "j": aug_transform}, return_items={"i", "j"})
        sample_weights = datasets.sample_weights(simclr_dataset.targets)
        self.simclr_loader = DataLoader(simclr_dataset, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"], sampler=WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True), drop_last=True)

        cls_train_dataset = datasets.get_dataset(config=self.config, split="train", transforms={"train": train_transform}, return_items={"train", "target"})
        self.cls_train_loader = DataLoader(cls_train_dataset, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"], shuffle=True)

        cls_val_dataset = datasets.get_dataset(config=self.config, split="train", transforms={"val": valid_transform}, return_items={"val", "target"})
        sample_weights = datasets.sample_weights(cls_val_dataset.targets)
        self.cls_val_loader = DataLoader(cls_val_dataset, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"], shuffle=False)

    def train(self):
        wandb.init("scan")
        train_step = 0
        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch}/{self.config['epochs']}:\n")
            pbar = tqdm(total=len(self.simclr_loader))
            for data in self.simclr_loader:
                metrics = self.model.train_one_step(data)
                log = ""
                for key, value in metrics.items():
                    log += f"{key}: {round(value, 4)}"
                wandb.log({**metrics, "train_step": train_step})
                pbar.set_description(log)
                pbar.update(1)
                train_step += 1
            pbar.close()
            wandb.log({"lr": self.model.optim.param_groups[0]["lr"], "epoch": epoch})
            if epoch+1 <= self.model.warmup_epochs:
                self.model.optim.param_groups[0]["lr"] = (epoch+1)/self.model.warmup_epochs*self.model.lr
            else:
                self.model.scheduler.step()
            if epoch % self.config["valid_every"]==0 and epoch!=0:
                print(f"\nValidate Epoch {epoch}:\n")
                metrics, best = self.model.validate(self.cls_train_loader, self.cls_val_loader)
                wandb.log({**metrics, "val_epoch": epoch})
                if best:
                    print(f"model improved")
                    self.model.save(os.path.join(self.output_dir, "best.ckpt"))
            self.model.save(os.path.join(self.output_dir, "last.ckpt"))
        
if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="simclr training script")
    arg_parser.add_argument("-c", "--config", required=True, type=str, help="path to config file")
    arg_parser.add_argument("-o", "--output", default=datetime.now().strftime('%Y-%m-%d_%H-%M'), type=str, help="output directory")
    
    args = arg_parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()