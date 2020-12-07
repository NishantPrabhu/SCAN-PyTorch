
"""
LARS Optimizer definition

"""

import torch
import torch.optim as optim
from math import pi, cos


class LARS(torch.optim.Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0005, eta=0.001, max_epoch=100):
        assert lr >= 0.0, 'lr should be >= 0.0, got {}'.format(lr)
        assert max_epoch > 0, 'max_epoch should be > 0, got {}'.format(max_epoch)
        assert momentum >= 0.0, 'momentum should be >= 0.0, got {}'.format(momentum)
        assert weight_decay >= 0.0, 'weight_decay should be >= 0.0, got {}'.format(weight_decay)
        assert eta >= 0.0, 'eta should be >= 0.0, got {}'.format(eta)

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, max_epoch=max_epoch)
        super(LARS, self).__init__(params, defaults)


    def step(self, epoch=None, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch 
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            max_epoch = group['max_epoch']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                g, w = p.grad.data, p.data

                # Global and local learning rates 
                global_lr = lr * (1. - float(epoch)/max_epoch)**2
                local_lr = eta * w.norm() / (g.norm() + weight_decay * w.norm()) 
                actual_lr = global_lr * local_lr

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']

                buf.mul_(momentum).add_(actual_lr, g + weight_decay*w)
                p.data.add_(-buf)

        return loss


class SimCLRScheduler:

    def __init__(self, optimizer, warmup_epochs, total_epochs, start_lr, base_lr):
        
        assert warmup_epochs <= total_epochs, 'warmup_epochs must be <= total_epochs'
        assert start_lr <= base_lr, 'start_lr must be <= base_lr'
        
        self.opt = optimizer 
        self.warmup_epochs = warmup_epochs 
        self.total_epochs = total_epochs
        self.start_lr = start_lr 
        self.base_lr = base_lr 
        if self.warmup_epochs > 0:
            self.warmup_slope = (base_lr - start_lr)/self.warmup_epochs


    def step(self, epoch):

        if epoch <= self.warmup_epochs:
            lr = self.start_lr + epoch * self.warmup_slope
        else:
            lr = 0.5 * self.base_lr * (1 + cos(pi * (epoch - self.warmup_epochs)/float(self.total_epochs - self.warmup_epochs)))

        for group in self.opt.param_groups:
            group['lr'] = lr

        return lr


class CosineScheduler:

    def __init__(self, optimizer, total_epochs, base_lr):
        assert base_lr >= 0.0, 'base_lr should be >= 0.0'
        self.opt = optimizer 
        self.total_epochs = total_epochs 
        self.base_lr = base_lr 

    def step(self, epoch):
        lr = 0.5 * self.base_lr * (1 + cos(pi * epoch/float(self.total_epochs)))
        for group in self.opt.param_groups:
            group['lr'] = lr 

        return lr


def get_optimizer(config, params):
    
    name = config.pop('name')
    if name == 'lars':
        return LARS(params=params, **config)
    elif name == 'adam':
        return optim.Adam(params=params, **config)
    elif name == 'sgd':
        return optim.SGD(params=params, **config)


def get_lr_scheduler(config, optimizer):
    
    name = config.pop('name')
    if name == 'warmup_cosine':
        return SimCLRScheduler(optimizer=optimizer, **config)
    elif name == 'cosine':
        return CosineScheduler(optimizer=optimizer, **config)