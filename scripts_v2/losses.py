
import torch
import torch.nn as nn 
import torch.nn.functional as F


def entropy(x, input_as_probabilities=True):

    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-10)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:
        return -b.sum()
    else:
        raise NotImplementedError('entropy input shape error')


class SCANLoss(torch.nn.Module):

    def __init__(self, entropy_weight=2):
        super(SCANLoss, self).__init__()
        self.ew = entropy_weight
        self.bce = nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, anchor_logits, neighbor_logits):
        b, n = anchor_logits.size()
        anchor_probs = self.softmax(anchor_logits)
        neighbor_probs = self.softmax(neighbor_logits)

        correlation = torch.bmm(anchor_probs.view(b, 1, n), neighbor_probs.view(b, n, 1)).squeeze()
        targets = torch.ones_like(correlation)
        consistency_loss = self.bce(correlation, targets)
        entropy_loss = entropy(torch.mean(anchor_probs, 0), True)
        loss = consistency_loss - self.ew * entropy_loss

        return loss, consistency_loss, entropy_loss


class MaskedCELoss(torch.nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()

    def forward(self, inputs, targets, mask, weight):
        if not (mask != 0).any():
            raise ValueError('Mask is all zeros')

        target_masked = torch.masked_select(targets, mask)
        b, c = inputs.size()
        n = target_masked.size(0)
        input_masked = torch.masked_select(inputs, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input_masked, target_masked, weight=weight)


class ConfidenceBasedCELoss(torch.nn.Module):

    def __init__(self, threshold=0.9, apply_class_balancing=True):
        super(ConfidenceBasedCELoss, self).__init__()
        self.threshold = threshold
        self.class_balance = apply_class_balancing
        self.softmax = nn.Softmax(dim=1)
        self.loss = MaskedCELoss()
    
    def forward(self, anchor_logits, augment_logits):
        anchor_probs = self.softmax(anchor_logits)
        max_prob, target = torch.max(anchor_probs, dim=1)
        mask = max_prob > self.threshold
        b, c = anchor_probs.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Class class balancing
        if self.class_balance:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1./(counts.float()/n)
            weight = torch.ones(c).to(anchor_logits.device)
            weight[idx] = freq
        else:
            weight = None

        loss = self.loss(augment_logits, target, mask, weight=weight)

        # Accuracy computation
        masked_augment_logits = torch.masked_select(augment_logits, mask.view(b, 1)).view(n, c)
        masked_augment_preds = self.softmax(masked_augment_logits).argmax(dim=-1)
        correct = masked_augment_preds.eq(target_masked.view_as(masked_augment_preds)).sum().item()
        acc = correct/n

        return loss, acc, target_masked



