
"""
Loss functions for SimCLR and SCAN.

Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies 
import torch 
import torch.nn as nn 
import torch.nn.functional as F


class SupervisedLoss(nn.Module):

    def __init__(self):
        super(SupervisedLoss, self).__init__()
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        loss = F.nll_loss(log_prob, target)
        return loss


class SimclrLoss(nn.Module):

    def __init__(self, normalize=False, temperature=1.0):
        super(SimclrLoss, self).__init__()
        self.normalize = normalize
        self.temperature = temperature
        
    def forward(self, zi, zj):
        bs = zi.shape[0]
        labels = torch.zeros(2*bs).long().to(zi.device)
        mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)
        
        if self.normalize:
            zi_norm = F.normalize(zi, p=2, dim=-1)
            zj_norm = F.normalize(zj, p=2, dim=-1)
        else:
            zi_norm = zi
            zj_norm = zj

        logits_ii = torch.mm(zi_norm, zi_norm.t()) / self.temperature
        logits_ij = torch.mm(zi_norm, zj_norm.t()) / self.temperature
        logits_ji = torch.mm(zj_norm, zi_norm.t()) / self.temperature
        logits_jj = torch.mm(zj_norm, zj_norm.t()) / self.temperature

        # Positive logits
        logits_ij_pos = logits_ij[torch.logical_not(mask)]                      # Shape (N,)
        logits_ji_pos = logits_ji[torch.logical_not(mask)]                      # Shape (N,)

        # Negative logits
        logits_ii_neg = logits_ii[mask].reshape(bs, -1)                         # Shape (N, N-1)
        logits_ij_neg = logits_ij[mask].reshape(bs, -1)                         # Shape (N, N-1)
        logits_ji_neg = logits_ji[mask].reshape(bs, -1)                         # Shape (N, N-1)
        logits_jj_neg = logits_jj[mask].reshape(bs, -1)                         # Shape (N, N-1)

        # Combine
        pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)     # Shape (2N, 1)
        neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)                # Shape (N, 2N-2)
        neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)                # Shape (N, 2N-2)
        neg = torch.cat((neg_i, neg_j), dim=0)                                  # Shape (2N, 2N-2)  
        
        logits = torch.cat((pos, neg), dim=1)                                   # Shape (2N, 2N-1)
        loss = F.cross_entropy(logits, labels)
        return loss


class ClusterLoss(nn.Module):

    def __init__(self, entropy_weight=2.0):
        super(ClusterLoss, self).__init__()
        self.entropy_weight = entropy_weight

    def forward(self, anchor_logits, neighbor_logits):
        # Compute probabilities
        b, n = anchor_logits.size()
        anchor_probs = F.softmax(anchor_logits, dim=1)
        neighbor_probs = F.softmax(neighbor_logits, dim=1)

        # Consistency loss
        similarity = torch.bmm(anchor_probs.view(b, 1, n), neighbor_probs.view(b, n, 1)).squeeze()
        consistency_loss = F.binary_cross_entropy(similarity, torch.ones_like(similarity))

        # Entropy
        p = torch.clamp(torch.mean(anchor_probs, dim=0), min=1e-10)
        entropy_loss = -(p * torch.log(p)).sum()

        # Combine loss
        loss = consistency_loss - self.entropy_weight * entropy_loss
        return loss, consistency_loss, entropy_loss


class SelflabelLoss(nn.Module):

    def __init__(self, confidence=0.99):
        super(SelflabelLoss, self).__init__()
        self.confidence = confidence

    def forward(self, anchor_logits, aug_logits):
        bs, n_cls = anchor_probs.size()
        # compute probability
        anchor_probs = F.softmax(anchor_logits, dim=1)
        
        # find predicted cls and mask based on confidence
        max_prob, target = torch.max(anchor_probs, dim=1)
        mask = max_prob > self.confidence
        target_masked = torch.masked_select(target, mask)
        input_masked = torch.masked_select(aug_logits, mask.view(bs, 1)).view(target_masked.size(0), n_cls)

        # find class weights in batch
        indx, counts = torch.unique(target_masked, return_counts=True)
        freq = target_masked.size(0)/counts.float()
        weight = torch.ones(n_cls).to(anchor_probs.device)
        weight[indx] = freq

        # compute loss
        loss = F.cross_entropy(input_masked, target_masked, weight=weight, reduction='mean')
        return loss

