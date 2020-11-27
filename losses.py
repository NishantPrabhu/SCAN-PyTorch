import math
import torch
from torch import nn
import torch.nn.functional as F

"""
Simclr training criterion
find similarity between i and j
positive logits will along the main diagonals lying in ij and ji.
negative logits will be all elements except the ones along the diagonals in ii, jj, ij, ji
then compute cross entropy loss
"""

class SimclrCriterion(nn.Module):
    def __init__(self, batch_size, normalize=True, temperature=1.0):
        super(SimclrCriterion, self).__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.batch_size = batch_size
        self.register_buffer('labels', torch.zeros(batch_size * 2).long())
        self.register_buffer('mask', torch.ones((batch_size, batch_size), dtype=bool).fill_diagonal_(0))

    def forward(self, z_i, z_j):
        if self.normalize:
            z_i_norm = F.normalize(z_i, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)
        else:
            z_i_norm = z_i
            z_j_norm = z_j

        # Cosine similarity between i and j
        logits_ii = torch.mm(z_i_norm, z_i_norm.t()) / self.temperature
        logits_jj = torch.mm(z_j_norm, z_j_norm.t()) / self.temperature
        logits_ij = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        logits_ji = torch.mm(z_j_norm, z_i_norm.t()) / self.temperature

        # Compute Postive Logits
        logits_ij_pos = logits_ij[torch.logical_not(self.mask)]
        logits_ji_pos = logits_ji[torch.logical_not(self.mask)]

        # Compute Negative Logits
        logit_ii_neg = logits_ii[self.mask].reshape(self.batch_size, -1)
        logit_jj_neg = logits_jj[self.mask].reshape(self.batch_size, -1)
        logit_ij_neg = logits_ij[self.mask].reshape(self.batch_size, -1)
        logit_ji_neg = logits_ji[self.mask].reshape(self.batch_size, -1)

        # Postive Logits over all samples
        pos = torch.cat((logits_ij_pos, logits_ji_pos)).unsqueeze(1)

        # Negative Logits over all samples
        neg_a = torch.cat((logit_ii_neg, logit_ij_neg), dim=1)
        neg_b = torch.cat((logit_ji_neg, logit_jj_neg), dim=1)
        neg = torch.cat((neg_a, neg_b), dim=0)

        # Compute cross entropy
        logits = torch.cat((pos, neg), dim=1)

        loss = F.cross_entropy(logits, self.labels.to(logits.device))
        return loss