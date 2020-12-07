
import torch 
import torch.nn as nn 
import torch.nn.functional as F


class SimCLRLoss(nn.Module):

    def __init__(self, batch_size, normalize=True, temperature=1.0):
        super(SimCLRLoss, self).__init__()
        self.normalize = normalize
        self.temp = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        labels = torch.zeros(2*batch_size).long()
        mask = torch.ones((batch_size, batch_size), dtype=bool).fill_diagonal_(0)

        if self.normalize:
            zi_norm = F.normalize(z_i, p=2, dim=-1)
            zj_norm = F.normalize(z_j, p=2, dim=-1)
        else:
            zi_norm = z_i
            zj_norm = z_j

        logits_ii = torch.mm(zi_norm, zi_norm.t()) / self.temp
        logits_jj = torch.mm(zj_norm, zj_norm.t()) / self.temp 
        logits_ij = torch.mm(zi_norm, zj_norm.t()) / self.temp 
        logits_ji = torch.mm(zj_norm, zi_norm.t()) / self.temp

        # Positive and negative samples 
        logits_ij_pos = logits_ij[torch.logical_not(mask)]                 # (N,)
        logits_ji_pos = logits_ji[torch.logical_not(mask)]                 # (N,)

        logits_ii_neg = logits_ii[mask].reshape(batch_size, -1)       # (N, N-1)
        logits_ij_neg = logits_ij[mask].reshape(batch_size, -1)       # (N, N-1)
        logits_ji_neg = logits_ij[mask].reshape(batch_size, -1)       # (N, N-1)
        logits_jj_neg = logits_jj[mask].reshape(batch_size, -1)       # (N, N-1)

        pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)     # (2N, 1) 
        neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)                # (N, 2N-2)
        neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)                # (N, 2N-2)
        neg = torch.cat((neg_i, neg_j), dim=0)                                  # (2N, 2N-2)

        out = torch.cat((pos, neg), dim=1)                                      # (2N, 2N-1)
        loss = F.cross_entropy(out, labels.to(out.device))
        return loss

