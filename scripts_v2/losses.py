
"""
All loss functions in one place.

Author: Nishant Prabhu
"""

import torch
import torch.nn.functional as F


def entropy(x, input_as_probabilities=True):
    """
        Shannon entropy of a tensor x averaged
        along batch dimension.

        Args:
            x <torch.Tensor>
                Tensor of shape (batch_size, num_clusters)

        Returns:
            float : Shannon entropy
    """
    # Data type checks
    assert isinstance(x, torch.Tensor), f"x has to be torch.Tensor, got {type(x)}"

    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-10)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 1:
        return -b.sum()
    elif len(b.size()) == 2:
        return -b.sum(dim=1).mean()
    else:
        raise ValueError("Entropy loss shape error")


class SCANLoss(torch.nn.Module):

    def __init__(self, entropy_weight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = torch.nn.BCELoss()
        self.ew = entropy_weight

    def forward(self, anchor_logits, neighbor_logits):
        """
        anchor_logits: shape (batch_size, num_classes)
        neighbor_logits: shape (batch_size, num_classes)
        """
        b, n = anchor_logits.size()
        anchor_probs = self.softmax(anchor_logits)
        neighbor_probs = self.softmax(neighbor_logits)

        similarity = torch.bmm(anchor_probs.view(b, 1, n), neighbor_probs.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        entropy_loss = entropy(torch.mean(anchor_probs, 0), True)
        total_loss = consistency_loss - self.ew * entropy_loss

        return total_loss, consistency_loss, entropy_loss


class MaskedCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, inp, trg, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError("Mask is all zeros")

        trg = torch.masked_select(trg, mask)
        b, c = inp.size()
        n = trg.size(0)
        inp = torch.masked_select(inp, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(inp, trg, weight=weight, reduction=reduction)


class ConfidenceBasedCE(torch.nn.Module):

    def __init__(self, threshold=0.9, apply_class_balancing=True):
        super(ConfidenceBasedCE, self).__init__()
        self.thresh = threshold
        self.softmax = torch.nn.Softmax(dim=1)
        self.class_balance = apply_class_balancing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = MaskedCrossEntropyLoss()

    def forward(self, anchor_logits, augment_logits):
        """
        Computes loss only for logits whose probability is more than threshold
        """
        anchor_probs = self.softmax(anchor_logits)
        max_prob, target = torch.max(anchor_probs, dim=1)
        mask = max_prob > self.thresh
        b, c = anchor_probs.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Class class balancing
        if self.class_balance:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1./(counts.float()/n)
            weight = torch.ones(c).to(self.device)
            weight[idx] = freq
        else:
            weight = None

        loss = self.loss(augment_logits, target, mask, weight=weight, reduction='mean')

        # Accuracy computation
        masked_augment_logits = torch.masked_select(augment_logits, mask.view(b, 1)).view(n, c)
        masked_augment_preds = self.softmax(masked_augment_logits).argmax(dim=-1)
        correct = masked_augment_preds.eq(target_masked.view_as(masked_augment_preds)).sum().item()
        acc = correct/n

        return loss, acc
