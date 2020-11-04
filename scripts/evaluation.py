
"""
Loss functions and other evaluation helpers

@author: Nishant Prabhu
"""

# Dependencies
import torch
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def entropy_loss(x):
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

    x_ = torch.clamp(x, min=1e-10)
    b = x_ * torch.log(x)

    if len(b.size()) == 1:
        return -b.sum()
    elif len(b.size()) == 2:
        return -b.sum(dim=1).mean()
    else:
        raise ValueError("Entropy loss shape error")


def similarity_loss(image_probs, neighbors_probs):
    """
        Similarity score for tensor of output probabilities.

        Args:
            image_probs <torch.Tensor>
                Output tensor for original images of shape (batch_size, num_clusters)

            neighbors_probs <torch.Tensor>
                Output tensor for neighbor images of shape (batch_size, num_clusters)

        Returns:
            torch.Tensor : Tensor of shape (batch_size, batch_size)
    """
    # Data type checks
    assert isinstance(image_probs, torch.Tensor), f"image_probs should be torch.Tensor, got {type(image_probs)}"
    assert isinstance(neighbors_probs, torch.Tensor), f"neighbors_probs should be torch.Tensor, got {type(neighbors_probs)}"

    # Compute similarity score of each image output with every neighbor output
    b, n = image_probs.size()
    sim_scores = torch.bmm(image_probs.view(b, 1, n), neighbors_probs.view(b, n, 1)).squeeze()

    # Compute binary cross entropy loss
    ones = torch.ones_like(sim_scores)
    loss = F.binary_cross_entropy(sim_scores, ones)

    return loss


def get_confusion_matrix(predictions, targets, class_names=None, title='', name='confusion_matrix.jpeg'):
    """
    Confusion matrix for given predictions and targets using seaborn.
    Saved as a JPEG file at save_path

    """
    # Generate confusion matrix and normalize to get ratios
    cm = metrics.confusion_matrix(predictions, targets)
    cm = cm / cm.sum(axis=0)

    # Figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    sns.heatmap(cm, square=True, annot=True, cmap='Reds')

    if class_names is not None:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    if title is not None:
        ax.set_title(title)

    plt.savefig('../saved_data/plots/' + name)


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None, confusion_matrix=True,
                       confusion_matrix_kwargs={'title': '', 'name': 'confusion_matrix.jpeg'}):

    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    targets = torch.cat([all_predictions['labels']]*2, dim=0).cpu()
    predictions = all_predictions['preds'][subhead_index].cpu()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    label_map = _hungarian_match(predictions, targets)
    reset_preds = np.array([label_map[c.item()] for c in predictions])

    # Gather performance metrics
    acc = (reset_preds == targets.numpy()).mean()
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())

    # Compute confusion matrix
    if compute_confusion_matrix:
        get_confusion_matrix(reset_preds, targets.numpy(), class_names, **confusion_matrix_kwargs)

    return {'accuracy': acc, 'ARI': ari, 'NMI': nmi, 'hungarian_match': label_map}


@torch.no_grad()
def find_unique(targets, preds):
    targets_uniq, preds_uniq = [], []

    for t, p in zip(targets, preds):
        if t not in targets_uniq:
            targets_uniq.append(t)
        if p not in preds_uniq:
            preds_uniq.append(p)

    return targets_uniq, preds_uniq


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets):
    """
    flat_preds   -> (2*batch_size,)
    flat_targets -> (2*batch_size,)

    """
    num_samples = flat_targets.shape[0]
    t_uniq, p_uniq = find_unique(flat_targets, flat_preds)
    targets_uniq, preds_uniq = torch.unique(flat_targets), torch.unique(flat_preds)
    num_k = max(targets_uniq.numel(), preds_uniq.numel())
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(len(preds_uniq)):
        for c2 in range(len(targets_uniq)):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == preds_uniq[c1]) * (flat_targets == targets_uniq[c2])).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(-num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    # mapping
    label_map = {preds_uniq[min(i, len(preds_uniq)-1)]: targets_uniq[j] for i, j in res}
    label_map = {j.item(): i.item() for i, j in label_map.items()}

    return label_map
