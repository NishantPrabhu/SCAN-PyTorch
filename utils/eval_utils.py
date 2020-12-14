import faiss
import numpy as np
from scipy.optimize import linear_sum_assignment

def find_neighbors(fvecs, topk=20):
    # Mine neighbors
    index = faiss.IndexFlatIP(fvecs.shape[1])
    index.add(fvecs)
    _, indices = index.search(fvecs, topk+1)
    return indices

def compute_neighbour_acc(targets, neighbor_indices, topk=20):
    # Compute accuracy 
    anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
    neighbor_targets = np.take(targets, neighbor_indices[:, 1:], axis=0) # ignore itself
    accuracy = np.mean(anchor_targets == neighbor_targets)
    return accuracy

def hungarian_match(pred, targets, pred_k, targets_k):
    num_samples = targets.shape[0]
    # works only if num of clusters in prediction and targets are the same!
    assert pred_k == targets_k
    num_correct = np.zeros((pred_k, pred_k))

    for c1 in range(pred_k):
        for c2 in range(pred_k):
            votes = int(((pred == c1) * (targets == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    cls_map = [(out_c, gt_c) for out_c, gt_c in match]
    return cls_map
