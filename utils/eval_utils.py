import faiss
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import torch


def find_neighbors(fvecs, topk=20):
    # Mine neighbors
    index = faiss.IndexFlatIP(fvecs.shape[1])
    index.add(fvecs)
    _, indices = index.search(fvecs, topk + 1)
    return indices


def compute_neighbour_acc(targets, neighbor_indices, topk=20):
    # Compute accuracy
    anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
    neighbor_targets = np.take(targets, neighbor_indices[:, 1:], axis=0)  # ignore itself
    accuracy = np.mean(anchor_targets == neighbor_targets)
    return accuracy


def hungarian_match(pred, targets, pred_k, targets_k):
    num_samples = targets.shape[0]
    num_correct = np.zeros((pred_k, pred_k))

    for c1 in range(pred_k):
        for c2 in range(pred_k):
            votes = int(((pred == c1) * (targets == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    cls_map = [(out_c, gt_c) for out_c, gt_c in match]
    return cls_map


def eval_clusters(probs, targets):
    pred = probs.argmax(dim=1)
    _, pred_t5 = probs.topk(5, 1, largest=True)

    cls_map = hungarian_match(pred, targets, len(torch.unique(pred)), len(torch.unique(targets)))

    remapped_pred = torch.zeros_like(pred)
    remapped_pred_t5 = torch.zeros_like(pred_t5)
    for pred_c, target_c in cls_map:
        remapped_pred[pred == int(pred_c)] = int(target_c)
        remapped_pred_t5[pred_t5 == int(pred_c)] = int(target_c)

    cluster_score = {}
    cluster_score["acc"] = int((remapped_pred == targets).sum()) / len(probs)
    cluster_score["acc top5"] = int(remapped_pred_t5.eq(targets.view(-1, 1).expand_as(remapped_pred_t5)).sum()) / len(
        probs
    )
    cluster_score["nmi"] = metrics.normalized_mutual_info_score(targets, pred)
    cluster_score["ari"] = metrics.adjusted_rand_score(targets, pred)

    for i in torch.unique(remapped_pred):
        indx = remapped_pred == i
        cluster_score[f"cls {i} acc"] = int((remapped_pred[indx] == targets[indx]).sum()) / len(remapped_pred[indx])

    return cluster_score
