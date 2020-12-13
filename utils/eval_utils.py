import faiss
import numpy as np

def find_neighbours(fvecs, topk=20):
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