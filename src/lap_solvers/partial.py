import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def partial_assignment(similarity, threshold):
    if similarity.ndim != 2 or not np.isfinite(threshold):
        raise ValueError('Expected a 2D similarity matrix and finite threshold')
    sim = similarity.detach().cpu().numpy()
    if not np.isfinite(sim).all():
        raise ValueError('Similarities must be finite')
    n, m = sim.shape
    result = np.zeros((n, m), dtype=np.float32)
    if n == 0 or m == 0:
        return torch.from_numpy(result).to(similarity.device)
    utility = np.zeros((n + m, n + m), dtype=np.float64)
    utility[:n, :m] = sim.astype(np.float64) - threshold
    row, col = linear_sum_assignment(-utility)
    keep = (row < n) & (col < m) & (sim[np.minimum(row, n - 1), np.minimum(col, m - 1)] > threshold)
    result[row[keep], col[keep]] = 1
    return torch.from_numpy(result).to(similarity.device)
