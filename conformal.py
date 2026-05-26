import math
import torch
import numpy as np


def softmax_nc(raw_scores: np.ndarray) -> np.ndarray:
    """1 - softmax(raw_scores)."""
    if raw_scores.size == 0:
        return np.array([], dtype=np.float32)
    probs = torch.softmax(torch.from_numpy(raw_scores.astype(np.float32)), dim=0).numpy()
    return 1.0 - probs


def minmax_nc(raw_scores: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    """1 - minmax_norm(raw_scores)."""
    if raw_scores.size == 0:
        return np.array([], dtype=np.float32)
    s_min = raw_scores.min()
    s_max = raw_scores.max()
    norm = (raw_scores - s_min) / (s_max - s_min + delta)
    return 1.0 - norm


def margin_nc(raw_scores: np.ndarray) -> np.ndarray:
    """max(raw_scores) - raw_scores  (equivalent to max_{others} - score)."""
    if raw_scores.size == 0:
        return np.array([], dtype=np.float32)
    return raw_scores.max() - raw_scores


def conformal_threshold(nc_scores: list, epsilon: float) -> float:
    """
    nc_scores: list of pooled calibration nonconformity scores
    epsilon:   desired error rate
    Returns q_hat = the k-th smallest score, k = ceil((n+1)*(1-epsilon)).
    If k > n, return +inf (keep all candidates).
    """
    n = len(nc_scores)
    if n == 0:
        return float('inf')
    sorted_scores = np.sort(nc_scores)
    k = math.ceil((n + 1) * (1.0 - epsilon))
    if k > n:
        return float('inf')
    # k is 1-indexed; convert to 0-indexed
    return float(sorted_scores[k - 1])
