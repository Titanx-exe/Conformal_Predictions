# EvidenceSmoothingLoss.py
import torch
import torch.nn.functional as F

# ------- EBLS helpers (functional) -------
def _pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=1)
    return x @ x.T

def _topk_mask(sim: torch.Tensor, k: int) -> torch.Tensor:
    """
    sim: [N, N] (larger = more similar). Diagonal is ignored.
    returns: boolean mask [N, N] where True means j is in top-k of i.
    """
    N = sim.size(0)
    if N <= 1:
        return torch.zeros_like(sim, dtype=torch.bool)

    k_eff = max(1, min(k, N - 1))         # clamp k to [1, N-1]
    sim = sim.clone()
    sim.fill_diagonal_(-1e9)              # exclude self
    topk_idx = sim.topk(k_eff, dim=1).indices  # [N, k_eff]
    mask = torch.zeros_like(sim, dtype=torch.bool)
    row_idx = torch.arange(N, device=sim.device).unsqueeze(1).expand_as(topk_idx)
    mask[row_idx, topk_idx] = True
    return mask

def _reciprocal_adj(knn_mask: torch.Tensor) -> torch.Tensor:
    """R[i,j] = True iff j in top-k(i) AND i in top-k(j)."""
    return knn_mask & knn_mask.T

def _jaccard_from_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    adj: [N, N] boolean (e.g., reciprocal adjacency).
    returns Jaccard(i,j) over neighbor sets (excluding self).
    """
    if adj.numel() == 0:
        return adj.float()
    A = adj.float()
    inter = A @ A.T                      # [N, N]
    deg = A.sum(dim=1, keepdim=True)     # [N,1]
    union = deg + deg.T - inter          # [N, N]
    jacc = inter / (union + 1e-8)
    jacc.fill_diagonal_(0.0)
    return jacc

def _ebls_targets_from_docs(
    cand_embs: torch.Tensor,             # [B, D] candidate embeddings (d_i is GT for query i)
    eps: float = 0.1,
    k:   int   = 8,
    lam: float = 0.5,                    # mix: lam * geometry + (1-lam) * rNN-Jaccard
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Build smoothed target distribution Y [B, B] per query i:
      y[i,i] = 1 - eps
      y[i,j!=i] = eps * w_ij / sum_j w_ij
    where w_ij is evidence from (geo + rNN-Jaccard) between GT doc d_i and candidate d_j.
    """
    B = cand_embs.size(0)
    device = cand_embs.device

    if B == 0:
        return torch.empty(0, 0, device=device)
    if B == 1:
        return torch.tensor([[1.0]], device=device)

    # 1) Document–document similarity
    if metric == "cosine":
        sim_dd = _pairwise_cosine(cand_embs)     # [B, B] in [-1, 1]
    else:
        sim_dd = cand_embs @ cand_embs.T         # dot product

    geo = sim_dd.clamp_min(0.0)
    geo.fill_diagonal_(0.0)

    # 2) k-NN and k-reciprocal adjacency
    knn_mask = _topk_mask(sim_dd, k=k)           # [B, B] bool
    rnn_adj  = _reciprocal_adj(knn_mask)         # [B, B] bool

    # 3) rNN Jaccard similarity
    rnn_jacc = _jaccard_from_adj(rnn_adj)        # [B, B] in [0,1], diag=0

    # 4) blended evidence
    e = lam * geo + (1.0 - lam) * rnn_jacc       # [B, B]
    e.fill_diagonal_(0.0)

    # Row-normalize evidence over negatives; fallback to uniform if row is all zeros
    e_sum = e.sum(dim=1, keepdim=True)           # [B,1]
    uniform_neg = torch.full_like(e, 1.0 / (B - 1))
    uniform_neg.fill_diagonal_(0.0)
    w = torch.where(e_sum > 0, e / (e_sum + 1e-12), uniform_neg)   # [B,B]
    w.fill_diagonal_(0.0)

    # 5) build targets
    Y = w * eps
    diag = torch.arange(B, device=device)
    Y[diag, diag] = 1.0 - eps
    return Y                                     # [B, B]

def ebls_loss(
    scores: torch.Tensor,            # [B, B] query–candidate logits
    cand_embs: torch.Tensor,         # [B, D] candidate embeddings
    eps: float = 0.1,
    k:   int   = 8,
    lam: float = 0.5,
    metric: str = "cosine",
    detach_evidence: bool = True
):
    """
    Compute EBLS loss: KL( log_softmax(scores) || Y_ebls ),
    where Y_ebls uses k-reciprocal neighborhoods among documents.
    """
    # ensure same device
    device = scores.device
    cand_embs = cand_embs.to(device)

    with torch.no_grad():
        emb_src = cand_embs.detach() if detach_evidence else cand_embs
        Y = _ebls_targets_from_docs(
            cand_embs=emb_src, eps=eps, k=k, lam=lam, metric=metric
        )  # [B, B]

    logp = F.log_softmax(scores, dim=1)          # [B, B]
    loss = F.kl_div(logp, Y, reduction="batchmean")
    return loss, Y




# Reuse your helpers:
# _pairwise_cosine, _topk_mask, _reciprocal_adj, _jaccard_from_adj


import torch
import torch.nn.functional as F

@torch.no_grad()
def _wsls_y_from_candidates_rowwise(
    cand_embs: torch.Tensor,   # [C, D]  (all candidates in this batch)
    pos_indices: torch.Tensor, # [Q]     (gold col index per query)
    eps: float = 0.1,
    k:   int   = 8,
    lam: float = 0.5,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Build row-wise soft targets for Q queries over the same C candidates.
    For each query i with gold index g = pos_indices[i]:
      y_i[g]     = 1 - eps
      y_i[j!=g]  = eps * w_gj / sum_{j!=g} w_gj
    where w_gj is blended evidence from geo + rNN-Jaccard in the CxC candidate graph.
    Returns Y: [Q, C].
    """
    device = cand_embs.device
    C = cand_embs.size(0)
    Q = pos_indices.size(0)

    if C == 1:
        Y = torch.zeros(Q, 1, device=device)
        Y[:, 0] = 1.0
        return Y

    # candidate-candidate similarity (shared for all queries)
    if metric == "cosine":
        sim = _pairwise_cosine(cand_embs)    # [C, C] in [-1,1]
    else:
        sim = cand_embs @ cand_embs.T
    geo = sim.clamp_min(0.0)
    geo.fill_diagonal_(0.0)

    k_eff = max(1, min(k, C - 1))
    knn_mask = _topk_mask(sim, k=k_eff)      # [C, C] bool
    rnn_adj  = _reciprocal_adj(knn_mask)
    rnn_jacc = _jaccard_from_adj(rnn_adj)    # [C, C], diag=0

    e = lam * geo + (1.0 - lam) * rnn_jacc   # [C, C]
    e.fill_diagonal_(0.0)

    # For each query, pick the row corresponding to its gold doc g
    Y = torch.zeros(Q, C, device=device)
    ar = torch.arange(Q, device=device)
    g = pos_indices.clamp(0, C - 1)

    w_g = e[g, :]                       # [Q, C]
    # set self position to 0 so we normalize over negatives only
    w_g[ar, g] = 0.0

    w_sum = w_g.sum(dim=1, keepdim=True)          # [Q, 1]
    # fallback: uniform over negatives if w_sum == 0
    uniform_neg = torch.full_like(w_g, 1.0 / (C - 1))
    uniform_neg[ar, g] = 0.0
    w_norm = torch.where(w_sum > 0, w_g / (w_sum + 1e-12), uniform_neg)

    # final targets
    Y = eps * w_norm
    Y[ar, g] = 1.0 - eps
    return Y


def ebls_loss_rows(
    scores: torch.Tensor,        # [Q, C] (queries × batch-candidates)
    cand_embs: torch.Tensor,     # [C, D] (same candidate set used for all Q rows)
    target: torch.Tensor,        # [Q]    (gold column index per row)
    eps: float = 0.1,
    k:   int   = 8,
    lam: float = 0.5,
    metric: str = "cosine",
    detach_evidence: bool = True,
):
    """
    EBLS for rectangular logits: KL( Y || softmax(scores) ), where Y is built per row
    using the candidate graph and the row's gold column (target).
    """
    Q, C = scores.shape
    assert cand_embs.size(0) == C, f"Cand embs C={cand_embs.size(0)} must match logits C={C}"
    device = scores.device

    emb_src = cand_embs.detach() if detach_evidence else cand_embs
    with torch.no_grad():
        Y = _wsls_y_from_candidates_rowwise(
            emb_src, pos_indices=target.to(device), eps=eps, k=k, lam=lam, metric=metric
        )  # [Q, C]

    logp = F.log_softmax(scores, dim=1)  # [Q, C]
    loss = F.kl_div(logp, Y, reduction="batchmean")
    return loss, Y


