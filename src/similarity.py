"""
src/similarity.py — Item-item similarity computation, TopK, cross-features.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict, List, Optional, Tuple


def smoothed_cosine_similarity(
    enc: np.ndarray,
    shrinkage: float = 20.0,
) -> np.ndarray:
    """
    Compute smoothed cosine similarity (Eq.3 in MARec paper).
    G_ij = (z_i · z_j) / (||z_i|| * ||z_j|| + δ)
    """
    sim_num = enc @ enc.T
    norms = np.linalg.norm(enc, axis=1)
    sim_den = np.outer(norms, norms) + shrinkage
    sim = sim_num / sim_den
    np.fill_diagonal(sim, 0.0)
    return sim


def year_similarity(enc: np.ndarray) -> np.ndarray:
    """1 - normalized Euclidean distance for year features."""
    dist = euclidean_distances(enc)
    max_dist = dist.max()
    sim = 1.0 - dist / (max_dist + 1e-10) if max_dist > 0 else np.zeros_like(dist)
    np.fill_diagonal(sim, 0.0)
    return sim


def topk_sparsify(sim: np.ndarray, k: int = 100) -> np.ndarray:
    """
    Keep only top-K neighbors per item. Symmetrize the result.
    Returns dense matrix (still needed for EASE alignment).
    """
    n = sim.shape[0]
    if k >= n:
        return sim.copy()
    sim_sparse = np.zeros_like(sim)
    for i in range(n):
        topk_idx = np.argpartition(-sim[i], k)[:k]
        sim_sparse[i, topk_idx] = sim[i, topk_idx]
    # Symmetrize: keep max of (i→j, j→i)
    return np.maximum(sim_sparse, sim_sparse.T)


def hadamard_cross(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Element-wise product of two similarity matrices (Eq.5 in paper)."""
    return s1 * s2


def build_similarity_matrices(
    encoded_features: Dict[str, np.ndarray],
    shrinkage: float = 20.0,
    per_feature_shrinkage: Optional[Dict[str, float]] = None,
    cross_pairs: Optional[List[Tuple[str, str]]] = None,
    topk: Optional[int] = None,
    topk_features: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build all similarity matrices from encoded features.

    Args:
        encoded_features: {name: ndarray} from features.py
        shrinkage: Default δ for smoothed cosine.
        per_feature_shrinkage: Override δ per feature, e.g. {"actors": 10}.
        cross_pairs: List of (feat_a, feat_b) for Hadamard products.
        topk: If set, also create TopK-sparsified versions.
        topk_features: Which features to create TopK versions for.

    Returns: {name: similarity_matrix}
    """
    S = {}

    for name, enc in encoded_features.items():
        if name == "years":
            S[name] = year_similarity(enc)
        else:
            delta = (per_feature_shrinkage or {}).get(name, shrinkage)
            S[name] = smoothed_cosine_similarity(enc, shrinkage=delta)

    # Cross-feature similarities (Hadamard products)
    if cross_pairs:
        for a, b in cross_pairs:
            if a in S and b in S:
                cross_name = f"{a}_x_{b}"
                S[cross_name] = hadamard_cross(S[a], S[b])

    # TopK sparsified versions
    if topk and topk_features:
        for name in topk_features:
            if name in S:
                S[f"{name}_topk{topk}"] = topk_sparsify(S[name], k=topk)

    return S
