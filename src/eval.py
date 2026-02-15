"""
src/eval.py — Evaluation metrics, bootstrap CI, statistical tests.

Metrics use corrected denominator: min(k, |I_u|) for hit rate.
"""
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional

try:
    import bottleneck as bn
    _USE_BN = True
except ImportError:
    _USE_BN = False
    print("Warning: bottleneck not installed, using numpy (slower)")


def _argpartition_topk(arr, k, axis=1):
    """Top-K argpartition using bottleneck or numpy."""
    if _USE_BN:
        return bn.argpartition(-arr, k, axis=axis)
    return np.argpartition(-arr, k, axis=axis)


def hit_rate_at_k(X_pred: np.ndarray, heldout: sp.csr_matrix, k: int) -> np.ndarray:
    """
    Per-user hit rate@k with correct denominator min(k, |I_u|).
    Returns array of per-user scores.
    """
    n = X_pred.shape[0]
    if k >= X_pred.shape[1]:
        k = X_pred.shape[1] - 1
    idx = _argpartition_topk(X_pred, k)
    pred_bin = np.zeros_like(X_pred, dtype=bool)
    pred_bin[np.arange(n)[:, np.newaxis], idx[:, :k]] = True
    true_bin = (heldout > 0).toarray() if sp.issparse(heldout) else (heldout > 0)
    hits = np.logical_and(true_bin, pred_bin).sum(axis=1).astype(np.float64)
    n_rel = true_bin.sum(axis=1).astype(np.float64)
    denom = np.minimum(k, n_rel)
    return hits / np.maximum(denom, 1.0)


def ndcg_at_k(X_pred: np.ndarray, heldout: sp.csr_matrix, k: int) -> np.ndarray:
    """
    Per-user NDCG@k for binary relevance.
    Returns array of per-user scores.
    """
    n = X_pred.shape[0]
    if k >= X_pred.shape[1]:
        k = X_pred.shape[1] - 1

    idx_part = _argpartition_topk(X_pred, k)
    topk_part = X_pred[np.arange(n)[:, np.newaxis], idx_part[:, :k]]
    idx_sort = np.argsort(-topk_part, axis=1)
    idx_topk = idx_part[np.arange(n)[:, np.newaxis], idx_sort]

    tp = 1.0 / np.log2(np.arange(2, k + 2))
    heldout_arr = heldout.toarray() if sp.issparse(heldout) else heldout
    DCG = (heldout_arr[np.arange(n)[:, np.newaxis], idx_topk] * tp).sum(axis=1)

    n_rel = (heldout_arr > 0).sum(axis=1)
    IDCG = np.array([tp[: min(int(nr), k)].sum() for nr in n_rel])
    return DCG / np.maximum(IDCG, 1e-10)


def evaluate(
    pred: np.ndarray,
    test_data: sp.csr_matrix,
    train_data: sp.csr_matrix,
    ks: List[int] = (10, 25),
    cold_items: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate predictions: mask train items AND warm items (for cold-start).
    Compute metrics on test users, ranking only cold items.
    
    Args:
        pred: (n_users, n_items) prediction matrix
        test_data: Sparse test matrix (only cold-item interactions)
        train_data: Sparse train matrix (only warm-item interactions)
        ks: List of k values for metrics
        cold_items: Optional array of cold item indices. If provided, masks all warm items.
    
    Returns dict of {metric_name: avg_score}.
    """
    pred = pred.copy()
    n_items = pred.shape[1]
    
    # Mask training interactions
    train_coo = train_data.tocoo()
    pred[train_coo.row, train_coo.col] = -np.inf

    # For cold-start evaluation: mask all warm items (non-cold items)
    if cold_items is not None:
        warm_items = np.setdiff1d(np.arange(n_items), cold_items)
        pred[:, warm_items] = -np.inf

    # Select users with test interactions
    test_users = np.where(np.asarray(test_data.sum(axis=1)).ravel() > 0)[0]
    if len(test_users) == 0:
        return {f"{m}@{k}": 0.0 for k in ks for m in ("hr", "ndcg")}

    p = pred[test_users]
    t = test_data[test_users]

    results = {}
    for k in ks:
        if k >= p.shape[1]:
            continue
        hr = hit_rate_at_k(p, t, k)
        nd = ndcg_at_k(p, t, k)
        results[f"hr@{k}"] = float(np.nanmean(hr))
        results[f"ndcg@{k}"] = float(np.nanmean(nd))
    return results


def bootstrap_ci(
    pred: np.ndarray,
    test_data: sp.csr_matrix,
    train_data: sp.csr_matrix,
    ks: List[int] = (10, 25),
    n_boot: int = 500,
    frac: float = 0.2,
    seed: int = 42,
    cold_items: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap 95% CI on metrics.
    Returns: {metric: (mean, lower, upper)}
    """
    rng = np.random.RandomState(seed)
    pred = pred.copy()
    n_items = pred.shape[1]
    
    train_coo = train_data.tocoo()
    pred[train_coo.row, train_coo.col] = -np.inf

    # For cold-start evaluation: mask all warm items
    if cold_items is not None:
        warm_items = np.setdiff1d(np.arange(n_items), cold_items)
        pred[:, warm_items] = -np.inf

    test_users = np.where(np.asarray(test_data.sum(axis=1)).ravel() > 0)[0]
    p = pred[test_users]
    t = test_data[test_users]
    n = len(test_users)
    ss = max(int(n * frac), 10)

    boot_metrics = {f"{m}@{k}": [] for k in ks for m in ("hr", "ndcg")}

    for _ in range(n_boot):
        idx = rng.choice(n, ss, replace=True)
        for k in ks:
            if k >= p.shape[1]:
                continue
            hr = np.nanmean(hit_rate_at_k(p[idx], t[idx], k))
            nd = np.nanmean(ndcg_at_k(p[idx], t[idx], k))
            boot_metrics[f"hr@{k}"].append(hr)
            boot_metrics[f"ndcg@{k}"].append(nd)

    result = {}
    for metric, vals in boot_metrics.items():
        if vals:
            result[metric] = (
                float(np.mean(vals)),
                float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5)),
            )
    return result


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_boot: int = 10000,
    seed: int = 42,
) -> float:
    """
    Paired bootstrap test: P(mean(A) > mean(B)).
    scores_a, scores_b: per-split metric values.
    Returns p-value (probability that A is NOT better than B).
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    n = len(diff)
    count = 0
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.mean(diff[idx]) <= 0:
            count += 1
    return count / n_boot


def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, float]:
    """Paired t-test on per-split scores. Returns (t_stat, p_value)."""
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)
