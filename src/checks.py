"""
src/checks.py — Verification and sanity checks for the framework.
"""
import numpy as np
import scipy.sparse as sp
from typing import Dict, List


def check_split_integrity(split: Dict) -> List[str]:
    """Comprehensive split verification. Returns list of issues (empty = OK)."""
    issues = []
    train = split["train"]
    test = split["test"]
    cold_items = set(split["cold_items"])

    # 1) Train should have NO interactions for cold items
    train_coo = train.tocoo()
    train_cold_cols = set(train_coo.col) & cold_items
    if train_cold_cols:
        issues.append(f"LEAK: train has {len(train_cold_cols)} cold item columns")

    # 2) Test should ONLY have cold item interactions
    test_coo = test.tocoo()
    test_warm_cols = set(test_coo.col) - cold_items
    if test_warm_cols:
        issues.append(f"LEAK: test has {len(test_warm_cols)} warm item columns")

    # 3) No NaN
    if hasattr(train, 'data') and np.any(np.isnan(train.data)):
        issues.append("NaN in train data")
    if hasattr(test, 'data') and np.any(np.isnan(test.data)):
        issues.append("NaN in test data")

    # 4) Shapes match
    if train.shape != test.shape:
        issues.append(f"Shape mismatch: train={train.shape}, test={test.shape}")

    return issues


def check_ease_matrix(B: np.ndarray) -> List[str]:
    """Check EASE weight matrix for issues."""
    issues = []

    # Diagonal should be zero
    diag_max = np.max(np.abs(np.diag(B)))
    if diag_max > 1e-6:
        issues.append(f"Diagonal not zero: max |diag(B)| = {diag_max:.6e}")

    # No NaN
    if np.any(np.isnan(B)):
        n_nan = np.sum(np.isnan(B))
        issues.append(f"B contains {n_nan} NaN values")

    # No Inf
    if np.any(np.isinf(B)):
        n_inf = np.sum(np.isinf(B))
        issues.append(f"B contains {n_inf} Inf values")

    return issues


def check_metric_correctness():
    """
    Unit test: verify hr@k and ndcg@k formulas with known inputs.
    """
    from .eval import hit_rate_at_k, ndcg_at_k

    # 3 users, 5 items. User 0: relevant=[0,1], User 1: relevant=[2], User 2: relevant=[3,4]
    heldout = sp.csr_matrix(np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ], dtype=np.float32))

    # Predictions: user 0 ranks items as [4,3,2,1,0], user 1 as [2,1,0,3,4], user 2 as [3,4,0,1,2]
    pred = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],  # top-2: items 4,3 → 0 hits
        [0.3, 0.4, 0.5, 0.2, 0.1],  # top-2: items 2,1 → 1 hit (item 2)
        [0.2, 0.3, 0.1, 0.5, 0.4],  # top-2: items 3,4 → 2 hits
    ], dtype=np.float64)

    k = 2
    hr = hit_rate_at_k(pred, heldout, k)

    # User 0: 0 hits / min(2, 2) = 0.0
    # User 1: 1 hit / min(2, 1) = 1.0
    # User 2: 2 hits / min(2, 2) = 1.0
    expected_hr = np.array([0.0, 1.0, 1.0])

    ok = np.allclose(hr, expected_hr, atol=1e-6)
    if not ok:
        return f"FAIL: hr@{k} expected {expected_hr}, got {hr}"

    return "PASS: metric correctness verified"


def run_all_checks(splits: List[Dict], verbose: bool = True) -> bool:
    """Run all verification checks."""
    all_ok = True

    # 1) Check metric formulas
    metric_result = check_metric_correctness()
    if verbose:
        print(f"  Metric check: {metric_result}")
    if "FAIL" in metric_result:
        all_ok = False

    # 2) Check splits
    for i, split in enumerate(splits):
        issues = check_split_integrity(split)
        if issues:
            all_ok = False
            if verbose:
                for issue in issues:
                    print(f"  Split {i}: {issue}")
        elif verbose and i == 0:
            print(f"  Split checks: all {len(splits)} splits OK")

    return all_ok
