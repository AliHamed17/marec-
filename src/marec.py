"""
src/marec.py — MARec core: EASE backbone + metadata alignment.

Supports:
  - Unconstrained linear regression (original)
  - Non-negative least squares (NNLS) for µ weights
  - L2-regularized weight learning
"""
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import nnls
from typing import List, Tuple, Optional, Dict


def learn_weights_regression(
    X: np.ndarray,
    sims: List[np.ndarray],
    fit_intercept: bool = True,
    pos_weight: float = 1.0,
    neg_weight: float = 0.2,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Original MARec alignment weight learning via weighted linear regression.
    Predicts X from X @ G_k for each feature k.
    """
    XG = [X @ G for G in sims]
    y = X.ravel()
    Xr = np.column_stack([xg.ravel() for xg in XG])
    w = np.where(y > 0, pos_weight, neg_weight)
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(Xr, y, sample_weight=w)
    return reg.coef_, XG


def learn_weights_nnls(
    X: np.ndarray,
    sims: List[np.ndarray],
    pos_weight: float = 1.0,
    neg_weight: float = 0.2,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Non-negative least squares for µ weights (µ_k >= 0).
    Uses scipy.optimize.nnls on the weighted system.
    """
    XG = [X @ G for G in sims]
    y = X.ravel()
    Xr = np.column_stack([xg.ravel() for xg in XG])
    w_sqrt = np.sqrt(np.where(y > 0, pos_weight, neg_weight))
    # Weight the system: W^{1/2} * Xr * mu = W^{1/2} * y
    Xr_w = Xr * w_sqrt[:, np.newaxis]
    y_w = y * w_sqrt
    coefs, _ = nnls(Xr_w, y_w)
    return coefs, XG


def learn_weights_ridge(
    X: np.ndarray,
    sims: List[np.ndarray],
    alpha_ridge: float = 1.0,
    fit_intercept: bool = True,
    pos_weight: float = 1.0,
    neg_weight: float = 0.2,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """L2-regularized weight learning."""
    XG = [X @ G for G in sims]
    y = X.ravel()
    Xr = np.column_stack([xg.ravel() for xg in XG])
    w = np.where(y > 0, pos_weight, neg_weight)
    reg = Ridge(alpha=alpha_ridge, fit_intercept=fit_intercept)
    reg.fit(Xr, y, sample_weight=w)
    return reg.coef_, XG


def compute_dr(
    X: np.ndarray,
    beta: float,
    percentile: int = 10,
) -> np.ndarray:
    """
    Cold-item weighting: step-linear decreasing function.
    d_j = (beta/p) * (p - r_j) if r_j <= p, else 0
    where r_j = sum of interactions for item j, p = percentile.
    """
    v = np.asarray(X.sum(axis=0)).ravel() if sp.issparse(X) else X.sum(axis=0)
    p = max(np.percentile(v, percentile), 1.0)
    k = beta / p
    return np.where(v <= p, k * (p - v), 0.0)


def ease_aligned(
    X: np.ndarray,
    Xtilde: np.ndarray,
    lambda1: float = 1.0,
    beta: float = 1.0,
    alpha: float = 1.0,
    dr_percentile: int = 10,
    XtX: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    EASE with metadata alignment (Eq.8-9 in MARec paper).

    min_B ||X - XB||^2 + λ1||B||^2 + ||XB D^R - X̃||^2
    s.t. diag(B) = 0

    Returns: B (item-item weight matrix)
    """
    n = X.shape[1]
    dr = compute_dr(X, beta, percentile=dr_percentile)
    Xt_dr = (alpha * Xtilde) * dr[np.newaxis, :]
    XtXt_IR = X.T @ Xt_dr
    if XtX is None:
        XtX = X.T @ X
    P = np.linalg.inv(XtX + lambda1 * np.eye(n) + XtXt_IR)
    Bt = P @ (XtX + XtXt_IR)
    # Zero-diagonal constraint via Lagrangian heuristic
    gamma = np.diag(Bt) / np.diag(P)
    B = Bt - P @ np.diag(gamma)
    return B


def run_marec(
    X: np.ndarray,
    sim_list: List[np.ndarray],
    lambda1: float = 1.0,
    beta: float = 100.0,
    alpha: float = 1.0,
    weight_method: str = "regression",
    weight_kwargs: Optional[Dict] = None,
    dr_percentile: int = 10,
    XtX: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full MARec pipeline: learn weights → build Xtilde → EASE.

    Args:
        weight_method: "regression" | "nnls" | "ridge"

    Returns: (predictions, coefs)
    """
    wkw = weight_kwargs or {}

    if weight_method == "nnls":
        coefs, XG = learn_weights_nnls(X, sim_list, **wkw)
    elif weight_method == "ridge":
        coefs, XG = learn_weights_ridge(X, sim_list, **wkw)
    else:
        coefs, XG = learn_weights_regression(X, sim_list, **wkw)

    Xtilde = sum(c * xg for c, xg in zip(coefs, XG))
    B = ease_aligned(
        X, Xtilde, lambda1=lambda1, beta=beta, alpha=alpha,
        dr_percentile=dr_percentile, XtX=XtX,
    )
    return X @ B, coefs


def run_marec_sparse(
    train_csr: sp.csr_matrix,
    sim_list: List[np.ndarray],
    lambda1: float = 1.0,
    beta: float = 100.0,
    alpha: float = 1.0,
    weight_method: str = "regression",
    weight_kwargs: Optional[Dict] = None,
    dr_percentile: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MARec with sparse input. Densifies only the train matrix (required
    for EASE inverse). Similarity matrices remain as provided.
    """
    X = train_csr.toarray().astype(np.float64)
    XtX = X.T @ X
    return run_marec(
        X, sim_list,
        lambda1=lambda1, beta=beta, alpha=alpha,
        weight_method=weight_method, weight_kwargs=weight_kwargs,
        dr_percentile=dr_percentile, XtX=XtX,
    )
