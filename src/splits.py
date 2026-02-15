"""
src/splits.py — Cold-start item splits (fully sparse, no .toarray()).
"""
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple


def create_cold_split_sparse(
    URM: sp.csr_matrix,
    cold_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Create a cold-start item split in FULLY SPARSE format.

    Selects cold_frac of items as cold; zeroes their columns in train,
    keeps them in test.

    Returns: (train_csr, test_csr, cold_items_array)
    """
    rng = np.random.RandomState(seed)
    n_u, n_i = URM.shape
    n_cold = int(n_i * cold_frac)
    cold_items = np.sort(rng.choice(n_i, n_cold, replace=False))
    cold_set = set(cold_items)

    # Convert to COO for efficient column-based splitting
    coo = URM.tocoo()
    train_mask = np.array([c not in cold_set for c in coo.col])
    test_mask = ~train_mask

    train = sp.csr_matrix(
        (coo.data[train_mask], (coo.row[train_mask], coo.col[train_mask])),
        shape=(n_u, n_i),
    )
    test = sp.csr_matrix(
        (coo.data[test_mask], (coo.row[test_mask], coo.col[test_mask])),
        shape=(n_u, n_i),
    )
    return train, test, cold_items


def generate_splits(
    URM: sp.csr_matrix,
    n_splits: int = 10,
    cold_frac: float = 0.20,
    base_seed: int = 42,
) -> List[Dict]:
    """Generate n_splits cold-start splits."""
    splits = []
    for i in range(n_splits):
        train, test, cold_items = create_cold_split_sparse(
            URM, cold_frac=cold_frac, seed=base_seed + i
        )
        splits.append({
            "train": train,
            "test": test,
            "cold_items": cold_items,
            "seed": base_seed + i,
        })
    return splits


def verify_split(split: Dict) -> bool:
    """
    Verify cold-start split integrity:
    - Train has zero entries for all cold items
    - Test only has entries for cold items
    - No overlap between train and test item columns
    - Log cold item count
    """
    train, test, cold_items = split["train"], split["test"], split["cold_items"]
    cold_set = set(cold_items)
    n_items = train.shape[1]
    warm_items = set(range(n_items)) - cold_set

    # Check train has no cold item interactions
    train_coo = train.tocoo()
    train_cold_cols = set(train_coo.col) & cold_set
    assert len(train_cold_cols) == 0, (
        f"Train contains {len(train_cold_cols)} cold item columns!"
    )
    
    # Verify train[:, cold_items] has zero nonzeros
    train_cold_submatrix = train[:, list(cold_items)]
    assert train_cold_submatrix.nnz == 0, (
        f"Train has {train_cold_submatrix.nnz} nonzeros in cold item columns!"
    )

    # Check test only has cold item interactions
    test_coo = test.tocoo()
    test_warm_cols = set(test_coo.col) - cold_set
    assert len(test_warm_cols) == 0, (
        f"Test contains {len(test_warm_cols)} warm item columns!"
    )
    
    # Verify test[:, warm_items] has zero nonzeros
    test_warm_submatrix = test[:, list(warm_items)]
    assert test_warm_submatrix.nnz == 0, (
        f"Test has {test_warm_submatrix.nnz} nonzeros in warm item columns!"
    )

    # Check no NaN
    assert not np.any(np.isnan(train.data)), "Train contains NaN"
    assert not np.any(np.isnan(test.data)), "Test contains NaN"
    
    # Log statistics
    print(f"    Cold items: {len(cold_items)} ({100*len(cold_items)/n_items:.1f}%), "
          f"Warm items: {len(warm_items)} ({100*len(warm_items)/n_items:.1f}%)")

    return True
