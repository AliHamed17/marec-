"""
src/config.py — Configuration system for experiments.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class Config:
    """Experiment configuration."""
    # Dataset
    data_dir: str = "hetrec_data"
    binarize_threshold: float = 3.0
    min_user_kcore: int = 5
    min_item_kcore: int = 5

    # Splits
    n_splits: int = 10
    cold_frac: float = 0.20
    base_seed: int = 42

    # Features
    min_feature_count: int = 2
    tag_mode: str = "no_tags"  # "no_tags" | "tags_train_only" | "tags_train_users" | "tags_train_pairs" | "tags_full"
    leakage_safe: bool = True  # If True, blocks tags_full

    # Similarity
    default_shrinkage: float = 20.0
    per_feature_shrinkage: Optional[Dict[str, float]] = None
    cross_pairs: Optional[List[Tuple[str, str]]] = None
    topk: Optional[int] = None
    topk_features: Optional[List[str]] = None

    # Feature sets to evaluate (list of feature name lists)
    feature_configs: Dict[str, List[str]] = field(default_factory=lambda: {
        "base9": [
            "genres", "actors", "directors", "countries",
            "loc1", "loc2", "loc3", "years", "locations",
        ],
        "top3": ["actors", "directors", "genres"],
    })

    # Hyperparameter grid
    hp_grid: Dict[str, List] = field(default_factory=lambda: {
        "lambda1": [0.1, 1, 10, 100],
        "beta": [1, 10, 100, 500],
        "alpha": [0.1, 1, 10, 100],
    })

    # Weight learning
    weight_method: str = "regression"  # "regression" | "nnls" | "ridge"
    weight_kwargs: Dict = field(default_factory=dict)

    # Cold-item weighting
    dr_percentile: int = 10

    # Evaluation
    eval_ks: List[int] = field(default_factory=lambda: [10, 25])
    bootstrap_n: int = 500
    bootstrap_frac: float = 0.2

    # HP tuning strategy
    tune_splits: List[int] = field(default_factory=lambda: [0, 1, 2])
    # Splits used for tuning. Remaining splits used for evaluation.
    # If [0], tunes on split 0 only (fast). If [0,1,2], avg of 3 splits (recommended).
    # After tuning, best HPs are frozen and used for all splits.

    # Output
    output_dir: str = "results"
    sanity_mode: bool = False  # Quick single-split run

    def validate(self):
        """Validate config consistency."""
        if self.leakage_safe and self.tag_mode == "tags_full":
            raise ValueError(
                "Cannot use tags_full when leakage_safe=True. "
                "Use tags_train_users, tags_train_pairs, or tags_train_only, or set leakage_safe=False."
            )
        assert self.tag_mode in ("no_tags", "tags_train_only", "tags_train_users", "tags_train_pairs", "tags_full")
        assert self.weight_method in ("regression", "nnls", "ridge")
        assert 0 < self.cold_frac < 1
        assert self.n_splits >= 1

    def save(self, path: str):
        """Save config to JSON."""
        d = {k: v for k, v in self.__dict__.items()}
        # Convert tuples to lists for JSON
        if d.get("cross_pairs"):
            d["cross_pairs"] = [list(p) for p in d["cross_pairs"]]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from JSON."""
        with open(path) as f:
            d = json.load(f)
        if d.get("cross_pairs"):
            d["cross_pairs"] = [tuple(p) for p in d["cross_pairs"]]
        if d.get("eval_ks"):
            d["eval_ks"] = list(d["eval_ks"])
        return cls(**d)


# ─── Pre-built configs ────────────────────────────────────────────────

def config_leakage_safe() -> Config:
    """Default leakage-safe config (recommended)."""
    return Config(
        tag_mode="tags_train_users",  # Recommended: filter by train users
        leakage_safe=True,
        feature_configs={
            "base9": [
                "genres", "actors", "directors", "countries",
                "loc1", "loc2", "loc3", "years", "locations",
            ],
            "top3": ["actors", "directors", "genres"],
            "top3_tags": ["actors", "directors", "genres", "tags"],
        },
        cross_pairs=[("actors", "directors"), ("actors", "genres"),
                     ("directors", "genres")],
    )


def config_tags_full_upper_bound() -> Config:
    """Upper-bound config with tags_full (for comparison only)."""
    return Config(
        tag_mode="tags_full",
        leakage_safe=False,
        feature_configs={
            "base9": [
                "genres", "actors", "directors", "countries",
                "loc1", "loc2", "loc3", "years", "locations",
            ],
            "top3": ["actors", "directors", "genres"],
            "top3_tags": ["actors", "directors", "genres", "tags"],
            "9feat_tags": [
                "genres", "actors", "directors", "countries",
                "loc1", "loc2", "loc3", "years", "locations", "tags",
            ],
        },
    )


def config_sanity() -> Config:
    """Quick sanity-check config (1 split, small grid)."""
    return Config(
        n_splits=1,
        sanity_mode=True,
        tag_mode="no_tags",
        feature_configs={"top3": ["actors", "directors", "genres"]},
        hp_grid={
            "lambda1": [1, 10],
            "beta": [1, 100],
            "alpha": [1, 10],
        },
        tune_splits=[0],  # Single split for sanity
    )
