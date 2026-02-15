#!/usr/bin/env python3
"""
run_experiments.py — Main entry point for MARec experiments.

Usage:
    python run_experiments.py                  # Default leakage-safe
    python run_experiments.py --sanity         # Quick sanity check (1 split)
    python run_experiments.py --upper-bound    # Tags-full upper bound
    python run_experiments.py --config cfg.json # Custom config
    python run_experiments.py --all            # Run full experiment suite
"""
import argparse
import sys
import time

from src.config import Config, config_leakage_safe, config_tags_full_upper_bound, config_sanity
from src.experiments import ExperimentRunner


def run_suite(args):
    """Run a full ordered experiment suite."""
    print("=" * 70)
    print("  MARec Experiment Suite — Full Ordered Run")
    print("=" * 70)

    suite = [
        # 1. Sanity check
        ("sanity", config_sanity()),
        # 2. Baseline (no tags)
        ("baseline_no_tags", Config(
            tag_mode="no_tags",
            feature_configs={
                "base9": [
                    "genres", "actors", "directors", "countries",
                    "loc1", "loc2", "loc3", "years", "locations",
                ],
                "top3": ["actors", "directors", "genres"],
            },
            output_dir="results/01_baseline",
        )),
        # 3. Leakage-safe tags
        ("safe_tags", Config(
            tag_mode="tags_train_only",
            leakage_safe=True,
            feature_configs={
                "top3": ["actors", "directors", "genres"],
                "top3_tags_safe": ["actors", "directors", "genres", "tags"],
            },
            output_dir="results/02_safe_tags",
        )),
        # 4. Cross-features
        ("cross_features", Config(
            tag_mode="no_tags",
            cross_pairs=[("actors", "directors"), ("actors", "genres"),
                         ("directors", "genres")],
            feature_configs={
                "top3_cross": [
                    "actors", "directors", "genres",
                    "actors_x_directors", "actors_x_genres", "directors_x_genres",
                ],
            },
            output_dir="results/03_cross",
        )),
        # 5. NNLS weight learning
        ("nnls_weights", Config(
            tag_mode="no_tags",
            weight_method="nnls",
            feature_configs={
                "top3_nnls": ["actors", "directors", "genres"],
            },
            output_dir="results/04_nnls",
        )),
        # 6. Cold-item percentile variants
        ("dr_percentiles", Config(
            tag_mode="no_tags",
            dr_percentile=5,
            feature_configs={
                "top3_p5": ["actors", "directors", "genres"],
            },
            output_dir="results/05_dr_p5",
        )),
        # 7. Upper bound (tags_full)
        ("upper_bound", Config(
            tag_mode="tags_full",
            leakage_safe=False,
            feature_configs={
                "top3_tags_full": ["actors", "directors", "genres", "tags"],
            },
            output_dir="results/06_upper_bound",
        )),
    ]

    for name, cfg in suite:
        print(f"\n{'='*60}")
        print(f"  Experiment: {name}")
        print(f"{'='*60}")
        try:
            cfg.validate()
            runner = ExperimentRunner(cfg)
            runner.run()
        except Exception as e:
            print(f"  FAILED: {e}")
            if name == "sanity":
                print("Sanity check failed — aborting suite.")
                sys.exit(1)

    print("\n" + "=" * 70)
    print("  Suite complete! Results in results/01_*..results/06_*")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MARec Experiment Runner")
    parser.add_argument("--sanity", action="store_true",
                        help="Quick sanity check (1 split)")
    parser.add_argument("--upper-bound", action="store_true",
                        help="Run tags_full upper bound experiment")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--all", action="store_true",
                        help="Run full experiment suite")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 split, small HP grid")
    args = parser.parse_args()

    if args.all:
        run_suite(args)
        return

    # Select config
    if args.config:
        cfg = Config.load(args.config)
    elif args.sanity:
        cfg = config_sanity()
    elif args.upper_bound:
        cfg = config_tags_full_upper_bound()
    else:
        cfg = config_leakage_safe()

    # Apply quick mode if requested
    if args.quick:
        cfg.n_splits = 1
        cfg.sanity_mode = True
        cfg.hp_grid = {
            "lambda1": [1, 10],
            "beta": [1, 100],
            "alpha": [1, 10],
        }
        cfg.tune_splits = [0]

    if args.output_dir:
        cfg.output_dir = args.output_dir

    cfg.validate()

    print("=" * 60)
    print(f"  MARec Experiment Runner")
    print(f"  Tag mode: {cfg.tag_mode}")
    print(f"  Leakage safe: {cfg.leakage_safe}")
    print(f"  Splits: {cfg.n_splits}")
    print(f"  Configs: {list(cfg.feature_configs.keys())}")
    print(f"  Weight method: {cfg.weight_method}")
    print("=" * 60)

    runner = ExperimentRunner(cfg)
    results = runner.run()


if __name__ == "__main__":
    main()
