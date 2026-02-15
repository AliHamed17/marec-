#!/usr/bin/env python3
"""Test script to verify tag mode switching works correctly."""
from src.config import Config
from src.experiments import ExperimentRunner

print("=" * 70)
print("Test 1: no_tags with top3")
print("=" * 70)
cfg1 = Config(
    n_splits=1,
    sanity_mode=True,
    tag_mode="no_tags",
    feature_configs={"top3": ["actors", "directors", "genres"]},
    hp_grid={"lambda1": [10], "beta": [1], "alpha": [1]},
    tune_splits=[0],
    output_dir="results/test_no_tags",
)
cfg1.validate()
runner1 = ExperimentRunner(cfg1)
results1 = runner1.run()

print("\n" + "=" * 70)
print("Test 2: tags_train_users with top3+tags")
print("=" * 70)
cfg2 = Config(
    n_splits=1,
    sanity_mode=True,
    tag_mode="tags_train_users",
    leakage_safe=True,
    feature_configs={"top3_tags": ["actors", "directors", "genres", "tags"]},
    hp_grid={"lambda1": [10], "beta": [1], "alpha": [1]},
    tune_splits=[0],
    output_dir="results/test_tags_train_users",
)
cfg2.validate()
runner2 = ExperimentRunner(cfg2)
results2 = runner2.run()

print("\n" + "=" * 70)
print("Test 3: tags_full with top3+tags")
print("=" * 70)
cfg3 = Config(
    n_splits=1,
    sanity_mode=True,
    tag_mode="tags_full",
    leakage_safe=False,
    feature_configs={"top3_tags": ["actors", "directors", "genres", "tags"]},
    hp_grid={"lambda1": [10], "beta": [1], "alpha": [1]},
    tune_splits=[0],
    output_dir="results/test_tags_full",
)
cfg3.validate()
runner3 = ExperimentRunner(cfg3)
results3 = runner3.run()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"no_tags hr@10: {results1.get('top3', {}).get('avg_metrics', {}).get('hr@10', 'N/A')}")
print(f"tags_train_users hr@10: {results2.get('top3_tags', {}).get('avg_metrics', {}).get('hr@10', 'N/A')}")
print(f"tags_full hr@10: {results3.get('top3_tags', {}).get('avg_metrics', {}).get('hr@10', 'N/A')}")
