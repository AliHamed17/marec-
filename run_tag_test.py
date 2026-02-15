#!/usr/bin/env python3
"""Run a quick test of tags_train_users mode."""
from src.config import Config
from src.experiments import ExperimentRunner

print("=" * 70)
print("Testing tags_train_users mode")
print("=" * 70)

cfg = Config(
    tag_mode="tags_train_users",
    leakage_safe=True,
    feature_configs={"top3_tags": ["actors", "directors", "genres", "tags"]},
    n_splits=1,
    hp_grid={"lambda1": [1, 10], "beta": [1, 100], "alpha": [1, 10]},
    tune_splits=[0],
    output_dir="results/test_tags_train_users",
)

cfg.validate()
print(f"Tag mode: {cfg.tag_mode}")
print(f"Feature configs: {list(cfg.feature_configs.keys())}")
print(f"Features in top3_tags: {cfg.feature_configs['top3_tags']}")

runner = ExperimentRunner(cfg)
results = runner.run()

print("\n" + "=" * 70)
print("Test completed!")
print("=" * 70)

