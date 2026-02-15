# MARec Framework — Modular Cold-Start Recommendation

A refactored, leakage-safe, memory-efficient experimental framework for reproducing and extending MARec (Metadata-Aligned Recommendation for Cold-Start Items).

## Architecture

```
marec_framework/
├── run_experiments.py          # CLI entry point
├── requirements.txt
├── configs/
│   ├── leakage_safe.json       # Default safe config (tags_train_only)
│   └── upper_bound.json        # Tags-full upper bound (labeled)
├── src/
│   ├── __init__.py
│   ├── data.py                 # Download, load, binarize, k-core filter
│   ├── splits.py               # Sparse cold-start splits (CSR throughout)
│   ├── features.py             # Multi-hot/TF-IDF encoding, 3 tag modes
│   ├── similarity.py           # Smoothed cosine, year sim, TopK, Hadamard cross
│   ├── marec.py                # EASE backbone + alignment (regression/NNLS/Ridge)
│   ├── eval.py                 # hr@k, ndcg@k (corrected), bootstrap CI, paired tests
│   ├── checks.py               # Split integrity, EASE matrix, metric unit tests
│   ├── config.py               # Dataclass config with validation + presets
│   └── experiments.py          # ExperimentRunner: tune → evaluate → save
└── results/                    # Auto-created output directory
```

### Module Responsibilities

| Module | Role | Key Design Choices |
|--------|------|--------------------|
| `data.py` | Data I/O | Auto-downloads HetRec; returns sparse CSR URM |
| `splits.py` | Cold-start splits | COO-based column splitting — never densifies |
| `features.py` | Feature encoding | 3 tag modes; `tags_train_only` filters by train interactions |
| `similarity.py` | Similarity matrices | Smoothed cosine with per-feature δ; TopK sparsification |
| `marec.py` | Core model | EASE + alignment; supports regression/NNLS/Ridge weights |
| `eval.py` | Metrics | Corrected hr@k denominator `min(k, \|I_u\|)`; bootstrap CI |
| `checks.py` | Verification | Split integrity, diagonal-free EASE, metric unit tests |
| `config.py` | Configuration | Validates leakage safety; serializes to/from JSON |
| `experiments.py` | Orchestrator | HP grid search, N-split evaluation, statistical tests, logging |

### Data Flow

```
download_hetrec() → load_all_dataframes() → binarize_and_kcore()
       ↓                                            ↓
    raw .dat files                              URM (sparse CSR)
                                                     ↓
                                           generate_splits() [sparse]
                                                     ↓
                                         build_feature_matrices()
                                           [3 tag modes: no/train_only/full]
                                                     ↓
                                         build_similarity_matrices()
                                           [cosine + cross + TopK]
                                                     ↓
                                              run_marec()
                                           [weights → X̃ → EASE]
                                                     ↓
                                              evaluate()
                                           [hr@k, ndcg@k, CI, tests]
```

## Quick Start

```bash
pip install -r requirements.txt

# Sanity check (1 split, ~2 min)
python run_experiments.py --sanity

# Leakage-safe experiment (10 splits, ~30 min)
python run_experiments.py

# Upper-bound with tags_full (10 splits)
python run_experiments.py --upper-bound

# Custom config
python run_experiments.py --config configs/leakage_safe.json

# Full ordered suite (all experiments)
python run_experiments.py --all
```

## Tag Leakage Modes

| Mode | Description | Leakage? | Use Case |
|------|-------------|----------|----------|
| `no_tags` | Tags excluded entirely | No | Baseline |
| `tags_train_only` | Tags built from train-only user-item pairs per split | No | **Recommended** |
| `tags_full` | All tag assignments used (ignores split) | **Yes** | Upper bound only |

When `leakage_safe=True` (default), the framework **refuses to run** with `tags_full` and raises `ValueError`.

### How `tags_train_only` Works

For each cold-start split, a tag assignment `(user_u, item_i, tag_t)` is included in the tag feature matrix **only if** `train[u, i] > 0` — meaning the user-item interaction is in the training set for that split. This ensures no information about cold-start test items leaks through tag co-occurrence.

## Weight Learning Methods

| Method | Description | Constraint |
|--------|-------------|------------|
| `regression` | Weighted OLS (original MARec) | None (µ can be negative) |
| `nnls` | Non-negative least squares | µ_k ≥ 0 |
| `ridge` | L2-regularized regression | Shrinks µ toward 0 |

## Hyperparameter Grid

Default grid (256 combinations):

| Parameter | Values | Role |
|-----------|--------|------|
| λ₁ | {0.1, 1, 10, 100} | EASE L2 regularization |
| β | {1, 10, 100, 500} | Cold-item weighting strength |
| α | {0.1, 1, 10, 100} | Alignment contribution weight |

## Output Files

After each experiment:

```
results/
├── metrics.csv              # Per-config averages + HP + tag_mode
├── per_split_metrics.csv    # Every split × config combination
├── best_config.json         # Best HP, metrics, bootstrap CI
├── config_used.json         # Exact config that was run
└── run.log                  # Timestamped execution log
```

## Experiment Checklist (Recommended Order)

Run these in order to build up from baseline to full results:

### Phase 1: Baseline
- [ ] **Sanity check** — `--sanity` (1 split, 2 configs, ~2 min)
- [ ] **No-tags baseline** — base9 and top3 features, 10 splits

### Phase 2: Leakage-Safe Tags
- [ ] **tags_train_only** — top3 + tags, 10 splits
- [ ] Compare vs no-tags baseline (paired t-test)

### Phase 3: Model Improvements
- [ ] **Cross-features** — Hadamard products (actors×directors, etc.)
- [ ] **NNLS weights** — Non-negative µ constraint
- [ ] **Ridge weights** — L2-regularized µ
- [ ] **TopK sparsification** — K ∈ {50, 100, 200, 500}
- [ ] **Cold percentile** — dr_percentile ∈ {5, 10, 20}
- [ ] **Per-feature shrinkage** — Different δ per feature type

### Phase 4: Upper Bound
- [ ] **tags_full** — All tag assignments (label as upper bound)
- [ ] Compare tags_train_only vs tags_full to quantify leakage gap

### Phase 5: Reporting
- [ ] Best config with bootstrap 95% CI
- [ ] Paired statistical tests (best vs baselines)
- [ ] Per-split variance plots

## Key Design Decisions

1. **Sparse splits**: Cold-start splits use COO→CSR manipulation, never calling `.toarray()` on the full URM during splitting.

2. **Dense EASE**: The EASE closed-form solution requires matrix inversion (`(XᵀX + λI)⁻¹`), which needs dense `XᵀX`. We densify only the `n_items × n_items` Gram matrix, not the full URM. The train matrix is densified per-split for the `X @ B` prediction step (unavoidable for EASE).

3. **Similarity matrices stay dense**: Item-item similarity matrices (`n_items × n_items ≈ 7433²`) are ~440 MB each in float64. With ~15 similarity matrices, this is the main memory bottleneck. TopK sparsification can reduce this but the alignment regression requires dense `X @ G` products.

4. **Per-split feature rebuilding**: When `tag_mode=tags_train_only`, features and similarities are rebuilt for **every split** since the tag representation depends on which items are cold. This is slower but leak-free.

5. **Metric correction**: Hit rate uses `min(k, |I_u|)` as denominator (not just `k`), matching standard practice and our paper's corrected formulation.

## Reproducing Paper Results

With the default leakage-safe config:

| Config | hr@10 | ndcg@10 | Note |
|--------|-------|---------|------|
| MARec paper | 0.293 | 0.307 | Original reported |
| base9 (our repro) | ~0.311 | ~0.330 | +6.4% (metric correction) |
| top3_tags_safe | TBD | TBD | Leakage-safe tags |
| top3_tags_full | ~0.589 | ~0.622 | Upper bound (leakage) |

The gap between `tags_train_only` and `tags_full` quantifies how much of the v6 improvement came from leaked test information.
