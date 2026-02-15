"""
src/experiments.py — Experiment runner with HP tuning, logging, and timing.
"""
import time
import json
import tracemalloc
from pathlib import Path
from itertools import product as iterproduct
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import pandas as pd

from .config import Config
from .data import load_dataset
from .splits import generate_splits, verify_split
from .features import build_feature_matrices
from .similarity import build_similarity_matrices
from .marec import run_marec, ease_aligned, learn_weights_regression, \
    learn_weights_nnls, learn_weights_ridge, compute_dr
from .eval import evaluate, bootstrap_ci, paired_bootstrap_test, paired_ttest
from .checks import run_all_checks, check_ease_matrix


class ExperimentRunner:
    """Orchestrates the full experimental pipeline."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cfg.validate()
        self.results = {}
        self.all_experiments = []
        self.log_lines = []
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line)
        except UnicodeEncodeError:
            # Fallback for Windows console encoding issues
            safe_msg = msg.encode('ascii', 'replace').decode('ascii')
            safe_line = f"[{ts}] {safe_msg}"
            print(safe_line)
        self.log_lines.append(line)

    def run(self):
        """Execute the full experimental pipeline."""
        t_start = time.time()
        tracemalloc.start()

        # 1) Load data
        self.log("Loading dataset...")
        ds = load_dataset(self.cfg.data_dir, self.cfg.binarize_threshold)
        self.ds = ds

        # 2) Generate splits
        self.log(f"Generating {self.cfg.n_splits} cold-start splits...")
        splits = generate_splits(
            ds["URM"], self.cfg.n_splits, self.cfg.cold_frac, self.cfg.base_seed
        )
        self.splits = splits

        # 3) Verify splits
        self.log("Running verification checks...")
        ok = run_all_checks(splits, verbose=True)
        if not ok:
            self.log("WARNING: Some checks failed!")

        # 4) For each feature config: build features, sims, tune, evaluate
        all_config_results = {}

        for config_name, feat_names in self.cfg.feature_configs.items():
            self.log(f"\n{'='*60}")
            self.log(f"Config: {config_name} — features: {feat_names}")
            self.log(f"{'='*60}")

            result = self._run_config(config_name, feat_names, splits, ds)
            all_config_results[config_name] = result

        # 5) Find best config
        best_name = max(
            all_config_results,
            key=lambda k: all_config_results[k]["avg_metrics"].get("hr@10", 0)
        )
        best = all_config_results[best_name]
        self.log(f"\nBEST CONFIG: {best_name}")
        for k, v in best["avg_metrics"].items():
            self.log(f"  {k}: {v:.4f}")

        # 6) Statistical tests: best vs all others
        self.log("\nStatistical comparisons (best vs others):")
        for name, res in all_config_results.items():
            if name == best_name:
                continue
            best_hr = best["per_split_metrics"]
            other_hr = res["per_split_metrics"]
            if len(best_hr) > 1 and len(other_hr) > 1:
                best_vals = [m.get("hr@10", 0) for m in best_hr]
                other_vals = [m.get("hr@10", 0) for m in other_hr]
                p_boot = paired_bootstrap_test(best_vals, other_vals)
                t_stat, p_tt = paired_ttest(best_vals, other_vals)
                self.log(
                    f"  {best_name} vs {name}: "
                    f"boot_p={p_boot:.4f}, ttest_p={p_tt:.4f}"
                )

        # 7) Bootstrap CI on best config's last split
        self.log(f"\nBootstrap CIs for {best_name}:")
        last_split_idx = self.cfg.n_splits - 1
        if last_split_idx < len(splits):
            pred_final, _ = self._get_prediction(
                splits[last_split_idx]["train"],
                best["best_hp"],
                best["feature_names"],
                ds, splits[last_split_idx],
            )
            ci = bootstrap_ci(
                pred_final,
                splits[last_split_idx]["test"],
                splits[last_split_idx]["train"],
                ks=self.cfg.eval_ks,
                n_boot=self.cfg.bootstrap_n,
                frac=self.cfg.bootstrap_frac,
                cold_items=splits[last_split_idx]["cold_items"],
            )
            for metric, (mean, lo, hi) in ci.items():
                self.log(f"  {metric}: {mean:.4f} [95% CI: {lo:.4f} — {hi:.4f}]")
            best["bootstrap_ci"] = {k: {"mean": v[0], "lo": v[1], "hi": v[2]}
                                    for k, v in ci.items()}

        # 8) Save results
        self._save_results(all_config_results)

        # Timing
        peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        tracemalloc.stop()
        elapsed = time.time() - t_start
        self.log(f"\nTotal time: {elapsed:.1f}s, peak memory: {peak_mem:.0f} MB")

        return all_config_results

    def _run_config(
        self,
        config_name: str,
        feat_names: List[str],
        splits: List[Dict],
        ds: Dict,
    ) -> Dict:
        """Run a single feature configuration: tune HPs then evaluate all splits."""

        needs_tags = any("tags" in f for f in feat_names)
        tag_mode = self.cfg.tag_mode if needs_tags else "no_tags"

        # For tags_train_* modes, features must be rebuilt per split.
        # For no_tags or tags_full, features are built once globally.
        rebuild_per_split = tag_mode in ("tags_train_only", "tags_train_users", "tags_train_pairs")

        tag_stats_list = []
        
        if not rebuild_per_split:
            self.log(f"  Building features globally (tag_mode={tag_mode})...")
            encoded = build_feature_matrices(
                ds["dfs"], ds["item2idx"], ds["idx2item"], ds["n_items"],
                tag_mode=tag_mode, min_count=self.cfg.min_feature_count,
            )
            
            # Validation: ensure tags are built when needed
            if needs_tags:
                assert "tags" in encoded, (
                    f"Tags required (feat_names includes 'tags') but tag matrix not built! "
                    f"tag_mode={tag_mode}, available features: {list(encoded.keys())}"
                )
                assert encoded["tags"].shape[0] == ds["n_items"], (
                    f"Tag matrix shape mismatch: {encoded['tags'].shape[0]} != {ds['n_items']}"
                )
                if hasattr(encoded["tags"], "nnz"):
                    assert encoded["tags"].nnz > 0, "Tag matrix is empty!"
                else:
                    assert np.any(encoded["tags"] > 0), "Tag matrix is empty!"
                self.log(f"    Tags matrix: shape={encoded['tags'].shape}, "
                        f"nnz={encoded['tags'].nnz if hasattr(encoded['tags'], 'nnz') else np.count_nonzero(encoded['tags'])}")
            
            # Log all feature shapes
            for name, mat in encoded.items():
                if name != "_tag_stats":
                    shape_str = f"{mat.shape}"
                    nnz_str = f", nnz={mat.nnz}" if hasattr(mat, "nnz") else f", nonzeros={np.count_nonzero(mat)}"
                    self.log(f"    {name}: {shape_str}{nnz_str}")
            
            # Save tag statistics if available
            if "_tag_stats" in encoded:
                tag_stats_list.append(encoded.pop("_tag_stats"))
            
            S = build_similarity_matrices(
                encoded,
                shrinkage=self.cfg.default_shrinkage,
                per_feature_shrinkage=self.cfg.per_feature_shrinkage,
                cross_pairs=self.cfg.cross_pairs,
                topk=self.cfg.topk,
                topk_features=self.cfg.topk_features,
            )
            
            # Validation: ensure similarity matrices include all required features
            missing = [f for f in feat_names if f not in S and not any(f in sname for sname in S.keys())]
            if missing:
                raise KeyError(
                    f"Similarity matrices missing required features: {missing}. "
                    f"Available: {list(S.keys())}, requested: {feat_names}"
                )
            
            sim_list = self._get_sims(feat_names, S)
        else:
            S = None
            sim_list = None
            self.log(f"  Features will be rebuilt per split (tag_mode={tag_mode} requires per-split rebuilding)")

        # --- HP tuning on designated splits ---
        self.log(f"  HP tuning on splits {self.cfg.tune_splits}...")
        t_hp = time.time()
        best_hp = self._tune_hp(
            feat_names, splits, ds, tag_mode,
            S_global=S if not rebuild_per_split else None,
        )
        self.log(
            f"  Best HP: lambda1={best_hp['lambda1']}, "
            f"beta={best_hp['beta']}, alpha={best_hp['alpha']} "
            f"({time.time()-t_hp:.1f}s)"
        )

        # --- Full evaluation on all splits ---
        self.log(f"  Full {self.cfg.n_splits}-split evaluation...")
        t_eval = time.time()
        per_split = []
        for si in range(self.cfg.n_splits):
            pred, split_tag_stats = self._get_prediction(
                splits[si]["train"], best_hp, feat_names, ds, splits[si],
                tag_mode=tag_mode,
                S_global=S if not rebuild_per_split else None,
            )
            # Collect tag stats from per-split rebuilds
            if rebuild_per_split and split_tag_stats and si == 0:
                tag_stats_list.append(split_tag_stats)
            # Check EASE matrix (on first split only)
            if si == 0:
                X = splits[si]["train"].toarray().astype(np.float64)
                # Just verify predictions have no NaN
                if np.any(np.isnan(pred)):
                    self.log(f"  WARNING: predictions contain NaN on split {si}")

            m = evaluate(pred, splits[si]["test"], splits[si]["train"],
                         ks=self.cfg.eval_ks,
                         cold_items=splits[si]["cold_items"])
            per_split.append(m)

        avg = {k: np.mean([m[k] for m in per_split if k in m])
               for k in per_split[0]}
        std = {k: np.std([m[k] for m in per_split if k in m])
               for k in per_split[0]}

        for k in sorted(avg.keys()):
            self.log(f"  {k}: {avg[k]:.4f} ± {std[k]:.4f}")

        self.log(f"  ({time.time()-t_eval:.1f}s)")

        return {
            "config_name": config_name,
            "feature_names": feat_names,
            "best_hp": best_hp,
            "avg_metrics": avg,
            "std_metrics": std,
            "per_split_metrics": per_split,
            "tag_mode": tag_mode,
            "tag_stats": tag_stats_list[0] if tag_stats_list else None,
        }

    def _tune_hp(
        self,
        feat_names: List[str],
        splits: List[Dict],
        ds: Dict,
        tag_mode: str,
        S_global: Optional[Dict] = None,
    ) -> Dict:
        """Grid-search HP tuning on self.cfg.tune_splits."""
        grid = self.cfg.hp_grid
        best_score = -1
        best_hp = {"lambda1": 1.0, "beta": 100.0, "alpha": 1.0}

        for l1, beta, alpha in iterproduct(
            grid["lambda1"], grid["beta"], grid["alpha"]
        ):
            scores = []
            for si in self.cfg.tune_splits:
                if si >= len(splits):
                    continue
                pred, _ = self._get_prediction(
                    splits[si]["train"],
                    {"lambda1": l1, "beta": beta, "alpha": alpha},
                    feat_names, ds, splits[si],
                    tag_mode=tag_mode,
                    S_global=S_global,
                )
                m = evaluate(
                    pred, splits[si]["test"], splits[si]["train"],
                    ks=[self.cfg.eval_ks[0]],
                    cold_items=splits[si]["cold_items"],
                )
                scores.append(m.get(f"hr@{self.cfg.eval_ks[0]}", 0))

            avg_score = np.mean(scores) if scores else 0
            if avg_score > best_score:
                best_score = avg_score
                best_hp = {"lambda1": l1, "beta": beta, "alpha": alpha}

        return best_hp

    def _get_prediction(
        self,
        train_csr: sp.csr_matrix,
        hp: Dict,
        feat_names: List[str],
        ds: Dict,
        split: Dict,
        tag_mode: str = "no_tags",
        S_global: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Get prediction matrix for a single split. Returns (prediction, tag_stats)."""

        needs_tags = any("tags" in f for f in feat_names)
        actual_tag_mode = tag_mode if needs_tags else "no_tags"
        
        # Determine if we need to rebuild features per split
        # Must rebuild if:
        # 1. tag_mode requires per-split rebuilding (tags_train_* modes)
        # 2. S_global is None (wasn't built globally)
        # 3. S_global exists but doesn't have required features (e.g., "tags" not in S_global but needed)
        must_rebuild = (
            actual_tag_mode in ("tags_train_only", "tags_train_users", "tags_train_pairs") or
            S_global is None or
            (needs_tags and "tags" not in S_global)
        )

        split_tag_stats = None
        if not must_rebuild and S_global is not None:
            # Use pre-built global similarity matrices
            sim_list = self._get_sims(feat_names, S_global)
        else:
            # Rebuild features for this specific split
            self.log(f"    Rebuilding features for split (tag_mode={actual_tag_mode})...")
            encoded = build_feature_matrices(
                ds["dfs"], ds["item2idx"], ds["idx2item"], ds["n_items"],
                tag_mode=actual_tag_mode,
                train_csr=train_csr if actual_tag_mode in ("tags_train_only", "tags_train_users", "tags_train_pairs") else None,
                user2idx=ds["user2idx"] if actual_tag_mode in ("tags_train_only", "tags_train_users", "tags_train_pairs") else None,
                min_count=self.cfg.min_feature_count,
            )
            
            # Validation: ensure tags are built when needed
            if needs_tags:
                assert "tags" in encoded, (
                    f"Tags required (feat_names includes 'tags') but tag matrix not built! "
                    f"tag_mode={actual_tag_mode}, available features: {list(encoded.keys())}"
                )
                assert encoded["tags"].shape[0] == ds["n_items"], (
                    f"Tag matrix shape mismatch: {encoded['tags'].shape[0]} != {ds['n_items']}"
                )
                if hasattr(encoded["tags"], "nnz"):
                    assert encoded["tags"].nnz > 0, "Tag matrix is empty!"
                else:
                    assert np.any(encoded["tags"] > 0), "Tag matrix is empty!"
                self.log(f"      Tags matrix: shape={encoded['tags'].shape}, "
                        f"nnz={encoded['tags'].nnz if hasattr(encoded['tags'], 'nnz') else np.count_nonzero(encoded['tags'])}")
            
            # Log all feature shapes
            for name, mat in encoded.items():
                if name != "_tag_stats":
                    shape_str = f"{mat.shape}"
                    nnz_str = f", nnz={mat.nnz}" if hasattr(mat, "nnz") else f", nonzeros={np.count_nonzero(mat)}"
                    self.log(f"      {name}: {shape_str}{nnz_str}")
            
            # Extract tag stats if available
            if "_tag_stats" in encoded:
                split_tag_stats = encoded.pop("_tag_stats")
            
            S = build_similarity_matrices(
                encoded,
                shrinkage=self.cfg.default_shrinkage,
                per_feature_shrinkage=self.cfg.per_feature_shrinkage,
                cross_pairs=self.cfg.cross_pairs,
                topk=self.cfg.topk,
                topk_features=self.cfg.topk_features,
            )
            
            # Validation: ensure similarity matrices include all required features
            missing = [f for f in feat_names if f not in S and not any(f in sname for sname in S.keys())]
            if missing:
                raise KeyError(
                    f"Similarity matrices missing required features: {missing}. "
                    f"Available: {list(S.keys())}, requested: {feat_names}"
                )
            
            sim_list = self._get_sims(feat_names, S)

        X = train_csr.toarray().astype(np.float64)
        XtX = X.T @ X

        # Weight learning
        wm = self.cfg.weight_method
        wkw = self.cfg.weight_kwargs
        if wm == "nnls":
            coefs, XG = learn_weights_nnls(X, sim_list, **wkw)
        elif wm == "ridge":
            coefs, XG = learn_weights_ridge(X, sim_list, **wkw)
        else:
            coefs, XG = learn_weights_regression(X, sim_list, **wkw)

        Xtilde = sum(c * xg for c, xg in zip(coefs, XG))
        B = ease_aligned(
            X, Xtilde,
            lambda1=hp["lambda1"],
            beta=hp["beta"],
            alpha=hp["alpha"],
            dr_percentile=self.cfg.dr_percentile,
            XtX=XtX,
        )
        return X @ B, split_tag_stats

    def _get_sims(self, feat_names: List[str], S: Dict) -> List[np.ndarray]:
        """Get similarity matrices for requested features."""
        sims = []
        for name in feat_names:
            if name in S:
                sims.append(S[name])
            else:
                # Check cross-feature names
                found = False
                for sname in S:
                    if name in sname:
                        sims.append(S[sname])
                        found = True
                        break
                if not found:
                    raise KeyError(
                        f"Similarity matrix '{name}' not found. "
                        f"Available: {list(S.keys())}"
                    )
        return sims

    def _save_results(self, all_results: Dict):
        """Save all results to disk."""
        out = Path(self.cfg.output_dir)

        # 1) Per-config summary CSV
        rows = []
        for name, res in all_results.items():
            row = {"config": name, "tag_mode": res["tag_mode"]}
            row.update(res["best_hp"])
            row.update(res["avg_metrics"])
            row.update({f"{k}_std": v for k, v in res["std_metrics"].items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

        # 2) Per-split metrics CSV
        split_rows = []
        for name, res in all_results.items():
            for si, m in enumerate(res["per_split_metrics"]):
                split_rows.append({"config": name, "split": si, **m})
        pd.DataFrame(split_rows).to_csv(out / "per_split_metrics.csv", index=False)

        # 2.5) Tag statistics
        tag_stats_rows = []
        for name, res in all_results.items():
            if res.get("tag_stats"):
                tag_stats_rows.append({
                    "config": name,
                    "tag_mode": res["tag_mode"],
                    **res["tag_stats"]
                })
        if tag_stats_rows:
            pd.DataFrame(tag_stats_rows).to_csv(out / "tag_stats.csv", index=False)

        # 3) Best config JSON
        best_name = max(
            all_results,
            key=lambda k: all_results[k]["avg_metrics"].get("hr@10", 0)
        )
        best = all_results[best_name]
        best_json = {
            "config_name": best_name,
            "best_hp": best["best_hp"],
            "avg_metrics": best["avg_metrics"],
            "std_metrics": best["std_metrics"],
            "tag_mode": best["tag_mode"],
            "feature_names": best["feature_names"],
        }
        if "bootstrap_ci" in best:
            best_json["bootstrap_ci"] = best["bootstrap_ci"]
        with open(out / "best_config.json", "w") as f:
            json.dump(best_json, f, indent=2, default=float)

        # 4) Config used
        self.cfg.save(str(out / "config_used.json"))

        # 5) Log
        with open(out / "run.log", "w") as f:
            f.write("\n".join(self.log_lines))

        self.log(f"\nResults saved to {out}/")
