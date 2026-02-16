# MARec Experiment Results Summary

**Date:** 2026-02-15
**Status:** Completed (Manually reconstructed from logs due to blocking UI)

## Key Findings

The experiments confirm that **Tags** provide a massive performance lift in recommendation accuracy (HR@10), while other metadata (Actors, Directors, Genres) provides little to no gain over the baseline (often adding noise).

| Configuration | Description | HR@10 Mean | Delta vs Baseline | Lift (%) |
|---|---|---|---|---|
| **top3_no_tags** | Baseline (Actors, Directors, Genres) | 0.2991 | - | - |
| **top3_tags_safe** | Baseline + Tags (Leakage Safe) | **0.5492** | +0.2501 | **+83.6%** |
| **top3_tags_full** | Baseline + Tags (Full/Aggregated) | **0.5830** | +0.2839 | **+94.9%** |
| **tags_shuffled** | Control (Tags Shuffled) | 0.3004 | +0.0013 | +0.4% (ns) |

## Detailed Results (Reconstructed)

| Config | Features | Mean Delta | t-stat | p-value | Sig |
|---|---|---|---|---|---|
| single_actors | actors | -0.0355 | -2.506 | 0.0664 | ns |
| single_directors | directors | -0.0738 | -10.415 | 0.0005 | *** |
| single_genres | genres | -0.2023 | -12.485 | 0.0002 | *** |
| top3_countries | +countries | +0.0012 | 2.5 | 0.0668 | ns |
| top3_locations | +locations | +0.0002 | 1.122 | 0.3248 | ns |
| top3_years | +years | +0.0015 | 5.416 | 0.0056 | ** |
| base9_no_tags | all metadata (no tags) | +0.0075 | 4.191 | 0.0138 | * |
| **single_tags** | tags only | **0.5071** (+0.208) | 12.796 | 0.0002 | *** |
| **base9_tags_safe** | all metadata + tags | **0.5475** (+0.248) | 31.965 | 0.0000 | *** |

## Conclusion

1. **Tags are dominant:** Adding tags improves HR@10 from ~0.30 to ~0.55-0.58.
2. **Leakage Safety:** The `tags_train_users` mode (safe) performs almost as well as `tags_full`, proving that the gain is real and robust, not just an artifact of train/test leakage.
3. **Valid Signal:** The `tags_shuffled` control result (0.3004) is statistically indistinguishable from baseline, confirming that the improvement comes from the **content** of the tags, not just their presence or density.
4. **Other Metadata:** Actors, Directors, and Genres alone are much weaker predictors. Combining them (Baseline) is better than any single one, but adding more (Countries, Locations, Years) yields diminishing returns (+0.0075 gain at best).
