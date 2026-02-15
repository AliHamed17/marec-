"""
src/features.py — Feature encoding with leakage-safe tag modes.

Tag modes:
  - "no_tags": Tags excluded entirely.
  - "tags_train_only": Build tag representation using ONLY users/items in train.
  - "tags_full": Use all tag assignments (upper bound, labeled as such).
"""
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


def _build_feature_dict(df, id_col: str, val_col: str) -> Dict[str, List[str]]:
    """Group values by item ID."""
    d = {}
    for _, row in df.iterrows():
        d.setdefault(str(row[id_col]), []).append(str(row[val_col]))
    return d


def encode_multihot(
    item_lists: List[List[str]],
    min_count: int = 2,
) -> np.ndarray:
    """Multi-hot encode a list of label lists with min frequency filtering."""
    flat = [v for lst in item_lists for v in lst]
    counter = Counter(flat)
    valid = sorted(k for k, v in counter.items() if v >= min_count and k.strip())
    if not valid:
        return np.zeros((len(item_lists), 1), dtype=np.float32)
    mlb = MultiLabelBinarizer(classes=valid, sparse_output=False)
    return mlb.fit_transform(item_lists).astype(np.float32)


def encode_tfidf(texts: List[str], max_features: int = 10000) -> np.ndarray:
    """TF-IDF encode a list of text strings. Returns zeros if vocabulary is empty."""
    # Check if there's any non-empty text
    if not any(t.strip() for t in texts):
        return np.zeros((len(texts), 1), dtype=np.float32)
    try:
        pipe = Pipeline([
            ("count", CountVectorizer(max_features=max_features)),
            ("tfidf", TfidfTransformer()),
        ])
        return pipe.fit_transform(texts).toarray().astype(np.float32)
    except ValueError:
        return np.zeros((len(texts), 1), dtype=np.float32)


def encode_years(year_values: List[int]) -> np.ndarray:
    """Encode years as normalized scalar values."""
    arr = np.array(year_values, dtype=np.float32).reshape(-1, 1)
    scaler = MinMaxScaler()
    return scaler.fit_transform(arr)


def _filter_tags_train_pairs(
    df_tags,
    train_csr: sp.csr_matrix,
    user2idx: Dict,
    item2idx: Dict,
) -> Dict[str, List[str]]:
    """
    Build tag dict using ONLY tag assignments from (user,item) pairs
    that exist in the TRAIN set. Strict leakage-safe mode.

    A tag assignment (userID, movieID, tagID) is kept iff:
      - movieID is in item2idx (survived k-core)
      - userID is in user2idx (survived k-core)
      - train[user2idx[userID], item2idx[movieID]] > 0
        (i.e. the user-item pair is in training data)
    """
    result = {}

    for _, row in df_tags.iterrows():
        uid = str(row.get("userID", ""))
        mid = str(row.get("movieID", ""))
        tid = str(row.get("tagID", ""))

        if uid not in user2idx or mid not in item2idx:
            continue

        u_idx = user2idx[uid]
        i_idx = item2idx[mid]

        # Only keep if this user-item pair is in training data
        if train_csr[u_idx, i_idx] > 0:
            result.setdefault(mid, []).append(tid)

    return result


def _filter_tags_train_users(
    df_tags,
    train_csr: sp.csr_matrix,
    user2idx: Dict,
    item2idx: Dict,
) -> Dict[str, List[str]]:
    """
    Build tag dict using tag assignments from users who appear in TRAIN.
    Recommended default leakage-safe mode (less strict than train_pairs).

    A tag assignment (userID, movieID, tagID) is kept iff:
      - movieID is in item2idx (survived k-core)
      - userID is in user2idx (survived k-core)
      - userID appears in train (has at least one training interaction)
    """
    # Get set of users who appear in training
    train_users = set(train_csr.tocoo().row)
    result = {}

    for _, row in df_tags.iterrows():
        uid = str(row.get("userID", ""))
        mid = str(row.get("movieID", ""))
        tid = str(row.get("tagID", ""))

        if uid not in user2idx or mid not in item2idx:
            continue

        u_idx = user2idx[uid]

        # Keep if user appears in training (regardless of which item)
        if u_idx in train_users:
            result.setdefault(mid, []).append(tid)

    return result


def build_feature_matrices(
    dfs: dict,
    item2idx: Dict[str, int],
    idx2item: Dict[int, str],
    n_items: int,
    tag_mode: str = "no_tags",
    train_csr: Optional[sp.csr_matrix] = None,
    user2idx: Optional[Dict] = None,
    min_count: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Build all encoded feature matrices (n_items × n_features_k).

    Args:
        tag_mode: "no_tags" | "tags_train_only" | "tags_train_users" | "tags_train_pairs" | "tags_full"
        train_csr: Required if tag_mode in ("tags_train_only", "tags_train_users", "tags_train_pairs")
        user2idx: Required if tag_mode in ("tags_train_only", "tags_train_users", "tags_train_pairs")

    Returns: dict of {feature_name: encoded_matrix}
    """
    assert tag_mode in ("no_tags", "tags_train_only", "tags_train_users", "tags_train_pairs", "tags_full"), (
        f"Invalid tag_mode: {tag_mode}"
    )
    if tag_mode in ("tags_train_only", "tags_train_users", "tags_train_pairs"):
        assert train_csr is not None and user2idx is not None, (
            f"{tag_mode} requires train_csr and user2idx"
        )

    # Build raw feature dicts
    dict_genres = _build_feature_dict(dfs["genres"], "movieID", "genre")
    dict_actors = _build_feature_dict(dfs["actors"], "movieID", "actorName")
    dict_directors = _build_feature_dict(dfs["directors"], "movieID", "directorName")
    dict_countries = _build_feature_dict(dfs["countries"], "movieID", "country")

    # Tags
    if tag_mode == "tags_full":
        dict_tags = _build_feature_dict(dfs["tags"], "movieID", "tagID")
    elif tag_mode == "tags_train_pairs":
        dict_tags = _filter_tags_train_pairs(
            dfs["tags"], train_csr, user2idx, item2idx
        )
    elif tag_mode == "tags_train_users":
        dict_tags = _filter_tags_train_users(
            dfs["tags"], train_csr, user2idx, item2idx
        )
    elif tag_mode == "tags_train_only":  # Alias for tags_train_pairs (backward compat)
        dict_tags = _filter_tags_train_pairs(
            dfs["tags"], train_csr, user2idx, item2idx
        )
    else:
        dict_tags = {}

    # Years and titles
    df_movies = dfs["movies"]
    dict_years = {}
    title_col = "title" if "title" in df_movies.columns else df_movies.columns[1]
    for _, row in df_movies.iterrows():
        mid = str(row["id"])
        try:
            dict_years[mid] = int(row["year"])
        except (ValueError, KeyError):
            dict_years[mid] = 2000

    # Locations
    dict_locs = [{}, {}, {}]
    if len(dfs["locations"]) > 0:
        lc = dfs["locations"].columns.tolist()
        for _, row in dfs["locations"].iterrows():
            mid = str(row[lc[0]])
            for lvl in range(min(3, len(lc) - 1)):
                dict_locs[lvl].setdefault(mid, []).append(str(row[lc[lvl + 1]]))

    # Build ordered arrays for each item
    feat_lists = {
        "genres": [], "actors": [], "directors": [], "countries": [],
        "loc1": [], "loc2": [], "loc3": [], "years": [],
    }
    loc_texts = []

    for i in range(n_items):
        mid = str(idx2item[i])
        feat_lists["genres"].append(dict_genres.get(mid, []))
        feat_lists["actors"].append(dict_actors.get(mid, []))
        feat_lists["directors"].append(dict_directors.get(mid, []))
        feat_lists["countries"].append(dict_countries.get(mid, []))
        feat_lists["loc1"].append(dict_locs[0].get(mid, []))
        feat_lists["loc2"].append(dict_locs[1].get(mid, []))
        feat_lists["loc3"].append(dict_locs[2].get(mid, []))
        feat_lists["years"].append(dict_years.get(mid, 2000))

        loc_parts = [" ".join(dict_locs[j].get(mid, [])) for j in range(3)]
        loc_texts.append(" ".join(loc_parts).strip())

    # Encode all features
    scaler = MinMaxScaler()
    encoded = {}

    for name in ["genres", "actors", "directors", "countries", "loc1", "loc2", "loc3"]:
        raw = encode_multihot(feat_lists[name], min_count=min_count)
        encoded[name] = scaler.fit_transform(raw)

    # Locations TF-IDF
    encoded["locations"] = scaler.fit_transform(
        encode_tfidf(loc_texts, max_features=10000)
    )

    # Years
    encoded["years"] = encode_years(feat_lists["years"])

    # Tags
    if tag_mode != "no_tags":
        if not dict_tags:
            # No tags found - create empty matrix
            encoded["tags"] = np.zeros((n_items, 1), dtype=np.float32)
            print(f"  Tags ({tag_mode}): WARNING - no tag assignments found, created empty matrix")
            tag_stats = {
                "mode": tag_mode,
                "n_entries": 0,
                "n_unique_tags": 0,
                "n_items_with_tags": 0,
                "avg_tags_per_item": 0.0,
                "n_features": 1,
            }
        else:
            tag_lists = [dict_tags.get(str(idx2item[i]), []) for i in range(n_items)]
            n_tag_entries = sum(len(t) for t in tag_lists)
            n_items_with_tags = sum(1 for t in tag_lists if len(t) > 0)
            unique_tags = set()
            for t_list in tag_lists:
                unique_tags.update(t_list)
            avg_tags_per_item = n_tag_entries / max(n_items_with_tags, 1)
            
            raw_tags = encode_multihot(tag_lists, min_count=min_count)
            encoded["tags"] = scaler.fit_transform(raw_tags)
            print(f"  Tags ({tag_mode}): {n_tag_entries} entries, "
                  f"{len(unique_tags)} unique tags, "
                  f"{n_items_with_tags} items with tags, "
                  f"{avg_tags_per_item:.2f} avg tags/item, "
                  f"{encoded['tags'].shape[1]} features after min_count={min_count}")
            
            # Return statistics for logging
            tag_stats = {
                "mode": tag_mode,
                "n_entries": n_tag_entries,
                "n_unique_tags": len(unique_tags),
                "n_items_with_tags": n_items_with_tags,
                "avg_tags_per_item": avg_tags_per_item,
                "n_features": encoded['tags'].shape[1],
            }
    else:
        # Explicitly do NOT include tags when mode is no_tags
        assert "tags" not in encoded, "Tags should not be in encoded dict when tag_mode=no_tags"
        print(f"  Tags: disabled (mode={tag_mode})")
        tag_stats = None

    # Attach tag stats to encoded dict if available
    if tag_stats:
        encoded["_tag_stats"] = tag_stats
    
    return encoded
