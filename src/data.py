"""
src/data.py — Data loading and preprocessing (fully sparse).
"""
import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HETREC_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"


def download_hetrec(data_dir: str) -> Path:
    """Download and extract HetRec 2011 dataset if not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files are already extracted (either in subdir or directly in data_dir)
    raw_dir = data_dir / "hetrec2011-movielens-2k-v2"
    if raw_dir.exists():
        return raw_dir
    
    # Check if files were extracted directly to data_dir
    ratings_file = data_dir / "user_ratedmovies-timestamps.dat"
    if ratings_file.exists():
        return data_dir
    
    zip_path = data_dir / "hetrec2011-movielens-2k-v2.zip"
    print(f"Downloading HetRec 2011 to {zip_path}...")
    urllib.request.urlretrieve(HETREC_URL, str(zip_path))
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(data_dir))
    
    # Check where files ended up after extraction
    if raw_dir.exists():
        print(f"Extracted to {raw_dir}")
        return raw_dir
    elif ratings_file.exists():
        print(f"Extracted to {data_dir}")
        return data_dir
    else:
        raise RuntimeError(f"Extraction failed - files not found in {raw_dir} or {data_dir}")


def read_dat(raw_dir: Path, filename: str) -> pd.DataFrame:
    """Read a tab-separated .dat file with latin-1 encoding."""
    path = raw_dir / filename
    if not path.exists():
        return pd.DataFrame()
    with open(str(path), "r", encoding="latin-1") as f:
        text = f.read()
    lines = text.strip().split("\n")
    rows = [line.replace("\r", "").split("\t") for line in lines]
    return pd.DataFrame(rows[1:], columns=rows[0])


def load_all_dataframes(raw_dir: Path) -> dict:
    """Load all raw dataframes from the HetRec directory."""
    dfs = {}
    dfs["movies"] = read_dat(raw_dir, "movies.dat")
    dfs["ratings"] = read_dat(raw_dir, "user_ratedmovies-timestamps.dat")
    dfs["genres"] = read_dat(raw_dir, "movie_genres.dat")
    dfs["actors"] = read_dat(raw_dir, "movie_actors.dat")
    dfs["directors"] = read_dat(raw_dir, "movie_directors.dat")
    dfs["countries"] = read_dat(raw_dir, "movie_countries.dat")
    dfs["tags"] = read_dat(raw_dir, "movie_tags.dat")
    dfs["locations"] = read_dat(raw_dir, "movie_locations.dat")
    return dfs


def binarize_and_kcore(
    df_ratings: pd.DataFrame,
    threshold: float = 3.0,
    min_user: int = 5,
    min_item: int = 5,
):
    """
    Binarize ratings >= threshold, then iterative k-core filtering.
    Ensures unique (user,item) pairs (keeps max rating if duplicates exist).
    Returns: URM_all (sparse CSR), user2idx, item2idx, idx2item, df_filtered.
    """
    df = df_ratings.copy()
    df["userID"] = df["userID"].astype(str)
    df["movieID"] = df["movieID"].astype(str)
    df["rating"] = df["rating"].astype(float)
    df = df[df["rating"] >= threshold].copy()

    # Ensure unique (user,item) pairs: keep max rating if duplicates exist
    n_before = len(df)
    df = df.groupby(["userID", "movieID"])["rating"].max().reset_index()
    n_after = len(df)
    if n_before != n_after:
        print(f"  Removed {n_before - n_after} duplicate (user,item) pairs (kept max rating)")

    # k-core
    for _ in range(200):
        n = len(df)
        ic = df["movieID"].value_counts()
        df = df[df["movieID"].isin(ic[ic >= min_item].index)]
        uc = df["userID"].value_counts()
        df = df[df["userID"].isin(uc[uc >= min_user].index)]
        if len(df) == n:
            break

    users = sorted(df["userID"].unique())
    items = sorted(df["movieID"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {m: i for i, m in enumerate(items)}
    idx2item = {i: m for m, i in item2idx.items()}

    # Verify uniqueness
    assert len(df) == len(df.groupby(["userID", "movieID"])), (
        "Duplicate (user,item) pairs remain after deduplication!"
    )

    rows = np.array([user2idx[u] for u in df["userID"]])
    cols = np.array([item2idx[m] for m in df["movieID"]])
    URM = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(users), len(items)),
    )
    return URM, user2idx, item2idx, idx2item, df


def load_dataset(data_dir: str = "hetrec_data", threshold: float = 3.0):
    """
    Full pipeline: download → load → binarize → k-core.
    Returns dict with all needed objects.
    """
    raw_dir = download_hetrec(data_dir)
    dfs = load_all_dataframes(raw_dir)
    URM, user2idx, item2idx, idx2item, df_filtered = binarize_and_kcore(
        dfs["ratings"], threshold=threshold
    )
    n_u, n_i = URM.shape
    density = URM.nnz / (n_u * n_i) * 100
    print(
        f"Dataset: {n_u} users × {n_i} items, "
        f"{URM.nnz} interactions, density={density:.2f}%"
    )
    return {
        "URM": URM,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "df_filtered": df_filtered,
        "dfs": dfs,
        "n_users": n_u,
        "n_items": n_i,
    }
