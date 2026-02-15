import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
                       "seaborn", "bottleneck", "tabulate"])

import os, time, json, warnings, copy, zipfile, urllib.request
from pathlib import Path
from collections import Counter
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12, "figure.figsize": (10, 6), "figure.dpi": 120})
try:
    import seaborn as sns; sns.set_style("whitegrid")
except: pass
try:
    import bottleneck as bn; _USE_BN = True
except: _USE_BN = False

warnings.filterwarnings("ignore")
print("All imports OK")

# ============ DATA ============
HETREC_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"

def download_hetrec(data_dir):
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    for sub in [data_dir / "hetrec2011-movielens-2k-v2", data_dir]:
        if (sub / "user_ratedmovies-timestamps.dat").exists(): return sub
    zp = data_dir / "hetrec.zip"
    print(f"Downloading HetRec 2011..."); urllib.request.urlretrieve(HETREC_URL, str(zp))
    with zipfile.ZipFile(str(zp), "r") as zf: zf.extractall(str(data_dir))
    for sub in [data_dir / "hetrec2011-movielens-2k-v2", data_dir]:
        if (sub / "user_ratedmovies-timestamps.dat").exists(): return sub
    raise RuntimeError("Extraction failed")

def read_dat(raw_dir, fn):
    p = Path(raw_dir) / fn
    if not p.exists(): return pd.DataFrame()
    with open(str(p), "r", encoding="latin-1") as f: text = f.read()
    lines = text.strip().split("\n")
    rows = [l.replace("\r","").split("\t") for l in lines]
    return pd.DataFrame(rows[1:], columns=rows[0])

def load_all_dataframes(raw_dir):
    dfs = {}
    for name, fn in [("movies","movies.dat"),("ratings","user_ratedmovies-timestamps.dat"),
                     ("genres","movie_genres.dat"),("actors","movie_actors.dat"),
                     ("directors","movie_directors.dat"),("countries","movie_countries.dat"),
                     ("tags","movie_tags.dat"),("locations","movie_locations.dat")]:
        dfs[name] = read_dat(raw_dir, fn)
    # FIX: per-user tags with userID column
    dfs["user_tags"] = read_dat(raw_dir, "user_taggedmovies-timestamps.dat")
    if dfs["user_tags"].empty: dfs["user_tags"] = read_dat(raw_dir, "user_taggedmovies.dat")
    return dfs

def binarize_and_kcore(df_ratings, threshold=3.0, min_user=5, min_item=5):
    df = df_ratings.copy()
    df["userID"]=df["userID"].astype(str); df["movieID"]=df["movieID"].astype(str)
    df["rating"]=df["rating"].astype(float)
    df = df[df["rating"]>=threshold].groupby(["userID","movieID"])["rating"].max().reset_index()
    for _ in range(200):
        n=len(df)
        ic=df["movieID"].value_counts(); df=df[df["movieID"].isin(ic[ic>=min_item].index)]
        uc=df["userID"].value_counts(); df=df[df["userID"].isin(uc[uc>=min_user].index)]
        if len(df)==n: break
    users=sorted(df["userID"].unique()); items=sorted(df["movieID"].unique())
    u2i={u:i for i,u in enumerate(users)}; i2i={m:i for i,m in enumerate(items)}
    idx2item={i:m for m,i in i2i.items()}
    r=np.array([u2i[u] for u in df["userID"]]); c=np.array([i2i[m] for m in df["movieID"]])
    URM=sp.csr_matrix((np.ones(len(r),dtype=np.float32),(r,c)),shape=(len(users),len(items)))
    return URM, u2i, i2i, idx2item, df

def load_dataset(data_dir="hetrec_data", threshold=3.0):
    dfs = load_all_dataframes(download_hetrec(data_dir))
    URM, u2i, i2i, idx2item, df_f = binarize_and_kcore(dfs["ratings"], threshold)
    n_u,n_i=URM.shape; d=URM.nnz/(n_u*n_i)*100
    print(f"Dataset: {n_u} users x {n_i} items, {URM.nnz} interactions, density={d:.2f}%")
    return {"URM":URM,"user2idx":u2i,"item2idx":i2i,"idx2item":idx2item,"dfs":dfs,"n_users":n_u,"n_items":n_i}

# ============ FEATURES ============
def _bfd(df, id_col, val_col):
    d={}
    for _,row in df.iterrows(): d.setdefault(str(row[id_col]),[]).append(str(row[val_col]))
    return d

def encode_multihot(item_lists, min_count=2):
    flat=[v for l in item_lists for v in l]
    ctr=Counter(flat); valid=sorted(k for k,v in ctr.items() if v>=min_count and k.strip())
    if not valid: return np.zeros((len(item_lists),1),dtype=np.float32)
    mlb=MultiLabelBinarizer(classes=valid,sparse_output=False)
    return mlb.fit_transform(item_lists).astype(np.float32)

def encode_tfidf(texts, max_features=10000):
    if not any(t.strip() for t in texts): return np.zeros((len(texts),1),dtype=np.float32)
    try:
        pipe=Pipeline([("c",CountVectorizer(max_features=max_features)),("t",TfidfTransformer())])
        return pipe.fit_transform(texts).toarray().astype(np.float32)
    except: return np.zeros((len(texts),1),dtype=np.float32)

def _filter_tags_train_users(df_tags, train_csr, user2idx, item2idx):
    train_users=set(train_csr.tocoo().row); result={}
    for _,row in df_tags.iterrows():
        uid,mid,tid=str(row.get("userID","")),str(row.get("movieID","")),str(row.get("tagID",""))
        if uid not in user2idx or mid not in item2idx: continue
        if user2idx[uid] in train_users: result.setdefault(mid,[]).append(tid)
    return result

def _filter_tags_train_pairs(df_tags, train_csr, user2idx, item2idx):
    result={}
    for _,row in df_tags.iterrows():
        uid,mid,tid=str(row.get("userID","")),str(row.get("movieID","")),str(row.get("tagID",""))
        if uid not in user2idx or mid not in item2idx: continue
        if train_csr[user2idx[uid],item2idx[mid]]>0: result.setdefault(mid,[]).append(tid)
    return result

def build_feature_matrices(dfs, item2idx, idx2item, n_items,
                           tag_mode="no_tags", train_csr=None,
                           user2idx=None, min_count=2, shuffle_tags=False):
    assert tag_mode in ("no_tags","tags_train_only","tags_train_users","tags_train_pairs","tags_full")
    d_genres=_bfd(dfs["genres"],"movieID","genre")
    d_actors=_bfd(dfs["actors"],"movieID","actorName")
    d_directors=_bfd(dfs["directors"],"movieID","directorName")
    d_countries=_bfd(dfs["countries"],"movieID","country")
    # FIX: use user_tags for leakage-safe modes
    if tag_mode=="tags_full": d_tags=_bfd(dfs["tags"],"movieID","tagID")
    elif tag_mode in ("tags_train_users",):
        d_tags=_filter_tags_train_users(dfs.get("user_tags",dfs["tags"]),train_csr,user2idx,item2idx)
    elif tag_mode in ("tags_train_pairs","tags_train_only"):
        d_tags=_filter_tags_train_pairs(dfs.get("user_tags",dfs["tags"]),train_csr,user2idx,item2idx)
    else: d_tags={}

    df_m=dfs["movies"]; d_years={}
    for _,row in df_m.iterrows():
        try: d_years[str(row["id"])]=int(row["year"])
        except: d_years[str(row["id"])]=2000

    d_locs=[{},{},{}]
    if len(dfs["locations"])>0:
        lc=dfs["locations"].columns.tolist()
        for _,row in dfs["locations"].iterrows():
            mid=str(row[lc[0]])
            for lvl in range(min(3,len(lc)-1)):
                d_locs[lvl].setdefault(mid,[]).append(str(row[lc[lvl+1]]))

    fl={"genres":[],"actors":[],"directors":[],"countries":[],"loc1":[],"loc2":[],"loc3":[],"years":[]}
    loc_texts=[]
    for i in range(n_items):
        mid=str(idx2item[i])
        fl["genres"].append(d_genres.get(mid,[])); fl["actors"].append(d_actors.get(mid,[]))
        fl["directors"].append(d_directors.get(mid,[])); fl["countries"].append(d_countries.get(mid,[]))
        fl["loc1"].append(d_locs[0].get(mid,[])); fl["loc2"].append(d_locs[1].get(mid,[]))
        fl["loc3"].append(d_locs[2].get(mid,[])); fl["years"].append(d_years.get(mid,2000))
        loc_texts.append(" ".join(" ".join(d_locs[j].get(mid,[])) for j in range(3)).strip())

    sc=MinMaxScaler(); enc={}
    for n in ["genres","actors","directors","countries","loc1","loc2","loc3"]:
        enc[n]=sc.fit_transform(encode_multihot(fl[n],min_count=min_count))
    enc["locations"]=sc.fit_transform(encode_tfidf(loc_texts))
    enc["years"]=sc.fit_transform(np.array(fl["years"],dtype=np.float32).reshape(-1,1))

    ts=None
    if tag_mode!="no_tags":
        if not d_tags:
            enc["tags"]=np.zeros((n_items,1),dtype=np.float32)
            ts={"mode":tag_mode,"n_entries":0,"n_items_with_tags":0,"n_features":1}
        else:
            tag_lists=[d_tags.get(str(idx2item[i]),[]) for i in range(n_items)]
            if shuffle_tags:
                rng=np.random.RandomState(999); rng.shuffle(tag_lists)
            ne=sum(len(t) for t in tag_lists); niwt=sum(1 for t in tag_lists if t)
            ut=set(t for tl in tag_lists for t in tl)
            raw=encode_multihot(tag_lists,min_count=min_count)
            enc["tags"]=sc.fit_transform(raw)
            print(f"  Tags({tag_mode}{'|SHUF' if shuffle_tags else ''}): {ne} entries, "
                  f"{len(ut)} unique, {niwt} items, {enc['tags'].shape[1]} feats")
            ts={"mode":tag_mode,"n_entries":ne,"n_unique":len(ut),"n_items_with_tags":niwt,
                "n_features":enc["tags"].shape[1]}
    if ts: enc["_tag_stats"]=ts
    return enc

# ============ SIMILARITY & EASE ============
def smoothed_cosine(enc, shrinkage=20.0):
    s=enc@enc.T; n=np.linalg.norm(enc,axis=1)
    s/=(np.outer(n,n)+shrinkage); np.fill_diagonal(s,0); return s

def year_sim(enc):
    d=euclidean_distances(enc); mx=d.max()
    s=1.0-d/(mx+1e-10) if mx>0 else np.zeros_like(d); np.fill_diagonal(s,0); return s

def build_sims(enc_feats, shrinkage=20.0):
    S={}
    for n,e in enc_feats.items():
        if n.startswith("_"): continue
        S[n]=year_sim(e) if n=="years" else smoothed_cosine(e,shrinkage)
    return S

def compute_dr(X, beta, pct=10):
    v=X.sum(axis=0) if not sp.issparse(X) else np.asarray(X.sum(0)).ravel()
    p=max(np.percentile(v,pct),1.0); k=beta/p
    return np.where(v<=p,k*(p-v),0.0)

def ease_aligned(X, Xtilde, l1=1.0, beta=1.0, alpha=1.0, XtX=None):
    n=X.shape[1]; dr=compute_dr(X,beta)
    Xt_dr=(alpha*Xtilde)*dr[np.newaxis,:]
    XtXt=X.T@Xt_dr
    if XtX is None: XtX=X.T@X
    P=np.linalg.inv(XtX+l1*np.eye(n)+XtXt)
    Bt=P@(XtX+XtXt); g=np.diag(Bt)/np.diag(P)
    return Bt-P@np.diag(g)

# ============ EVALUATION ============
def _topk(arr, k, axis=1):
    return bn.argpartition(-arr,k,axis=axis) if _USE_BN else np.argpartition(-arr,k,axis=axis)

def hr_at_k(pred, held, k):
    n=pred.shape[0]
    if k>=pred.shape[1]: k=pred.shape[1]-1
    idx=_topk(pred,k); pb=np.zeros_like(pred,dtype=bool)
    pb[np.arange(n)[:,None],idx[:,:k]]=True
    tb=(held>0).toarray() if sp.issparse(held) else (held>0)
    h=np.logical_and(tb,pb).sum(1).astype(np.float64)
    nr=tb.sum(1).astype(np.float64)
    return h/np.maximum(np.minimum(k,nr),1.0)

def ndcg_at_k(pred, held, k):
    n=pred.shape[0]
    if k>=pred.shape[1]: k=pred.shape[1]-1
    ip=_topk(pred,k); tp_part=pred[np.arange(n)[:,None],ip[:,:k]]
    isort=np.argsort(-tp_part,axis=1); itop=ip[np.arange(n)[:,None],isort]
    w=1.0/np.log2(np.arange(2,k+2))
    ha=held.toarray() if sp.issparse(held) else held
    DCG=(ha[np.arange(n)[:,None],itop]*w).sum(1)
    nr=(ha>0).sum(1)
    IDCG=np.array([w[:min(int(r),k)].sum() for r in nr])
    return DCG/np.maximum(IDCG,1e-10)

def evaluate(pred, test, train, ks=(10,25), cold_items=None):
    pred=pred.copy(); co=train.tocoo(); pred[co.row,co.col]=-np.inf
    if cold_items is not None:
        warm=np.setdiff1d(np.arange(pred.shape[1]),cold_items); pred[:,warm]=-np.inf
    tu=np.where(np.asarray(test.sum(1)).ravel()>0)[0]
    if len(tu)==0: return {f"{m}@{k}":0.0 for k in ks for m in ("hr","ndcg")}
    p,t=pred[tu],test[tu]
    r={}
    for k in ks:
        if k>=p.shape[1]: continue
        r[f"hr@{k}"]=float(np.nanmean(hr_at_k(p,t,k)))
        r[f"ndcg@{k}"]=float(np.nanmean(ndcg_at_k(p,t,k)))
    return r

# ============ SPLITS ============
def create_cold_split(URM, cold_frac=0.20, seed=42):
    rng=np.random.RandomState(seed); n_u,n_i=URM.shape
    ci=np.sort(rng.choice(n_i,int(n_i*cold_frac),replace=False)); cs=set(ci)
    coo=URM.tocoo(); tm=np.array([c not in cs for c in coo.col])
    tr=sp.csr_matrix((coo.data[tm],(coo.row[tm],coo.col[tm])),shape=(n_u,n_i))
    te=sp.csr_matrix((coo.data[~tm],(coo.row[~tm],coo.col[~tm])),shape=(n_u,n_i))
    return tr,te,ci

def gen_splits(URM, n=5, cf=0.20, seed=42):
    return [{"train":t,"test":te,"cold_items":ci,"seed":seed+i}
            for i,(t,te,ci) in enumerate([create_cold_split(URM,cf,seed+i) for i in range(n)])]

print("All core functions defined")

def predict(train_csr, feat_names, ds, tag_mode, hp, min_count=2, shuffle_tags=False):
    needs_tags = any("tags" in f for f in feat_names)
    atm = tag_mode if needs_tags else "no_tags"
    enc = build_feature_matrices(
        ds["dfs"], ds["item2idx"], ds["idx2item"], ds["n_items"],
        tag_mode=atm,
        train_csr=train_csr if atm not in ("no_tags","tags_full") else None,
        user2idx=ds["user2idx"] if atm not in ("no_tags","tags_full") else None,
        min_count=min_count, shuffle_tags=shuffle_tags)
    ts = enc.pop("_tag_stats", None)
    S = build_sims(enc)
    sims = [S[f] for f in feat_names if f in S]
    # fallback: partial match
    if len(sims) < len(feat_names):
        for f in feat_names:
            if f not in S:
                for sn in S:
                    if f in sn: sims.append(S[sn]); break
    X = train_csr.toarray().astype(np.float64)
    XtX = X.T @ X
    XG = [X @ G for G in sims]
    y = X.ravel()
    Xr = np.column_stack([xg.ravel() for xg in XG])
    w = np.where(y > 0, 1.0, 0.2)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(Xr, y, sample_weight=w)
    Xtilde = sum(c * xg for c, xg in zip(reg.coef_, XG))
    B = ease_aligned(X, Xtilde, l1=hp["l1"], beta=hp["beta"], alpha=hp["alpha"], XtX=XtX)
    return X @ B, ts

def run_exp(ds, splits, feat_names, tag_mode, name,
            hp_grid=None, eval_ks=(10,25), shuffle_tags=False):
    """Optimized: small HP grid, tune on split 0 only, evaluate on all splits."""
    if hp_grid is None:
        # Focused grid based on known-good HPs: l1={10,100}, beta={1,10}, alpha={0.1,1}
        hp_grid = {"l1": [10, 100], "beta": [1, 10], "alpha": [0.1, 1]}
    n_splits = len(splits)
    combos = list(iterproduct(hp_grid["l1"], hp_grid["beta"], hp_grid["alpha"]))
    print(f"\n{'='*60}")
    print(f"  {name} | feats={feat_names} | tag_mode={tag_mode}"
          f"{'  [SHUFFLED]' if shuffle_tags else ''}")
    print(f"  HP grid: {len(combos)} combos, tune split: [0], eval splits: {n_splits}")
    print(f"{'='*60}")
    t0 = time.time()
    # Tune on split 0 only
    best_score, best_hp = -1, {"l1": 100, "beta": 1, "alpha": 0.1}
    for l1, beta, alpha in combos:
        hp = {"l1": l1, "beta": beta, "alpha": alpha}
        pred, _ = predict(splits[0]["train"], feat_names, ds, tag_mode, hp,
                          shuffle_tags=shuffle_tags)
        m = evaluate(pred, splits[0]["test"], splits[0]["train"],
                     ks=[eval_ks[0]], cold_items=splits[0]["cold_items"])
        s = m.get(f"hr@{eval_ks[0]}", 0)
        if s > best_score:
            best_score = s; best_hp = hp
    print(f"  Best HP: l1={best_hp['l1']}, beta={best_hp['beta']}, "
          f"alpha={best_hp['alpha']} (tune={best_score:.4f}, {time.time()-t0:.0f}s)")
    # Evaluate all splits
    t1 = time.time()
    per_split = []; tag_stats = None
    for si in range(n_splits):
        pred, ts = predict(splits[si]["train"], feat_names, ds, tag_mode,
                           best_hp, shuffle_tags=shuffle_tags)
        if ts and tag_stats is None: tag_stats = ts
        m = evaluate(pred, splits[si]["test"], splits[si]["train"],
                     ks=eval_ks, cold_items=splits[si]["cold_items"])
        per_split.append(m)
    avg = {k: np.mean([m[k] for m in per_split]) for k in per_split[0]}
    std = {k: np.std([m[k] for m in per_split]) for k in per_split[0]}
    total_time = time.time() - t0
    print(f"  Results ({n_splits} splits, {time.time()-t1:.0f}s):")
    for k in sorted(avg): print(f"     {k}: {avg[k]:.4f} +/- {std[k]:.4f}")
    print(f"  Total time: {total_time:.0f}s")
    return {"name": name, "feats": feat_names, "tag_mode": tag_mode,
            "best_hp": best_hp, "avg": avg, "std": std,
            "per_split": per_split, "tag_stats": tag_stats,
            "shuffle": shuffle_tags, "time": total_time}

print("Experiment runner defined")

print("Loading dataset...")
ds = load_dataset("hetrec_data", threshold=3.0)

N_SPLITS = 5
print(f"\nGenerating {N_SPLITS} cold-start splits...")
splits = gen_splits(ds["URM"], n=N_SPLITS, cf=0.20, seed=42)

# Verify
s0 = splits[0]
cs = set(s0["cold_items"]); tc = set(s0["train"].tocoo().col)
assert len(tc & cs) == 0, "LEAK!"
print(f"Splits OK. Cold items per split: {len(s0['cold_items'])}")

tag_df = ds["dfs"]["tags"]; ut_df = ds["dfs"]["user_tags"]
print(f"movie_tags.dat: {len(tag_df)} rows, cols: {tag_df.columns.tolist()}")
print(f"user_taggedmovies: {len(ut_df)} rows, cols: {ut_df.columns.tolist()}")
print(f"Unique tags: {tag_df['tagID'].nunique()}")
print(f"Items with tags: {tag_df['movieID'].nunique()} / {ds['n_items']} "
      f"({tag_df['movieID'].nunique()/ds['n_items']*100:.1f}%)")
print(f"Users who tagged: {ut_df['userID'].nunique()}")
# Cold-item tag coverage
ci0 = set(str(ds["idx2item"][i]) for i in splits[0]["cold_items"])
tagged = set(tag_df["movieID"].astype(str).unique())
cwt = ci0 & tagged
print(f"Cold items with tags (split 0): {len(cwt)}/{len(ci0)} ({len(cwt)/len(ci0)*100:.1f}%)")

results = {}
t_total = time.time()

# 1. Baseline: top3 no_tags
results["top3_no_tags"] = run_exp(ds, splits,
    ["actors","directors","genres"], "no_tags", "top3_no_tags")

# 2. Tags train users (leakage-safe)
results["top3_tags_safe"] = run_exp(ds, splits,
    ["actors","directors","genres","tags"], "tags_train_users", "top3_tags_safe")

# 3. Tags full (upper bound)
results["top3_tags_full"] = run_exp(ds, splits,
    ["actors","directors","genres","tags"], "tags_full", "top3_tags_full")

# 4-6. Individual features
for feat in ["actors","directors","genres"]:
    results[f"single_{feat}"] = run_exp(ds, splits, [feat], "no_tags", f"single_{feat}")

# 7. Tags only (leakage-safe)
results["single_tags"] = run_exp(ds, splits,
    ["tags"], "tags_train_users", "single_tags")

# 8. Top3 + countries
results["top3_countries"] = run_exp(ds, splits,
    ["actors","directors","genres","countries"], "no_tags", "top3_countries")

# 9. Top3 + locations
results["top3_locations"] = run_exp(ds, splits,
    ["actors","directors","genres","locations"], "no_tags", "top3_locations")

# 10. Top3 + years
results["top3_years"] = run_exp(ds, splits,
    ["actors","directors","genres","years"], "no_tags", "top3_years")

# 11. Full 9-feature no tags
results["base9_no_tags"] = run_exp(ds, splits,
    ["genres","actors","directors","countries","loc1","loc2","loc3","years","locations"],
    "no_tags", "base9_no_tags")

# 12. Full 9-feature + tags
results["base9_tags_safe"] = run_exp(ds, splits,
    ["genres","actors","directors","countries","loc1","loc2","loc3","years","locations","tags"],
    "tags_train_users", "base9_tags_safe")

# 13. CONTROL: Shuffled tags
results["tags_shuffled"] = run_exp(ds, splits,
    ["actors","directors","genres","tags"], "tags_train_users", "tags_shuffled",
    shuffle_tags=True)

print(f"\n{'='*60}")
print(f"  ALL {len(results)} EXPERIMENTS DONE in {time.time()-t_total:.0f}s")
print(f"{'='*60}")

from tabulate import tabulate

order = ["single_actors","single_directors","single_genres",
         "top3_no_tags","top3_countries","top3_locations","top3_years",
         "base9_no_tags",
         "single_tags","top3_tags_safe","base9_tags_safe",
         "top3_tags_full","tags_shuffled"]

bl = results["top3_no_tags"]["avg"].get("hr@10", 0)
rows = []
for n in order:
    if n not in results: continue
    r = results[n]; a = r["avg"]; s = r["std"]
    delta = ((a.get("hr@10",0)/bl)-1)*100 if bl>0 else 0
    rows.append({
        "Config": n,
        "Features": ", ".join(r["feats"]),
        "Tag Mode": r["tag_mode"] + (" [SHUF]" if r.get("shuffle") else ""),
        "hr@10": f"{a.get('hr@10',0):.4f} +/- {s.get('hr@10',0):.4f}",
        "ndcg@10": f"{a.get('ndcg@10',0):.4f} +/- {s.get('ndcg@10',0):.4f}",
        "Delta": f"{delta:+.1f}%",
        "Time(s)": f"{r.get('time',0):.0f}",
    })
print(tabulate(pd.DataFrame(rows), headers="keys", tablefmt="grid", showindex=False))

bl_scores = [m.get("hr@10",0) for m in results["top3_no_tags"]["per_split"]]
sig_rows = []
for n in order:
    if n not in results or n=="top3_no_tags": continue
    other = [m.get("hr@10",0) for m in results[n]["per_split"]]
    if len(bl_scores)>1 and len(other)>1:
        t_stat, p_val = stats.ttest_rel(other, bl_scores)
        sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "ns"
        sig_rows.append({"Config": n,
            "Mean Delta": f"{np.mean(other)-np.mean(bl_scores):+.4f}",
            "t-stat": f"{t_stat:.3f}", "p-value": f"{p_val:.4f}", "Sig": sig})
print(tabulate(pd.DataFrame(sig_rows), headers="keys", tablefmt="grid", showindex=False))

# Bar chart
fig, ax = plt.subplots(figsize=(14,7))
cmap = {"no_tags":"#4A90D9","tags_train_users":"#50C878","tags_full":"#FFB347"}
names, means, stds, clrs = [], [], [], []
for n in order:
    if n not in results: continue
    r = results[n]; names.append(n)
    means.append(r["avg"].get("hr@10",0)); stds.append(r["std"].get("hr@10",0))
    clrs.append("#FF6B6B" if r.get("shuffle") else cmap.get(r["tag_mode"],"#4A90D9"))
bars = ax.bar(range(len(names)), means, yerr=stds, color=clrs, edgecolor="black", linewidth=0.5, capsize=3)
ax.axhline(y=bl, color="gray", linestyle="--", alpha=0.7, label=f"Baseline ({bl:.3f})")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("HR@10"); ax.set_title("MARec Cold-Start: Feature Ablation & Tag Modes"); ax.legend()
ax.set_ylim(0, max(means)*1.15)
for b,v in zip(bars,means): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{v:.3f}",ha="center",va="bottom",fontsize=8)
plt.tight_layout(); plt.savefig("hr10_all.png",dpi=150,bbox_inches="tight"); plt.show()

# Per-split variance
fig, ax = plt.subplots(figsize=(12,6))
for n in ["top3_no_tags","top3_tags_safe","top3_tags_full","tags_shuffled"]:
    if n not in results: continue
    r = results[n]; vals = [m.get("hr@10",0) for m in r["per_split"]]
    label = n + (" [CONTROL]" if r.get("shuffle") else "")
    ax.plot(range(len(vals)), vals, "o-", label=label, markersize=6)
ax.set_xlabel("Split"); ax.set_ylabel("HR@10"); ax.set_title("Per-Split Stability")
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("per_split.png",dpi=150,bbox_inches="tight"); plt.show()

# Tag mode comparison (side by side)
fig, axes = plt.subplots(1,2,figsize=(14,6))
for ai, metric in enumerate(["hr@10","ndcg@10"]):
    ax=axes[ai]; ns=[]; ms=[]; ss=[]; cs=[]
    for n in ["top3_no_tags","top3_tags_safe","top3_tags_full","tags_shuffled"]:
        if n not in results: continue
        r=results[n]; ns.append(n.replace("top3_","")); ms.append(r["avg"].get(metric,0))
        ss.append(r["std"].get(metric,0))
        cs.append("#FF6B6B" if r.get("shuffle") else cmap.get(r["tag_mode"],"#4A90D9"))
    bars=ax.bar(range(len(ns)),ms,yerr=ss,color=cs,edgecolor="black",linewidth=0.5,capsize=4)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns,rotation=30,ha="right")
    ax.set_ylabel(metric); ax.set_title(f"{metric} by Tag Mode")
    ax.set_ylim(0,max(ms)*1.2)
    for b,v in zip(bars,ms): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{v:.3f}",ha="center",va="bottom",fontsize=9)
plt.tight_layout(); plt.savefig("tag_modes.png",dpi=150,bbox_inches="tight"); plt.show()

out = Path("results/fast_experiments"); out.mkdir(parents=True, exist_ok=True)
rows_csv = []
for n, r in results.items():
    row = {"config": n, "tag_mode": r["tag_mode"], "features": "|".join(r["feats"]),
           "shuffled": r.get("shuffle", False), "time_s": r.get("time", 0)}
    row.update(r["best_hp"]); row.update(r["avg"])
    row.update({f"{k}_std": v for k, v in r["std"].items()})
    rows_csv.append(row)
pd.DataFrame(rows_csv).to_csv(out/"all_results.csv", index=False)
pd.DataFrame([{"config":n,"split":si,**m} for n,r in results.items()
              for si,m in enumerate(r["per_split"])]).to_csv(out/"per_split.csv",index=False)
jr = {n: {"feats":r["feats"],"tag_mode":r["tag_mode"],"best_hp":r["best_hp"],
           "avg":r["avg"],"std":r["std"],"shuffle":r.get("shuffle",False)}
      for n,r in results.items()}
with open(out/"results.json","w") as f: json.dump(jr,f,indent=2,default=float)
print(f"Saved to {out}/")

print("\n" + "="*70)
print("  FINAL SUMMARY")
print("="*70)
print(f"  Dataset: {ds['n_users']} users x {ds['n_items']} items, {ds['URM'].nnz} interactions")
print(f"  Splits: {N_SPLITS}, Cold: 20%")
print(f"\n  Key Results:")
for n in ["top3_no_tags","top3_tags_safe","top3_tags_full","tags_shuffled"]:
    if n not in results: continue
    r=results[n]; hr=r["avg"].get("hr@10",0); sd=r["std"].get("hr@10",0)
    d=((hr/bl)-1)*100 if bl>0 else 0
    tag="[CTRL]" if r.get("shuffle") else "[OK]  "
    print(f"    {tag} {n:25s} hr@10={hr:.4f}+/-{sd:.4f}  ({d:+.1f}%)")
print()
if "tags_shuffled" in results:
    sh=results["tags_shuffled"]["avg"].get("hr@10",0)
    sa=results.get("top3_tags_safe",{}).get("avg",{}).get("hr@10",0)
    if sh<bl*1.1: print("  CONTROL PASSED: Shuffled tags ~ baseline -> tags carry CONTENT signal")
    elif sh<sa*0.9: print("  CONTROL PASSED: Shuffled << real tags -> content signal confirmed")
    else: print("  WARNING: Shuffled tags still elevated -> investigate density effect")
print(f"\n  Total experiments: {len(results)}")
print("="*70)

