# =============================================================================
# multi_mouse_discovery.py — Empirical multi-animal habitat discovery
# =============================================================================
# Loads habitats_result.csv from N mice, pools per-cluster centroids (already
# robust-scaled), and discovers common habitat patterns via meta-clustering
# (GMM or K-means).  No prior biological labels are used during discovery;
# user labels are shown only in the validation cross-tabulation (F6).
#
# Usage: python multi_mouse_discovery.py
# =============================================================================

import os
import re
import datetime
import warnings
import threading
import queue

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from sklearn.mixture     import GaussianMixture
from sklearn.cluster     import KMeans
from sklearn.metrics     import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from umap import UMAP as _UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 35
SIZE_FLOOR  = 30    # min voxels per cluster to keep

_PARAM_ORDER = ["DTI-MD", "DTI-AD", "DTI-RD", "DTI-FA", "T2", "T2star", "MTR"]

_NON_PARAM_COLS = {
    "X", "Y", "Slice", "Habitat", "Habitat_label",
    "label_confidence", "label_ambiguous",
}

_GENERIC_FOLDER_NAMES = {
    "v2", "v3", "v4", "v5", "v6", "gmm", "kmeans", "srsc",
    "results", "output", "out", "run",
}

_PALETTE = [
    "#E85D24", "#3B8BD4", "#2EAA72", "#9B59B6", "#F39C12",
    "#1ABC9C", "#E74C3C", "#E91E8C", "#8BC34A", "#607D8B",
    "#FF6B6B", "#4ECDC4",
]

# Per-mouse colours — used consistently in mouse-index figure and radar traces
_MOUSE_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#888888", "#17BECF", "#D4B000",
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
]

def _mouse_col(num: int) -> str:
    """Return a hex colour for 1-based mouse number (cycles after 15)."""
    return _MOUSE_COLORS[(num - 1) % len(_MOUSE_COLORS)]

# Pastel colours for segmentation maps (one per meta-habitat)
_MH_COLORS_SEG = [
    "#FF9999",  # pastel red
    "#99B4FF",  # pastel blue
    "#99FF99",  # pastel green
    "#E099FF",  # pastel purple
    "#FFCC88",  # pastel orange
    "#88FFEE",  # pastel teal
    "#FFB3D9",  # pastel pink
    "#C8FF99",  # pastel lime
    "#FFFF99",  # pastel yellow
    "#FFBB99",  # pastel peach
    "#99EEFF",  # pastel sky
    "#FF99CC",  # pastel rose
]

def _hex_to_rgb(h: str):
    """Convert '#RRGGBB' to (r, g, b) floats in [0, 1]."""
    return int(h[1:3], 16) / 255.0, int(h[3:5], 16) / 255.0, int(h[5:7], 16) / 255.0

_THEME = {
    "bg"         : "#F7F8FA",
    "bg_card"    : "#FFFFFF",
    "bg_input"   : "#FFFFFF",
    "accent"     : "#3A6EA5",
    "text"       : "#1C1C1E",
    "text_sub"   : "#6B7280",
    "separator"  : "#DDE1E7",
}


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def _inject_virtual_md(centroid: dict) -> dict:
    """Add virtual DTI-MD = (AD + 2*RD)/3 when AD+RD present but MD absent."""
    if "DTI-AD" in centroid and "DTI-RD" in centroid and "DTI-MD" not in centroid:
        centroid = dict(centroid)
        centroid["DTI-MD"] = (centroid["DTI-AD"] + 2.0 * centroid["DTI-RD"]) / 3.0
    return centroid


def _mouse_name(csv_path: str) -> str:
    """
    Derive a unique mouse identifier from the CSV path.
    If the parent folder is a generic name (V2, gmm, …), prepend the grandparent.
    """
    parts = os.path.abspath(csv_path).replace("\\", "/").split("/")
    parent = parts[-2] if len(parts) >= 2 else "unknown"
    if parent.lower() in _GENERIC_FOLDER_NAMES and len(parts) >= 3:
        return f"{parts[-3]} ({parent})"
    return parent


def load_mouse_data(csv_path: str) -> dict | None:
    """
    Load one habitats_result.csv.
    Returns a dict with keys: name, csv_path, clusters (list of cluster dicts).
    Clusters with fewer than SIZE_FLOOR voxels are dropped.
    Returns None on error or if no valid clusters remain.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "Habitat" not in df.columns:
        return None

    param_cols = [c for c in df.columns if c not in _NON_PARAM_COLS]
    if not param_cols:
        return None

    # Apply the same DTI resolution as resolve_dti_parameters in loading.py:
    # DTI-MD is a linear combination of AD+RD and biases distances when redundant.
    # Drop it when both AD and RD are present, whether the value is real or virtual.
    _has_ad = 'DTI-AD' in param_cols
    _has_rd = 'DTI-RD' in param_cols
    if _has_ad and _has_rd and 'DTI-MD' in param_cols:
        param_cols = [p for p in param_cols if p != 'DTI-MD']

    total_vox = max(len(df), 1)
    clusters  = []

    for cid, sub in df.groupby("Habitat"):
        n_vox = len(sub)
        if n_vox < SIZE_FLOOR:
            continue

        vals     = sub[param_cols].to_numpy(dtype=float)
        centroid = {p: float(np.nanmean(vals[:, i]))
                    for i, p in enumerate(param_cols)}
        # Only inject virtual MD when AD+RD are both absent (MD is the sole
        # diffusion metric for that mouse).  When AD+RD are present, MD is
        # redundant and should not be added.
        if not (_has_ad and _has_rd):
            centroid = _inject_virtual_md(centroid)

        user_label = "Unknown"
        if "Habitat_label" in sub.columns:
            modes = sub["Habitat_label"].dropna().mode()
            if len(modes):
                user_label = str(modes.iat[0])
                # Strip disambiguation suffixes for cleaner display
                # e.g. "Necrosis (MTR^)" -> "Necrosis"
                # Keep biological sub-types in parentheses (hemorrhagic, cystic…)
                _bio_subtypes = {"hemorrhagic", "proteic", "cystic",
                                 "edematous", "infiltrated",
                                 "hémorragique", "protéique", "kystique",
                                 "oedémateux", "infiltré"}
                import re
                # Strip trailing disambiguation markers and +
                user_label = re.sub(r"\s*\+\s*$", "", user_label).strip()
                # If a parenthetical group is NOT a bio-subtype, strip it
                def _strip_suffix(lbl):
                    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", lbl)
                    if m and m.group(2).lower().strip() not in _bio_subtypes:
                        return _strip_suffix(m.group(1).strip())
                    return lbl
                user_label = _strip_suffix(user_label)

        clusters.append({
            "cluster_id": int(cid),
            "user_label": user_label,
            "n_voxels"  : n_vox,
            "proportion": n_vox / total_vox,
            "centroid"  : centroid,
        })

    if not clusters:
        return None

    return {
        "name"    : _mouse_name(csv_path),
        "csv_path": csv_path,
        "clusters": clusters,
    }


# =============================================================================
# 2. PARAMETER ALIGNMENT
# =============================================================================

def find_common_parameters(mice_list: list) -> list:
    """
    Intersection of centroid keys across all mice, ordered by _PARAM_ORDER.
    Virtual DTI-MD is already injected inside load_mouse_data.
    """
    param_sets = []
    for m in mice_list:
        pset = set()
        for cl in m["clusters"]:
            pset.update(cl["centroid"].keys())
        param_sets.append(pset)

    common  = set.intersection(*param_sets) if param_sets else set()
    ordered = [p for p in _PARAM_ORDER if p in common]
    ordered += sorted(p for p in common if p not in _PARAM_ORDER)
    return ordered


def build_centroid_matrix(mice_list: list, common_params: list):
    """
    Stack all per-cluster centroids into one matrix.
    Returns:
        X        : ndarray (N_total_clusters, n_params)
        row_meta : list of dicts {mouse, cluster_id, user_label, n_voxels, proportion}
    """
    rows, meta = [], []
    for mouse in mice_list:
        for cl in mouse["clusters"]:
            rows.append([cl["centroid"].get(p, 0.0) for p in common_params])
            meta.append({
                "mouse"     : mouse["name"],
                "cluster_id": cl["cluster_id"],
                "user_label": cl["user_label"],
                "n_voxels"  : cl["n_voxels"],
                "proportion": cl["proportion"],
            })
    return np.array(rows, dtype=float), meta


# =============================================================================
# 3. META-CLUSTERING
# =============================================================================

def _silhouette_safe(X, labels) -> float:
    if len(set(labels)) < 2:
        return -1.0
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0


def _fit_predict(X, k: int, n_init: int, seed: int, method: str) -> np.ndarray:
    if method == "gmm":
        model = GaussianMixture(
            n_components=k, n_init=n_init, max_iter=500,
            random_state=seed, covariance_type="full",
        )
        model.fit(X)
        return model.predict(X)
    else:
        model = KMeans(n_clusters=k, n_init=n_init, max_iter=500, random_state=seed)
        return model.fit_predict(X)


def _kneedle_elbow(ks: list, vals: list) -> int:
    """
    Kneedle-style elbow detection for a DECREASING silhouette curve.

    Finds K where the curve transitions from steep to flat by measuring each
    interior point's distance below the straight line that connects the first
    and last (K, score) data points. Endpoints are excluded so the global-max
    K (often K=2 when extreme clusters like Necrosis inflate the score) is
    never selected; the chosen K is always an interior bend in the curve.

    Falls back to the highest score if fewer than 3 candidate Ks exist.
    """
    if len(ks) < 3:
        return ks[int(np.argmax(vals))]

    k_arr = np.array(ks,   dtype=float)
    s_arr = np.array(vals, dtype=float)

    k_norm = (k_arr - k_arr[0]) / max(k_arr[-1] - k_arr[0], 1e-10)
    s_rng  = s_arr.max() - s_arr.min()
    s_norm = (s_arr - s_arr.min()) / max(s_rng, 1e-10)

    # Distance below the diagonal (positive → curve is lower than the line)
    line_at_k = s_norm[0] + k_norm * (s_norm[-1] - s_norm[0])
    below     = line_at_k - s_norm

    below[0]  = -np.inf   # exclude first endpoint
    below[-1] = -np.inf   # exclude last endpoint

    return ks[int(np.argmax(below))]


def find_optimal_meta_k(X, k_range, n_init: int, seed: int,
                         method: str, log,
                         criterion: str = 'elbow') -> tuple[int, dict]:
    """
    Grid-search over k_range using silhouette score.
    criterion='elbow'          : Kneedle elbow — avoids K=2 outlier trap from
                                 extreme clusters inflating the score.
    criterion='max_silhouette' : raw maximum silhouette — picks the K with
                                 the globally best separation score.
    Returns (chosen_k, {k: score}).
    """
    scores = {}
    for k in k_range:
        if k >= len(X):
            break
        labels    = _fit_predict(X, k, n_init, seed, method)
        scores[k] = _silhouette_safe(X, labels)
        log(f"    K={k:2d}  silhouette={scores[k]:.3f}")

    ks    = sorted(scores)
    vals  = [scores[k] for k in ks]
    max_k = max(scores, key=scores.get)

    if criterion == 'max_silhouette':
        best_k = max_k
        log(f"    Max silhouette → K={best_k}  (sil={scores[best_k]:.3f})")
    else:
        best_k = _kneedle_elbow(ks, vals)
        if max_k != best_k:
            log(f"    Elbow → K={best_k}  (max silhouette at K={max_k}: "
                f"skipped — extreme-cluster outlier effect)")
    return best_k, scores


def run_meta_clustering(X, k: int, n_init: int, seed: int,
                         method: str) -> np.ndarray:
    """Final meta-clustering run. Returns integer label array."""
    return _fit_predict(X, k, n_init, seed, method)


# =============================================================================
# 4. STATISTICS
# =============================================================================

def compute_meta_stats(X, meta_labels, row_meta: list,
                        common_params: list, k: int,
                        n_mice_total: int) -> list:
    """
    Returns a list of dicts (one per meta-habitat), sorted by presence_rate desc.
    Adds a display_label field (MH1, MH2, …) after sorting.
    """
    stats = []
    for lbl in range(k):
        mask    = meta_labels == lbl
        members = [row_meta[i] for i in range(len(row_meta)) if mask[i]]
        X_sub   = X[mask]
        mice_in = set(m["mouse"] for m in members)

        stats.append({
            "meta_label"   : lbl,
            "n_clusters"   : int(mask.sum()),
            "n_mice"       : len(mice_in),
            "presence_rate": len(mice_in) / n_mice_total if n_mice_total else 0.0,
            "mean_centroid": X_sub.mean(axis=0) if len(X_sub) else np.zeros(len(common_params)),
            "std_centroid" : X_sub.std(axis=0)  if len(X_sub) else np.zeros(len(common_params)),
            "members"      : members,
            "mice"         : mice_in,
        })

    stats.sort(key=lambda s: -s["presence_rate"])
    for i, s in enumerate(stats):
        s["display_label"] = f"MH{i + 1}"
    return stats


# =============================================================================
# 4b. SECOND PASS — DIRECTIONAL MERGE
# =============================================================================

def compute_cosine_similarity(meta_stats: list, common_params: list,
                               threshold: float = 0.92):
    """
    Compute pairwise cosine similarity between meta-habitat mean centroids.

    Centroids whose L2 norm is < 0.25 IQR units (i.e. near the tumour median
    origin) are flagged as near-zero: cosine is undefined for them and they are
    excluded from merge candidates.

    Returns
    -------
    sim_matrix   : ndarray (n_mh, n_mh)  — cosine similarities; NaN for
                   rows/cols of near-zero centroids (diagonal always 1.0)
    merge_groups : list of lists of int  — indices into meta_stats for each
                   connected component of pairs exceeding `threshold`
    near_zero    : ndarray (n_mh,) bool  — True when centroid is near origin
    """
    centroids = np.array([s["mean_centroid"] for s in meta_stats], dtype=float)
    norms     = np.linalg.norm(centroids, axis=1)
    near_zero = norms < 0.25

    norms_safe = np.where(norms < 1e-8, 1.0, norms)
    c_hat      = centroids / norms_safe[:, None]

    sim = np.clip(c_hat @ c_hat.T, -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)

    n = len(meta_stats)
    for i in range(n):
        if near_zero[i]:
            sim[i, :] = np.nan
            sim[:, i] = np.nan
            sim[i, i] = 1.0   # keep self-reference for display

    # Union-find to collect connected components above threshold
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if near_zero[i]:
            continue
        for j in range(i + 1, n):
            if near_zero[j]:
                continue
            v = sim[i, j]
            if not np.isnan(v) and v >= threshold:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

    groups_raw: dict = {}
    for i in range(n):
        groups_raw.setdefault(_find(i), []).append(i)

    merge_groups = [sorted(g) for g in groups_raw.values() if len(g) > 1]
    return sim, merge_groups, near_zero


def apply_directional_merge(merge_groups: list, meta_stats: list,
                             meta_labels: np.ndarray,
                             X: np.ndarray, row_meta: list,
                             common_params: list, n_mice_total: int):
    """
    Re-label meta_labels by merging each group in merge_groups into its most
    prevalent member (first index in the sorted group, since meta_stats is
    sorted by prevalence descending).

    Returns new_labels (0-based), new_meta_stats (recomputed).
    """
    label_map = {s["meta_label"]: s["meta_label"] for s in meta_stats}
    for group in merge_groups:
        rep = meta_stats[group[0]]["meta_label"]   # most prevalent
        for idx in group[1:]:
            label_map[meta_stats[idx]["meta_label"]] = rep

    raw = np.array([label_map[int(l)] for l in meta_labels])
    unique = sorted(set(raw))
    remap  = {old: new for new, old in enumerate(unique)}
    new_labels = np.array([remap[l] for l in raw])
    new_k      = len(unique)

    new_meta_stats = compute_meta_stats(
        X, new_labels, row_meta, common_params, new_k, n_mice_total)
    return new_labels, new_meta_stats


# =============================================================================
# 4c. LEVEL-2 SUB-CLUSTERING (OPTIONAL)
# =============================================================================

def collect_mh_voxels(mice_list: list, row_meta: list, meta_labels,
                       common_params: list, mh_meta_label: int):
    """
    For a given meta-habitat (identified by its raw meta_label integer), load
    all raw voxels from the original habitats_result.csv files.

    Returns (X_vox, mri_params, spatial_df) where:
        X_vox      : ndarray (N_vox, n_mri_params)  — d_topo_norm excluded
        mri_params : list of parameter names matching X_vox columns
        spatial_df : DataFrame with columns [mouse, X, Y, Slice] for each voxel
    Returns (None, [], None) if nothing found.
    """
    mh_map: dict[str, set] = {}
    for rm, ml in zip(row_meta, meta_labels):
        if int(ml) == mh_meta_label:
            mh_map.setdefault(rm["mouse"], set()).add(rm["cluster_id"])

    mri_params  = [p for p in common_params if p != "d_topo_norm"]
    rows:     list[np.ndarray] = []
    spatials: list[pd.DataFrame] = []

    for mouse in mice_list:
        cids = mh_map.get(mouse["name"])
        if not cids:
            continue
        try:
            df = pd.read_csv(mouse["csv_path"])
        except Exception:
            continue

        mask = df["Habitat"].isin(cids)
        sub  = df[mask]
        if len(sub) == 0:
            continue

        # Feature matrix
        avail = [p for p in mri_params if p in df.columns]
        if avail:
            feat = sub[avail].to_numpy(dtype=float)
            if len(avail) < len(mri_params):
                full = np.zeros((len(feat), len(mri_params)), dtype=float)
                for j, p in enumerate(mri_params):
                    if p in avail:
                        full[:, j] = feat[:, avail.index(p)]
                feat = full
            rows.append(feat)

        # Spatial metadata — keep whatever coordinate columns exist
        sp_cols = [c for c in ("X", "Y", "Slice") if c in df.columns]
        sp = sub[sp_cols].copy()
        sp["mouse"] = mouse["name"]
        spatials.append(sp)

    if not rows:
        return None, [], None

    spatial_df = pd.concat(spatials, ignore_index=True) if spatials else None
    return np.nan_to_num(np.vstack(rows), nan=0.0), mri_params, spatial_df


def run_sub_clustering(mice_list: list, row_meta: list, meta_labels,
                        meta_stats: list, common_params: list,
                        k_sub_max: int = 4,
                        n_init: int = 5,
                        min_voxels: int = 200,
                        log=print) -> dict:
    """
    Level-2 sub-clustering: within each meta-habitat, pool raw voxels from all
    mice (already per-mouse robust z-scored) and fit a BIC-optimal GMM to find
    intra-habitat sub-populations.

    K=1 (no split) is always evaluated as baseline. If BIC selects K=1, the
    meta-habitat is considered homogeneous and no sub-clustering is reported.
    This prevents forcing artificial splits on unimodal distributions.

    k_sub_max : maximum number of sub-clusters to try (K=1 to k_sub_max).
    """
    mri_idx = [i for i, p in enumerate(common_params) if p != "d_topo_norm"]

    results: dict = {}

    for mh_idx, ms in enumerate(meta_stats):
        mh_label = ms["display_label"]
        meta_lbl = ms["meta_label"]

        log(f"[Sub-clust] {mh_label}: collecting voxels…")
        X_vox, mri_params, spatial_df = collect_mh_voxels(
            mice_list, row_meta, meta_labels, common_params, meta_lbl)

        n_vox = 0 if X_vox is None else len(X_vox)

        if X_vox is None or n_vox < min_voxels:
            log(f"[Sub-clust] {mh_label}: SKIP — {n_vox} voxels < {min_voxels} minimum")
            results[mh_idx] = {"skipped": True,
                                "skip_reason": f"{n_vox} voxels < {min_voxels}",
                                "meta_stat": ms}
            continue

        # ── Per-mouse centering: remove inter-mouse normalisation offset ──────
        # Each mouse was robust z-scored against its own tumour median, so within
        # a pooled MH their voxel clouds can be offset from each other.  Centering
        # each mouse's contribution to zero-mean (within this MH) removes that
        # additive offset and lets the GMM find intra-mouse tissue structure.
        if spatial_df is not None and "mouse" in spatial_df.columns:
            mouse_ids   = spatial_df["mouse"].values
            X_for_clust = X_vox.copy().astype(float)
            for m in np.unique(mouse_ids):
                mask_m = mouse_ids == m
                X_for_clust[mask_m] -= X_for_clust[mask_m].mean(axis=0)
            n_mice_here = len(np.unique(mouse_ids))
            log(f"[Sub-clust] {mh_label}: per-mouse centering applied ({n_mice_here} mice)")
        else:
            X_for_clust = X_vox

        # Cap so each sub-cluster has at least 50 voxels on average
        k_cap = min(k_sub_max, n_vox // 50)
        k_cap = max(k_cap, 1)

        log(f"[Sub-clust] {mh_label}: {n_vox} voxels — BIC K=1…{k_cap}")

        best_bic, best_k, best_gmm = np.inf, 1, None
        for k in range(1, k_cap + 1):
            try:
                gmm = GaussianMixture(n_components=k, n_init=(1 if k == 1 else n_init),
                                      covariance_type="full",
                                      reg_covar=1e-4,
                                      random_state=RANDOM_SEED, max_iter=300)
                gmm.fit(X_for_clust)
                bic = gmm.bic(X_for_clust)
                log(f"[Sub-clust]   K={k}  BIC={bic:.1f}")
                if bic < best_bic:
                    best_bic, best_k, best_gmm = bic, k, gmm
            except Exception as exc:
                log(f"[Sub-clust]   K={k} failed: {exc}")

        if best_gmm is None:
            results[mh_idx] = {"skipped": True, "skip_reason": "GMM failed",
                                "meta_stat": ms}
            continue

        # K=1 wins → no meaningful sub-structure
        if best_k == 1:
            log(f"[Sub-clust] {mh_label}: homogeneous — BIC prefers K=1, no split.")
            results[mh_idx] = {"skipped": True,
                                "skip_reason": "BIC prefers K=1 (homogeneous)",
                                "meta_stat": ms}
            continue

        sub_labels = best_gmm.predict(X_for_clust)

        # ── Minimum sub-cluster size filter ───────────────────────────────────
        # Eliminates degenerate micro-clusters (outliers captured as a component).
        # If any cluster is too small, reduce K by 1 and refit until all pass.
        min_sub_size = max(30, int(0.05 * n_vox))
        for _attempt in range(best_k - 1):
            sizes = [(sub_labels == s).sum() for s in range(best_k)]
            if min(sizes) >= min_sub_size:
                break
            best_k -= 1
            log(f"[Sub-clust]   → micro-cluster detected, reducing to K={best_k}")
            if best_k == 1:
                break
            try:
                gmm2 = GaussianMixture(n_components=best_k, n_init=n_init,
                                       covariance_type="full", reg_covar=1e-4,
                                       random_state=RANDOM_SEED, max_iter=300)
                gmm2.fit(X_for_clust)
                sub_labels = gmm2.predict(X_for_clust)
                best_gmm   = gmm2
            except Exception:
                break

        if best_k == 1:
            log(f"[Sub-clust] {mh_label}: homogeneous after size filter.")
            results[mh_idx] = {"skipped": True,
                                "skip_reason": "BIC prefers K=1 (homogeneous)",
                                "meta_stat": ms}
            continue

        log(f"[Sub-clust] {mh_label}: K_sub={best_k} selected (BIC={best_bic:.1f})")

        # Centroids reported in original (non-centred) space for interpretability
        meta_centroid = ms["mean_centroid"][mri_idx]

        sub_stats = []
        for s in range(best_k):
            mask = sub_labels == s
            n_s  = int(mask.sum())
            cen  = X_vox[mask].mean(axis=0)   # original space
            delta = cen - meta_centroid
            sub_stats.append({
                "sub_label"    : s,
                "display_label": f"{mh_label}.{s + 1}",
                "n_voxels"     : n_s,
                "centroid"     : {p: float(cen[j]) for j, p in enumerate(mri_params)},
                "delta"        : {p: float(delta[j]) for j, p in enumerate(mri_params)},
            })
            log(f"[Sub-clust]   {mh_label}.{s + 1}: {n_s:,} voxels")

        results[mh_idx] = {
            "skipped"      : False,
            "meta_stat"    : ms,
            "X"            : X_vox,
            "mri_params"   : mri_params,
            "sub_labels"   : sub_labels,
            "sub_k"        : best_k,
            "sub_stats"    : sub_stats,
            "meta_centroid": meta_centroid,
            "spatial_df"   : spatial_df,
        }

    return results


# =============================================================================
# 5. FIGURES
# =============================================================================

_DIVMAP = LinearSegmentedColormap.from_list(
    "bwr2", ["#2471A3", "#FFFFFF", "#C0392B"], N=256
)


def _ax_style(ax):
    for sp in ax.spines.values():
        sp.set_color("#DDE1E7")
    ax.tick_params(colors="#6B7280", labelsize=8)
    ax.set_facecolor("#FAFBFC")


# --- F1: K selection ---------------------------------------------------------

def fig_k_selection(k_scores: dict, chosen_k: int, out_dir: str,
                     criterion: str = 'elbow') -> str:
    ks    = sorted(k_scores)
    vals  = [k_scores[k] for k in ks]
    max_k = max(k_scores, key=k_scores.get)

    if criterion == 'max_silhouette':
        chosen_label = f"Chosen K = {chosen_k}  (max silhouette,  sil = {k_scores[chosen_k]:.3f})"
        subtitle     = "(max silhouette: K with the globally highest separation score)"
    else:
        chosen_label = f"Chosen K = {chosen_k}  (elbow,  sil = {k_scores[chosen_k]:.3f})"
        subtitle     = "(elbow criterion: K where steep descent gives way to flat plateau)"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, vals, "o-", color="#3A6EA5", linewidth=2, markersize=6)
    ax.axvline(chosen_k, color="#E85D24", linestyle="--", linewidth=2,
               label=chosen_label)
    if max_k != chosen_k:
        not_chosen_lbl = (f"Elbow K = {max_k}  ({k_scores[max_k]:.3f})  — not chosen"
                          if criterion == 'max_silhouette'
                          else f"Max silhouette K = {max_k}  ({k_scores[max_k]:.3f})  — not chosen")
        ax.axvline(max_k, color="#888888", linestyle=":", linewidth=1.2,
                   label=not_chosen_lbl)
    ax.set_xlabel("Number of meta-habitats (K)", fontsize=10)
    ax.set_ylabel("Silhouette score", fontsize=10)
    ax.set_title(
        f"Meta-clustering — K selection\n{subtitle}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    _ax_style(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "f2_k_selection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- F2: PCA scatter ---------------------------------------------------------

def fig_pca_scatter(X, meta_labels, meta_stats: list, out_dir: str) -> str:
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X2  = pca.fit_transform(X)
    ev  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _ax_style(ax)

    for s in meta_stats:
        lbl  = s["meta_label"]
        mask = meta_labels == lbl
        col  = _PALETTE[lbl % len(_PALETTE)]
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=col, alpha=0.72, s=90, zorder=3,
                   label=s["display_label"])
        cx, cy = X2[mask, 0].mean(), X2[mask, 1].mean()
        ax.scatter(cx, cy, marker="*", s=260, color=col,
                   edgecolors="white", linewidths=0.8, zorder=5)

    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)", fontsize=10)
    ax.set_title("PCA of cluster centroids — meta-habitat coloring\n"
                 "★ = meta-habitat archetype (mean centroid)",
                 fontsize=10, fontweight="bold")
    ax.legend(title="Meta-habitat", fontsize=8, title_fontsize=8,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "f3_pca_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- F3: Mean centroid heatmap -----------------------------------------------

def fig_profiles_heatmap(meta_stats: list, common_params: list, out_dir: str) -> str:
    n_mh   = len(meta_stats)
    n_p    = len(common_params)
    data   = np.array([s["mean_centroid"] for s in meta_stats])
    labels = [s["display_label"] for s in meta_stats]
    vmax   = float(np.nanmax(np.abs(data))) or 1.0

    fig, ax = plt.subplots(
        figsize=(max(6, n_p * 0.95 + 1.2), max(2.5, n_mh * 0.65 + 1.5))
    )
    im = ax.imshow(data, cmap=_DIVMAP, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_p))
    ax.set_xticklabels(common_params, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_mh))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        "Mean centroid per meta-habitat  (robust-scaled, 0 = tumour median)",
        fontsize=10, fontweight="bold",
    )
    for i in range(n_mh):
        for j in range(n_p):
            v = data[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(v) > vmax * 0.55 else "black")

    fig.colorbar(im, ax=ax, label="Robust-scaled value (IQR units)", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, "f4_profiles_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- F_cos: Cosine similarity heatmap (2nd pass) -----------------------------

def fig_cosine_heatmap(sim_matrix: np.ndarray, meta_stats: list,
                        near_zero: np.ndarray, merge_groups: list,
                        threshold: float, common_params: list,
                        out_dir: str, merge_mode: str = "suggest") -> str:
    """
    NxN heatmap of cosine similarity between meta-habitat mean centroids.
    Merge-candidate pairs (above threshold) are highlighted with an orange
    border.  Near-zero centroids are shown as N/A.
    """
    n      = len(meta_stats)
    labels = [s["display_label"] for s in meta_stats]

    cmap_cos = LinearSegmentedColormap.from_list(
        "cos_map", ["#2471A3", "#EAECEE", "#C0392B"], N=256
    )

    cell = max(0.80, min(1.3, 9.0 / max(n, 1)))
    fig, ax = plt.subplots(figsize=(n * cell + 2.5, n * cell + 2.8))
    fig.patch.set_facecolor(_THEME["bg"])

    masked = np.ma.masked_invalid(sim_matrix)
    im = ax.imshow(masked, cmap=cmap_cos, vmin=-1.0, vmax=1.0, aspect="auto")

    from matplotlib.patches import Rectangle
    for i in range(n):
        for j in range(n):
            if near_zero[i] or near_zero[j]:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=6, color="#AAAAAA")
                continue
            v = sim_matrix[i, j]
            if np.isnan(v):
                continue
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#888888")
            else:
                txt_col = "white" if abs(v) > 0.55 else "#111111"
                fw      = "bold" if v >= threshold else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_col, fontweight=fw)
                if v >= threshold:
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        lw=2.0, edgecolor="#E85D24",
                        facecolor="none", zorder=5))

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    mode_str = {"suggest": "suggestion only", "auto": "auto-merged"}.get(merge_mode, "")
    ax.set_title(
        f"Directional similarity between meta-habitat centroids  (cosine)\n"
        f"Threshold = {threshold:.2f}  ·  orange-bordered = merge candidates"
        + (f"  [{mode_str}]" if mode_str else ""),
        fontsize=10, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)

    nz_labels = [labels[i] for i in range(n) if near_zero[i]]
    if nz_labels:
        ax.text(0.01, -0.04,
                f"N/A = centroid near origin (cosine undefined): {', '.join(nz_labels)}",
                transform=ax.transAxes, fontsize=7, color="#777777", va="top")

    # Merge candidate summary box
    if merge_groups:
        lines = []
        for g in merge_groups:
            g_labels = [meta_stats[i]["display_label"] for i in g]
            shared   = np.nanmean([meta_stats[i]["mean_centroid"] for i in g], axis=0)
            n_p      = len(common_params)
            pos_i    = sorted([i for i in range(n_p) if shared[i] >  0.15],
                               key=lambda x: -shared[x])[:3]
            neg_i    = sorted([i for i in range(n_p) if shared[i] < -0.15],
                               key=lambda x:  shared[x])[:3]
            sig_pos  = "  ".join(f"{common_params[i]}↑" for i in pos_i)
            sig_neg  = "  ".join(f"{common_params[i]}↓" for i in neg_i)
            sig      = "  /  ".join(filter(None, [sig_pos, sig_neg])) or "mixed"
            pairs    = [
                f"{meta_stats[g[a]]['display_label']}↔"
                f"{meta_stats[g[b]]['display_label']}: "
                f"{sim_matrix[g[a], g[b]]:.2f}"
                for a in range(len(g)) for b in range(a + 1, len(g))
            ]
            lines.append(
                f"  {' + '.join(g_labels)}"
                f"  [{sig}]"
                f"  ({', '.join(pairs)})"
            )
        txt = "Merge candidates (same direction, different amplitude):\n" + "\n".join(lines)
        fig.text(
            0.01, -0.01, txt,
            fontsize=7.5, va="top", fontfamily="monospace", color="#1C1C1E",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7",
                      alpha=0.92, edgecolor="#F39C12", linewidth=1.2),
        )
    else:
        fig.text(
            0.5, -0.01,
            f"No merge candidates found above threshold {threshold:.2f}.",
            ha="center", fontsize=8, color="#6B7280",
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "f_cosine.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- F4: Presence matrix (mice × meta-habitats) ------------------------------

def fig_presence_matrix(mice_list: list, meta_labels, row_meta: list,
                         meta_stats: list, out_dir: str) -> str:
    mouse_names = [m["name"] for m in mice_list]
    mh_labels   = [s["display_label"] for s in meta_stats]
    mh_orig_ids = [s["meta_label"]    for s in meta_stats]
    n_mice      = len(mouse_names)
    n_mh        = len(meta_stats)

    mouse_idx = {name: i for i, name in enumerate(mouse_names)}
    mh_idx    = {ml: j for j, ml in enumerate(mh_orig_ids)}
    matrix    = np.zeros((n_mice, n_mh))

    for i, rm in enumerate(row_meta):
        mi = mouse_idx.get(rm["mouse"])
        mj = mh_idx.get(int(meta_labels[i]))
        if mi is not None and mj is not None:
            matrix[mi, mj] = max(matrix[mi, mj], rm["proportion"])

    cell_h  = max(0.3, min(0.6, 12.0 / max(n_mice, 1)))
    fig_h   = max(4.0, n_mice * cell_h + 2.0)
    fig, ax = plt.subplots(figsize=(max(5, n_mh * 1.1 + 1.5), fig_h))

    im = ax.imshow(matrix, cmap="YlOrRd",
                   vmin=0, vmax=max(matrix.max(), 0.01), aspect="auto")

    ax.set_xticks(range(n_mh))
    ax.set_xticklabels(mh_labels, fontsize=9)
    ax.set_yticks(range(n_mice))
    fs_y = max(5, min(9, int(180 / max(n_mice, 1))))
    ax.set_yticklabels(mouse_names, fontsize=fs_y)
    ax.set_xlabel("Meta-habitat", fontsize=10)
    ax.set_title(
        "Habitat presence per mouse  (colour = max cluster proportion)",
        fontsize=10, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Max cluster proportion of tumour volume",
                 shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, "f4_presence_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- F5: Per-meta-habitat bar profiles ---------------------------------------

def fig_meta_profiles_bar(meta_stats: list, common_params: list,
                            n_mice_total: int, out_dir: str) -> str:
    n_mh  = len(meta_stats)
    n_p   = len(common_params)
    ncols = min(3, n_mh)
    nrows = int(np.ceil(n_mh / ncols))
    x     = np.arange(n_p)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.2),
        squeeze=False,
    )

    for idx, s in enumerate(meta_stats):
        r, c  = divmod(idx, ncols)
        ax    = axes[r][c]
        color = _PALETTE[s["meta_label"] % len(_PALETTE)]

        ax.bar(x, s["mean_centroid"], color=color, alpha=0.75, zorder=3)
        ax.errorbar(x, s["mean_centroid"], yerr=s["std_centroid"],
                    fmt="none", color="#444", linewidth=1.2, capsize=3, zorder=4)
        ax.axhline(0, color="#AAA", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(common_params, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Robust-scaled value", fontsize=8)
        ax.set_title(
            f"{s['display_label']}  —  {s['n_mice']}/{n_mice_total} mice,"
            f" {s['n_clusters']} clusters",
            fontsize=9, fontweight="bold",
        )
        _ax_style(ax)

    for idx in range(n_mh, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Per-meta-habitat MRI profiles  (mean ± std, robust-scaled)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "f5_meta_profiles_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- F6: Cross-tabulation user_label × meta-habitat (validation) -------------

def fig_crosstab(row_meta: list, meta_labels, meta_stats: list,
                  out_dir: str,
                  merge_groups: list | None = None) -> str:
    """
    Counts how many clusters of each user label end up in each meta-habitat.
    This is purely for post-hoc validation; labels are NOT used in discovery.
    merge_groups : list of index lists (into meta_stats) — columns in the same
                   group get the same highlight color. Only pass in suggest mode.
    """
    _MERGE_PALETTE = ["#E67E22", "#27AE60", "#8E44AD", "#C0392B", "#2980B9"]

    disp_map = {s["meta_label"]: s["display_label"] for s in meta_stats}
    all_user_labels = sorted(set(rm["user_label"] for rm in row_meta))
    mh_display      = [s["display_label"] for s in meta_stats]

    n_ul = len(all_user_labels)
    n_mh = len(meta_stats)
    ul_idx = {ul: i for i, ul in enumerate(all_user_labels)}
    mh_idx = {s["meta_label"]: j for j, s in enumerate(meta_stats)}

    matrix = np.zeros((n_ul, n_mh), dtype=int)
    for i, rm in enumerate(row_meta):
        r = ul_idx.get(rm["user_label"])
        c = mh_idx.get(int(meta_labels[i]))
        if r is not None and c is not None:
            matrix[r, c] += 1

    row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    matrix_norm = matrix / row_sums

    # Build col → group-index map
    col_group: dict[int, int] = {}
    if merge_groups:
        for gi, group in enumerate(merge_groups):
            for idx in group:
                if 0 <= idx < n_mh:
                    col_group[idx] = gi

    # Extra bottom margin when a legend is needed
    bottom_margin = 0.18 if col_group else 0.05
    cell_h  = max(0.35, min(0.7, 10.0 / max(n_ul, 1)))
    fig_h   = max(3.5, n_ul * cell_h + 1.8)
    fig, ax = plt.subplots(figsize=(max(5, n_mh * 1.0 + 1.5), fig_h))
    fig.subplots_adjust(bottom=bottom_margin)

    # Colored column backgrounds (drawn before the heatmap so it stays behind)
    for c, gi in col_group.items():
        color = _MERGE_PALETTE[gi % len(_MERGE_PALETTE)]
        ax.axvspan(c - 0.5, c + 0.5, color=color, alpha=0.12, zorder=0)

    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_mh))
    ax.set_xticklabels(mh_display, fontsize=9)
    ax.set_yticks(range(n_ul))
    fs_y = max(5, min(9, int(170 / max(n_ul, 1))))
    ax.set_yticklabels(all_user_labels, fontsize=fs_y)
    ax.set_xlabel("Empirical meta-habitat", fontsize=10)
    ax.set_ylabel("User label", fontsize=10)
    ax.set_title(
        "Validation: user label → meta-habitat assignment\n"
        "(row-normalised; labels were NOT used during discovery)",
        fontsize=10, fontweight="bold",
    )

    # Color the x-tick labels of merge-candidate columns
    ax.figure.canvas.draw()
    for c, lbl in enumerate(ax.get_xticklabels()):
        if c in col_group:
            color = _MERGE_PALETTE[col_group[c] % len(_MERGE_PALETTE)]
            lbl.set_color(color)
            lbl.set_fontweight("bold")

    for r in range(n_ul):
        for c in range(n_mh):
            v = matrix_norm[r, c]
            raw = matrix[r, c]
            ax.text(c, r, f"{raw}\n({v*100:.0f}%)",
                    ha="center", va="center", fontsize=6,
                    color="white" if v > 0.55 else "#333")

    fig.colorbar(im, ax=ax, label="Fraction of clusters from that user label",
                 shrink=0.8)

    # Legend for merge groups
    if col_group:
        from matplotlib.patches import Patch
        seen = sorted(set(col_group.values()))
        handles = []
        for gi in seen:
            color = _MERGE_PALETTE[gi % len(_MERGE_PALETTE)]
            members = [meta_stats[i]["display_label"]
                       for i in merge_groups[gi] if i < n_mh]
            handles.append(Patch(facecolor=color, alpha=0.5,
                                 label=f"Merge candidate {gi + 1}: {' + '.join(members)}"))
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), ncol=min(len(handles), 3),
                  fontsize=7, title="2nd pass — directional merge (suggest)",
                  title_fontsize=7, framealpha=0.85)

    path = os.path.join(out_dir, "f8_crosstab.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# 5b. NEW FIGURES: mouse index, summary table + radar charts
# =============================================================================

# --- F0: Mouse index ----------------------------------------------------------

def fig_mouse_index(mice_list: list, out_dir: str) -> str:
    """
    Table: number → mouse name + path context + per-mouse K.
    Splits into 2 side-by-side sub-tables when n > 10 so the figure stays
    readable regardless of the number of selected mice.
    """
    n = len(mice_list)

    def _path_ctx(csv_path: str, n_parts: int = 3) -> str:
        """Return last n_parts directory components of the CSV path."""
        if not csv_path:
            return ""
        parts = os.path.normpath(csv_path).replace("\\", "/").split("/")
        dirs  = parts[:-1]          # drop the CSV filename
        tail  = dirs[-n_parts:] if len(dirs) >= n_parts else dirs
        prefix = ".." + "/" if len(dirs) > n_parts else ""
        return prefix + "/".join(tail)

    def _tint(hex_col: str) -> tuple:
        r = int(hex_col[1:3], 16) / 255
        g = int(hex_col[3:5], 16) / 255
        b = int(hex_col[5:7], 16) / 255
        return (0.78 + 0.22 * r, 0.78 + 0.22 * g, 0.78 + 0.22 * b)

    # ── Split into 1 or 2 groups ──────────────────────────────────────────
    if n > 10:
        mid    = (n + 1) // 2
        groups = [mice_list[:mid], mice_list[mid:]]
        starts = [1, mid + 1]
    else:
        groups = [mice_list]
        starts = [1]

    n_grps    = len(groups)
    max_rows  = max(len(g) for g in groups)   # rows in the tallest column

    # ── Figure sizing ─────────────────────────────────────────────────────
    # Desired physical row height (inches), shrinks slightly for large groups
    row_h_in  = max(0.28, min(0.42, 4.5 / (max_rows + 1)))
    fig_h     = (max_rows + 1) * row_h_in + 1.0   # +1 for suptitle
    fig_w     = 9.5 * n_grps

    # Normalised row height so the table fits inside its axes (≤ 0.88 total)
    row_h_norm = 0.88 / (max_rows + 1)
    hdr_h_norm = row_h_norm * 1.15

    # Font: shrinks gently as the group gets larger
    fs = max(7, min(9, int(120 / (max_rows + 1)) + 5))

    # ── Build figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_grps, figsize=(fig_w, fig_h))
    if n_grps == 1:
        axes = [axes]
    fig.patch.set_facecolor(_THEME["bg"])

    n_cols   = 4
    col_w    = [0.06, 0.28, 0.57, 0.09]
    hdr_clrs = [(0.14, 0.22, 0.35)] * n_cols

    for ax, grp, start in zip(axes, groups, starts):
        ax.axis("off")
        cell_text   = []
        cell_colors = []
        for i, m in enumerate(grp):
            num  = start + i
            tint = _tint(_mouse_col(num))
            cell_text.append([
                f"M{num}",
                m["name"],
                _path_ctx(m.get("csv_path", "")),
                str(len(m["clusters"])),
            ])
            cell_colors.append([tint] * n_cols)

        tbl = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=["#", "Mouse name", "Full path  (…/parent/folder)", "K"],
            colColours=hdr_clrs,
            colWidths=col_w,
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fs)

        for j in range(n_cols):
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
            tbl[(0, j)].set_height(hdr_h_norm)
        for i in range(1, len(grp) + 1):
            tbl[(i, 0)].set_text_props(fontweight="bold")
            for j in range(n_cols):
                tbl[(i, j)].set_height(row_h_norm)

    fig.suptitle(
        "Mouse index  —  M# labels are used throughout this report\n"
        "(cluster compositions, radar chart traces)",
        fontsize=10, fontweight="bold", y=0.99, color="#1C1C1E",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "f0_mouse_index.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# --- Fd: Segmentation maps (6 mice per page, 3×2 grid, contour borders) ------

_SEG_MICE_PER_PAGE = 6
_SEG_MICE_PER_ROW  = 3   # columns of mice per page
_SEG_MAX_SLICES    = 4   # max slices shown per mouse cell


def _draw_slice(ax, df_sl, mouse_name, cluster_to_meta, mh_color,
                sl_val, all_meta_seen, show_slice_title=True):
    """
    Fill one axes with the 2D habitat map for a single slice.
    Uses ax.contour for habitat boundaries (vector, very thin).
    """
    xi_raw = df_sl["X"].values.astype(int)
    yi_raw = df_sl["Y"].values.astype(int)
    x_min, y_min = xi_raw.min(), yi_raw.min()
    w = int(xi_raw.max() - x_min + 1)
    h = int(yi_raw.max() - y_min + 1)
    xi = xi_raw - x_min
    yi = yi_raw - y_min

    cids = df_sl["Habitat"].values.astype(int)
    vml  = np.array([cluster_to_meta.get((mouse_name, c), -1) for c in cids])

    # RGBA image — light grey background (areas not in tumor = transparent)
    img = np.full((h, w, 4), [0.93, 0.93, 0.93, 1.0], dtype=float)

    # Habitat grid: -1 = outside tumor
    hab_grid = np.full((h, w), -1, dtype=float)
    hab_grid[yi, xi] = cids.astype(float)

    # Fill each meta-habitat colour
    for ml_val in np.unique(vml):
        if ml_val < 0:
            continue
        all_meta_seen.add(int(ml_val))
        r, g, b = _hex_to_rgb(mh_color.get(int(ml_val), "#CCCCCC"))
        mask = vml == ml_val
        img[yi[mask], xi[mask], 0] = r
        img[yi[mask], xi[mask], 1] = g
        img[yi[mask], xi[mask], 2] = b

    ax.imshow(img, interpolation="nearest", origin="upper",
              aspect="equal")

    # ── Sub-pixel habitat boundaries via LineCollection ──────────────────
    # Draw vector segments exactly on the edge between adjacent pixels
    # (at half-integer coordinates) — one unique line per border, no doubling.
    if len(np.unique(hab_grid)) > 1:
        from matplotlib.collections import LineCollection
        segs = []
        # Horizontal edges: boundary between row r and row r+1
        rs, cs = np.where(hab_grid[:-1, :] != hab_grid[1:, :])
        if len(rs):
            h_segs = np.stack([
                np.column_stack([cs - 0.5, rs + 0.5]),
                np.column_stack([cs + 0.5, rs + 0.5]),
            ], axis=1)
            segs.append(h_segs)
        # Vertical edges: boundary between col c and col c+1
        rs, cs = np.where(hab_grid[:, :-1] != hab_grid[:, 1:])
        if len(rs):
            v_segs = np.stack([
                np.column_stack([cs + 0.5, rs - 0.5]),
                np.column_stack([cs + 0.5, rs + 0.5]),
            ], axis=1)
            segs.append(v_segs)
        if segs:
            ax.add_collection(LineCollection(
                np.concatenate(segs, axis=0),
                colors=[(0, 0, 0, 0.45)], linewidths=0.4, antialiased=False))

    # ── Cluster numbers (1-based) ─────────────────────────────────────────
    hab_int = np.full((h, w), -1, dtype=int)
    hab_int[yi, xi] = cids
    for cid in sorted(set(cids)):
        cell_mask = hab_int == int(cid)
        n_pix = int(cell_mask.sum())
        if n_pix < 3:
            continue
        ys_c, xs_c = np.where(cell_mask)
        cy, cx = float(ys_c.mean()), float(xs_c.mean())
        fs = max(5, min(10, int(n_pix ** 0.40)))
        ax.text(cx, cy, str(int(cid) + 1),
                ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.10",
                          facecolor="white", alpha=0.50,
                          edgecolor="none"))

    if show_slice_title:
        ax.set_title(f"S{int(sl_val)}", fontsize=7, pad=2)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#BBBBBB")


def fig_segmentation_maps(mice_list: list, row_meta: list, meta_labels,
                           meta_stats: list, out_dir: str) -> list:
    """
    Segmentation maps: up to 6 mice per page in a 3-column × 2-row grid.
    Each mouse cell uses GridSpecFromSubplotSpec for its individual slices.
    Mouse label = 'M{num}' only (index table gives the full mapping).
    Habitat boundaries = thin vector contour lines (linewidths=0.35).
    Returns list of (path, caption) for PDF inclusion.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    cluster_to_meta = {(rm["mouse"], rm["cluster_id"]): int(meta_labels[i])
                       for i, rm in enumerate(row_meta)}
    mh_color = {s["meta_label"]: _MH_COLORS_SEG[j % len(_MH_COLORS_SEG)]
                for j, s in enumerate(meta_stats)}
    mh_label = {s["meta_label"]: s["display_label"] for s in meta_stats}
    mouse_num = {m["name"]: i + 1 for i, m in enumerate(mice_list)}

    mice_frames: dict = {}
    for mouse in mice_list:
        try:
            df = pd.read_csv(mouse["csv_path"])
            if "X" not in df.columns or "Habitat" not in df.columns:
                continue
            slices = sorted(df["Slice"].unique()) if "Slice" in df.columns else [0]
            mice_frames[mouse["name"]] = (df, slices[:_SEG_MAX_SLICES])
        except Exception:
            continue

    valid_mice = [m for m in mice_list if m["name"] in mice_frames]
    if not valid_mice:
        return []

    n_total = len(valid_mice)
    n_pages = max(1, (n_total + _SEG_MICE_PER_PAGE - 1) // _SEG_MICE_PER_PAGE)
    results  = []

    for page in range(n_pages):
        batch      = valid_mice[page * _SEG_MICE_PER_PAGE:
                                (page + 1) * _SEG_MICE_PER_PAGE]
        n_batch    = len(batch)
        n_out_cols = min(_SEG_MICE_PER_ROW, n_batch)
        n_out_rows = (n_batch + n_out_cols - 1) // n_out_cols

        # Each mouse cell is 4.8" wide × 4.2" tall
        fig_w = n_out_cols * 4.8 + 0.5
        fig_h = n_out_rows * 4.2 + 1.5

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(_THEME["bg"])
        outer = GridSpec(n_out_rows, n_out_cols, figure=fig,
                         hspace=0.35, wspace=0.10)

        all_meta_seen: set = set()

        for m_idx, mouse in enumerate(batch):
            out_row = m_idx // n_out_cols
            out_col = m_idx % n_out_cols
            df, slices = mice_frames[mouse["name"]]
            n_sl   = len(slices)
            mnum   = mouse_num[mouse["name"]]
            mname  = mouse["name"]

            # Invisible spanning axes → provides the centered "M{num}" title
            ax_lbl = fig.add_subplot(outer[out_row, out_col])
            ax_lbl.axis("off")
            ax_lbl.set_title(f"M{mnum}", fontsize=12, fontweight="bold",
                             pad=6, color="#1C1C1E")

            # Inner grid: 1 row × n_sl columns (one per slice)
            inner = GridSpecFromSubplotSpec(
                1, n_sl,
                subplot_spec=outer[out_row, out_col],
                wspace=0.05,
            )
            show_sl = (n_sl > 1)   # only label slices when >1

            for s_idx, sl_val in enumerate(slices):
                ax = fig.add_subplot(inner[0, s_idx])
                df_sl = (df[df["Slice"] == sl_val]
                         if "Slice" in df.columns else df)
                if len(df_sl) == 0:
                    ax.axis("off")
                    continue
                _draw_slice(ax, df_sl, mname, cluster_to_meta,
                            mh_color, sl_val, all_meta_seen,
                            show_slice_title=show_sl)

        # ── Legend ────────────────────────────────────────────────────────
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1,
                           facecolor=mh_color.get(ml, "#CCC"),
                           edgecolor="#555", linewidth=0.7,
                           label=mh_label.get(ml, f"MH{ml + 1}"))
            for ml in sorted(all_meta_seen)
        ]
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=min(len(legend_handles), 8),
                bbox_to_anchor=(0.5, 0.0),
                fontsize=8,
                title="Colour = meta-habitat  ·  numbers = cluster id (1-based)",
                title_fontsize=7,
                framealpha=0.95,
            )

        page_str = f"  (page {page + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(
            "Segmentation maps — colour = meta-habitat  ·  "
            f"contours = cluster boundaries{page_str}",
            fontsize=9.5, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])

        fname = f"f9_seg_p{page + 1:02d}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        m_start = mouse_num[batch[0]["name"]]
        m_end   = mouse_num[batch[-1]["name"]]
        mrange  = (f"M{m_start}–M{m_end}" if m_start != m_end
                   else f"M{m_start}")
        results.append((fpath,
                        f"F9 — Segmentation maps  ({mrange}){page_str}"))

    return results


# ── Radar helpers ─────────────────────────────────────────────────────────────

_RADAR_VMIN = -2.5   # IQR units for inner radar bound
_RADAR_VMAX =  2.5   # IQR units for outer radar bound


def _radar_setup(ax, params: list) -> list:
    """Configure a polar ax for radar charts. Returns the angle list."""
    n_p    = len(params)
    angles = np.linspace(0, 2 * np.pi, n_p, endpoint=False).tolist()
    vspan  = _RADAR_VMAX - _RADAR_VMIN          # display range (5.0)

    ax.set_ylim(0, vspan)
    ax.set_theta_direction(-1)                   # clockwise
    ax.set_theta_zero_location("N")              # first param at top
    ax.set_facecolor("#FAFBFC")

    # Grid rings labelled in IQR units
    tick_iqr = [-2, -1, 0, 1, 2]
    tick_r   = [v - _RADAR_VMIN for v in tick_iqr]
    ax.set_yticks(tick_r)
    ax.set_yticklabels([f"{v:+.0f}" for v in tick_iqr], fontsize=5, color="#999")
    ax.tick_params(axis="y", pad=1)

    # Dashed reference circle at 0 IQR (tumour median)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta, [-_RADAR_VMIN] * 300, color="#999", lw=0.7, ls="--", zorder=1)

    ax.set_xticks(angles)
    ax.set_xticklabels(params, fontsize=8)
    try:
        ax.spines["polar"].set_color("#DDE1E7")
    except Exception:
        pass
    return angles


def _radar_trace(ax, angles: list, values_iqr, color,
                 label=None, alpha=0.25, lw=2):
    """Plot one radar profile on a pre-configured polar ax."""
    disp = (np.clip(np.asarray(values_iqr, dtype=float),
                    _RADAR_VMIN, _RADAR_VMAX) - _RADAR_VMIN).tolist()
    a = angles + [angles[0]]
    d = disp   + [disp[0]]
    ax.plot(a, d, color=color, linewidth=lw, zorder=3, label=label)
    ax.fill(a, d, color=color, alpha=alpha, zorder=2)


# --- Fa: Summary recap table --------------------------------------------------

def fig_summary_table(meta_stats: list, common_params: list,
                       n_mice_total: int, out_dir: str) -> str:
    """
    Matplotlib table: rows = meta-habitats, columns = info cols + one per param.
    Parameter cells are colour-coded by mean centroid value
    (red = elevated, blue = decreased, white = near median).
    """
    n_mh = len(meta_stats)
    n_p  = len(common_params)

    meta_hdrs = ["Meta-hab.", "Mice", "Clusters", "Prev. %"]
    all_hdrs  = meta_hdrs + common_params

    all_c = np.array([s["mean_centroid"] for s in meta_stats])
    vmax  = max(float(np.abs(all_c).max()), 0.5)
    cmap  = plt.cm.RdBu_r

    def _cell_color(v):
        r, g, b, _ = cmap((v + vmax) / (2 * vmax))
        return (0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b)

    cell_text   = []
    cell_colors = []

    for s in meta_stats:
        mu = s["mean_centroid"]
        sd = s["std_centroid"]
        row_t = [
            s["display_label"],
            f"{s['n_mice']}/{n_mice_total}",
            str(s["n_clusters"]),
            f"{s['presence_rate'] * 100:.0f}%",
        ] + [f"{mu[i]:+.2f} ±{sd[i]:.2f}" for i in range(n_p)]
        row_c = [(0.93, 0.95, 0.98)] * len(meta_hdrs) + [
            _cell_color(mu[i]) for i in range(n_p)
        ]
        cell_text.append(row_t)
        cell_colors.append(row_c)

    meta_w  = [1.1, 0.7, 0.85, 0.75]
    param_w = [1.15] * n_p
    all_w   = meta_w + param_w
    total_w = sum(all_w)
    col_w   = [w / total_w for w in all_w]

    fig_w = max(9, total_w * 0.9 + 0.5)
    fig_h = max(2.5, n_mh * 0.60 + 1.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor(_THEME["bg"])

    hdr_colors = [(0.14, 0.22, 0.35)] * len(meta_hdrs) + \
                 [(0.10, 0.38, 0.52)] * n_p

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=all_hdrs,
        colColours=hdr_colors,
        colWidths=col_w,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for j in range(len(all_hdrs)):
        cell = tbl[(0, j)]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.15)
    for i in range(1, n_mh + 1):
        for j in range(len(all_hdrs)):
            tbl[(i, j)].set_height(0.13)

    ax.set_title(
        "Meta-habitat summary   (MRI values in IQR units — 0 = tumour median)\n"
        "Red = elevated  ·  Blue = decreased  ·  White = near tumour median",
        fontsize=9, fontweight="bold", pad=8, loc="left", color="#1C1C1E",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "f1_summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Fb: Per-meta-habitat radar — all individual cluster traces ---------------

def fig_radar_per_mh(meta_stats: list, common_params: list,
                      n_mice_total: int,
                      X, meta_labels, row_meta: list,
                      out_dir: str) -> str:
    """
    One radar subplot per meta-habitat.
    Each subplot shows EVERY constituent cluster as a thin trace (coloured by
    mouse number) plus the group mean as a thick black outline, so outlier
    clusters that stand out from the rest are immediately visible.
    """
    n_mh  = len(meta_stats)
    ncols = min(3, n_mh)
    nrows = int(np.ceil(n_mh / ncols))

    # Build mouse → number map from row_meta insertion order
    mouse_num: dict[str, int] = {}
    for rm in row_meta:
        if rm["mouse"] not in mouse_num:
            mouse_num[rm["mouse"]] = len(mouse_num) + 1
    n_mice_seen = len(mouse_num)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 4.5),
        subplot_kw=dict(polar=True),
        squeeze=False,
    )
    fig.patch.set_facecolor(_THEME["bg"])

    for idx, s in enumerate(meta_stats):
        r, c    = divmod(idx, ncols)
        ax      = axes[r][c]
        mh_lbl  = s["meta_label"]
        mh_col  = _PALETTE[mh_lbl % len(_PALETTE)]
        angles  = _radar_setup(ax, common_params)

        # ── Individual cluster traces (thin, coloured by mouse) ────────────
        for i, rm in enumerate(row_meta):
            if int(meta_labels[i]) != mh_lbl:
                continue
            m_col = _mouse_col(mouse_num[rm["mouse"]])
            _radar_trace(ax, angles, X[i], m_col, alpha=0.12, lw=1.1)

        # ── Mean profile (thick black outline, no fill) ────────────────────
        disp_mean = (
            np.clip(s["mean_centroid"], _RADAR_VMIN, _RADAR_VMAX) - _RADAR_VMIN
        ).tolist()
        a_closed = angles + [angles[0]]
        d_closed = disp_mean + [disp_mean[0]]
        ax.plot(a_closed, d_closed, color="#111111", linewidth=2.8, zorder=6)

        ax.set_title(
            f"{s['display_label']}  ·  {s['n_mice']}/{n_mice_total} mice"
            f"  ·  {s['n_clusters']} clusters",
            fontsize=8.5, fontweight="bold", pad=14, color=mh_col,
        )

    for idx in range(n_mh, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # ── Shared legend (mouse colours + mean) ──────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=_mouse_col(num), lw=1.8,
                   label=f"M{num}")
        for _, num in sorted(mouse_num.items(), key=lambda kv: kv[1])
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#111111", lw=2.8, label="Mean")
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(n_mice_seen + 1, 8),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8,
        title="Colour = mouse (M#)  ·  Bold black = group mean",
        title_fontsize=7.5,
        framealpha=0.9,
    )

    fig.suptitle(
        "Individual cluster profiles per meta-habitat\n"
        "(IQR units — dashed ring = tumour median  [0]"
        " — outliers visible as traces far from group)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f6_radar_per_mh.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.join(out_dir, "f6_radar_per_mh.png")


# --- Fc: Combined radar chart (all meta-habitats overlaid) --------------------

def fig_radar_combined(meta_stats: list, common_params: list,
                        out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(_THEME["bg"])

    angles = _radar_setup(ax, common_params)

    for s in meta_stats:
        color = _PALETTE[s["meta_label"] % len(_PALETTE)]
        _radar_trace(ax, angles, s["mean_centroid"], color,
                     label=s["display_label"], alpha=0.15)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.45, 1.15),
        fontsize=9,
        title="Meta-habitat",
        title_fontsize=9,
        framealpha=0.9,
    )
    ax.set_title(
        "Combined radar — mean MRI profiles per meta-habitat\n"
        "(IQR units — dashed circle = tumour median  [0])",
        fontsize=10, fontweight="bold", pad=22,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "f7_radar_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# 6. CSV EXPORT
# =============================================================================

def export_assignment_csv(row_meta: list, meta_labels, meta_stats: list,
                           out_dir: str) -> str:
    disp_map = {s["meta_label"]: s["display_label"] for s in meta_stats}
    rows = []
    for i, rm in enumerate(row_meta):
        ml = int(meta_labels[i])
        rows.append({
            "mouse"        : rm["mouse"],
            "cluster_id"   : rm["cluster_id"],
            "user_label"   : rm["user_label"],
            "n_voxels"     : rm["n_voxels"],
            "proportion"   : round(rm["proportion"], 4),
            "meta_habitat" : disp_map.get(ml, f"MH?"),
            "meta_label_idx": ml,
        })
    path = os.path.join(out_dir, "meta_habitat_assignments.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# =============================================================================
# 7. COHORT TUMOR PROFILE
# =============================================================================

def fig_cohort_tumor_profile(mice_list: list, out_dir: str) -> str | None:
    """
    Heatmap of raw tumor medians and IQR across the cohort.

    Reads scaling_params.csv from the output folder of each mouse
    (parent directory of csv_path).  Produces two panels side by side:
      Left  — raw median per parameter  (column z-scored for colour,
               raw value annotated in each cell)
      Right — IQR per parameter         (same layout)

    Interpretation:
      • Left panel: where does each mouse sit in absolute parameter space?
        A row that is consistently blue = globally low signal (e.g. very
        necrotic tumour); red = globally elevated.
      • Right panel: how heterogeneous is each parameter within the tumour?
        High IQR = large intra-tumoral variability for that parameter.
    """
    import pandas as pd

    # ── Collect scaling_params.csv for each mouse ─────────────────────────
    records_med: dict = {}   # mouse_name -> {param: median_raw}
    records_iqr: dict = {}   # mouse_name -> {param: iqr_raw}

    for mouse in mice_list:
        csv_path   = mouse.get("csv_path", "")
        mouse_name = mouse["name"]
        scale_path = os.path.join(os.path.dirname(csv_path), "scaling_params.csv")
        if not os.path.exists(scale_path):
            continue
        try:
            df_sp = pd.read_csv(scale_path)
            records_med[mouse_name] = dict(zip(df_sp["parameter"], df_sp["median_raw"]))
            records_iqr[mouse_name] = dict(zip(df_sp["parameter"], df_sp["iqr_raw"]))
        except Exception:
            continue

    if len(records_med) < 2:
        return None   # not enough data

    # ── Build matrices ────────────────────────────────────────────────────
    all_params = sorted({p for r in records_med.values() for p in r})
    mouse_names = list(records_med.keys())

    mat_med = np.array([[records_med[m].get(p, np.nan) for p in all_params]
                        for m in mouse_names])
    mat_iqr = np.array([[records_iqr[m].get(p, np.nan) for p in all_params]
                        for m in mouse_names])

    # Column z-score for colour (so cross-parameter scale differences don't dominate)
    def col_zscore(mat: np.ndarray) -> np.ndarray:
        out = np.full_like(mat, np.nan)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) < 2:
                continue
            mu, sd = valid.mean(), valid.std()
            if sd < 1e-12:
                out[:, j] = 0.0
            else:
                out[:, j] = (col - mu) / sd
        return out

    zmed = col_zscore(mat_med)
    ziqr = col_zscore(mat_iqr)

    # ── Figure ────────────────────────────────────────────────────────────
    n_rows = len(mouse_names)
    n_cols = len(all_params)
    cell_w, cell_h = 1.5, 0.55
    fig_w = max(10, n_cols * cell_w * 2 + 3.5)
    fig_h = max(4, n_rows * cell_h + 2.5)

    fig, (ax_med, ax_iqr) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.35},
    )
    fig.patch.set_facecolor("#F7F8FA")

    _DIVMAP = LinearSegmentedColormap.from_list(
        "_cohort_div",
        ["#1565C0", "#FFFFFF", "#C62828"], N=256,
    )

    def _draw_panel(ax, zmat, rawmat, title, unit_label):
        im = ax.imshow(zmat, cmap=_DIVMAP, vmin=-2.5, vmax=2.5, aspect="auto")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(all_params, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(mouse_names, fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)

        # Annotate with raw values
        for i in range(n_rows):
            for j in range(n_cols):
                v = rawmat[i, j]
                z = zmat[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.3g}"
                color = "white" if abs(z) > 1.4 else "#222"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=color)

        fig.colorbar(im, ax=ax, label=f"Column z-score ({unit_label})",
                     shrink=0.65, pad=0.02)

    _draw_panel(ax_med, zmed, mat_med,
                "Tumor median per parameter\n(raw, unscaled values)",
                "raw units")
    _draw_panel(ax_iqr, ziqr, mat_iqr,
                "Intra-tumor IQR per parameter\n(raw, unscaled — measure of heterogeneity)",
                "raw units")

    fig.suptitle(
        "Cohort tumor profile  —  absolute position in parameter space\n"
        "Colour = column z-score across mice  ·  Cell = raw value\n"
        "Blue = low signal  ·  Red = elevated signal",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = os.path.join(out_dir, "f00_cohort_tumor_profile.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# 8. PDF REPORT  (was 7)
# =============================================================================

def generate_pdf(meta_stats: list, common_params: list,
                  n_mice: int, chosen_k: int, method: str,
                  k_scores: dict, out_dir: str,
                  seg_figs: list | None = None,
                  criterion: str = 'elbow',
                  merge_mode: str = 'off',
                  cosine_threshold: float = 0.92,
                  merge_group_labels: list | None = None,
                  original_k: int | None = None) -> str:
    """
    Build the discovery PDF.
    seg_figs : optional list of (abs_path, caption) tuples — per-mouse
               segmentation maps appended after the standard figures.
    """
    _FIG_TITLES = [
        ("f00_cohort_tumor_profile.png",
                                  "F0.0 — Cohort tumor profile  (raw medians & IQR per mouse × parameter)"),
        ("f0_mouse_index.png",    "F0 — Mouse index  (M# labels used throughout the report)"),
        ("f1_summary_table.png",  "F1 — Meta-habitat summary table  (mean ± std per parameter)"),
        ("f2_k_selection.png",    f"F2 — K selection  "
                                   f"({'elbow criterion' if criterion == 'elbow' else 'max silhouette'}"
                                   f" on silhouette curve)"),
        ("f3_pca_scatter.png",    "F3 — PCA of cluster centroids  (meta-habitat colouring)"),
        ("f4_profiles_heatmap.png","F4 — Mean centroid heatmap per meta-habitat"),
        ("f5_meta_profiles_bar.png","F5 — Per-meta-habitat MRI bar profiles  (mean ± std)"),
        ("f6_radar_per_mh.png",   "F6 — Radar profiles — one chart per meta-habitat"),
        ("f7_radar_combined.png", "F7 — Radar profiles — all meta-habitats overlaid"),
        ("f8_crosstab.png",       "F8 — Validation: user label → meta-habitat  (post-hoc)"),
        ("f_cosine.png",          "F_cos — Directional similarity between meta-habitats  (2nd pass)"),
    ]

    path = os.path.join(out_dir, "multi_mouse_discovery_report.pdf")
    with PdfPages(path) as pdf:

        # ── Title page ───────────────────────────────────────────────────
        folder_name = os.path.basename(os.path.abspath(out_dir))
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ax.text(0.5, 0.97, folder_name, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top", ha="center",
                color="#1C1C1E")

        summary_lines = "".join(
            f"  {s['display_label']:5s}  {s['n_mice']:3d}/{n_mice} mice   "
            f"{s['n_clusters']:3d} clusters   "
            f"presence {s['presence_rate']*100:5.1f}%\n"
            for s in meta_stats
        )
        k_line = "  " + "  ".join(f"K={k}: {v:.3f}" for k, v in sorted(k_scores.items()))

        max_k_pdf    = max(k_scores, key=k_scores.get)
        criterion_lbl = ("silhouette elbow criterion"
                         if criterion == 'elbow'
                         else "max silhouette")
        if criterion == 'elbow' and max_k_pdf != chosen_k:
            elbow_note = (
                f"  (max silhouette at K={max_k_pdf}: "
                f"{k_scores[max_k_pdf]:.3f} — not chosen;\n"
                f"   elbow avoids K=2 outlier trap from extreme clusters)\n"
            )
        elif criterion == 'max_silhouette' and max_k_pdf != chosen_k:
            elbow_note = (
                f"  (elbow would have chosen K={max_k_pdf})\n"
            )
        else:
            elbow_note = ""

        # 2nd pass section
        if merge_mode == 'off':
            merge_note = "2nd pass      : disabled\n"
        else:
            merge_note = (
                f"2nd pass      : {merge_mode}  (cosine ≥ {cosine_threshold})\n"
            )
            if merge_group_labels:
                for g in merge_group_labels:
                    merge_note += f"  → Merge candidate: {' + '.join(g)}\n"
                if merge_mode == 'auto' and original_k is not None:
                    merge_note += (
                        f"  → K reduced: {original_k} → {len(meta_stats)}"
                        f"  (auto-merge applied)\n"
                    )
            else:
                merge_note += "  → No candidates above threshold\n"

        has_cosine_fig = merge_mode != 'off'
        text = (
            f"Multi-Mouse Habitat Discovery\n"
            f"{'=' * 45}\n\n"
            f"Date            : {now}\n"
            f"Mice analysed   : {n_mice}\n"
            f"Method          : {method.upper()}\n"
            f"K selection     : {criterion_lbl}\n"
            f"Chosen K        : {chosen_k}\n"
            f"Parameters      : {', '.join(common_params)}\n"
            f"{merge_note}\n"
            f"Discovery is purely data-driven — no biological\n"
            f"labels are used during meta-clustering.\n"
            f"User labels appear ONLY in F8 (validation).\n\n"
            f"Silhouette scores by K:\n{k_line}\n"
            f"{elbow_note}\n"
            f"Meta-habitats (sorted by prevalence):\n"
            f"{summary_lines}\n"
            f"Figures:\n"
            f"  F0.0 — Cohort tumor profile (raw medians & IQR)\n"
            f"  F1 — Meta-habitat summary table\n"
            f"  F2 — K selection curve\n"
            f"  F3 — PCA scatter (centroid space)\n"
            f"  F4 — Mean centroid heatmap\n"
            f"  F5 — Per-MH MRI bar profiles\n"
            f"  F6 — Radar profiles (per meta-habitat)\n"
            f"  F7 — Radar profiles (all overlaid)\n"
            f"  F8 — User label vs MH cross-tabulation\n"
            + ("  F_cos — Directional similarity (2nd pass)\n" if has_cosine_fig else "")
            + f"  F9 — Segmentation maps\n"
        )
        ax.text(0.08, 0.94, text, transform=ax.transAxes,
                fontsize=10, va="top", fontfamily="monospace", color="#1C1C1E")
        pdf.savefig(fig)
        plt.close(fig)

        # ── One page per standard figure ─────────────────────────────────
        all_pages = [(os.path.join(out_dir, fname), title)
                     for fname, title in _FIG_TITLES]
        # Append per-mouse segmentation maps after the standard figures
        if seg_figs:
            all_pages.extend(seg_figs)

        for fpath, title in all_pages:
            if not os.path.exists(fpath):
                continue
            img  = mpimg.imread(fpath)
            fig2 = plt.figure(figsize=(10, 7.5))
            ax2  = fig2.add_axes([0, 0.04, 1, 0.92])
            ax2.imshow(img)
            ax2.axis("off")
            fig2.text(0.5, 0.01, title, ha="center", fontsize=9, color="#6B7280")
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    return path


# =============================================================================
# 7b. LEVEL-2 SUB-CLUSTERING FIGURES + PDF
# =============================================================================

_SUB_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
               "#00BCD4", "#795548", "#607D8B"]


def fig_sub_profiles(sub_results: dict, out_dir: str) -> list:
    """
    One bar-chart per non-skipped meta-habitat showing sub-habitat centroid
    profiles side by side, with the meta-centroid overlaid as a dashed step.

    Returns list of (abs_path, mh_label, k_sub) tuples.
    """
    paths = []
    for mh_idx, res in sub_results.items():
        if res.get("skipped"):
            continue

        mh_label      = res["meta_stat"]["display_label"]
        sub_stats     = res["sub_stats"]
        mri_params    = res["mri_params"]
        meta_centroid = res["meta_centroid"]
        k_sub         = res["sub_k"]
        n_params      = len(mri_params)

        x     = np.arange(n_params)
        width = 0.72 / k_sub

        fig, ax = plt.subplots(figsize=(max(7, n_params * 1.3), 5))

        for s, ss in enumerate(sub_stats):
            vals   = [ss["centroid"].get(p, 0.0) for p in mri_params]
            color  = _SUB_COLORS[s % len(_SUB_COLORS)]
            offset = (s - k_sub / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.92,
                   label=f"{ss['display_label']}  ({ss['n_voxels']:,} vox)",
                   color=color, alpha=0.82, edgecolor="black", linewidth=0.4)

        # Meta-centroid reference
        mc_x = np.concatenate([[x[0] - 0.45],
                                np.repeat(x[1:], 2) - 0.45,
                                [x[-1] + 0.45]])
        mc_y = np.repeat(meta_centroid, 2)
        ax.plot(mc_x, mc_y, color="black", linewidth=1.4,
                linestyle="--", label="Meta-centroid (MH mean)", zorder=5)

        ax.axhline(0, color="#888", linewidth=0.7, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("DTI-", "") for p in mri_params],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Robust-scaled value (IQR units)", fontsize=9)
        ax.set_title(f"Level 2 — Sub-habitats within  {mh_label}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9, loc="best")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        fname = f"f10_sub_{mh_label.lower()}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append((fpath, mh_label, k_sub))

    return paths


# ── colour palette for sub-habitats (max 8) ──────────────────────────────────
_SUB_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]


_REF_PARAM_PRIORITY = ["MTR", "T2", "T2star", "DTI-FA", "DTI-AD", "DTI-RD", "DTI-MD"]


def _pick_ref_param(df: pd.DataFrame) -> str | None:
    """Return the first priority MRI parameter found in df, or None."""
    for p in _REF_PARAM_PRIORITY:
        if p in df.columns:
            return p
    extras = [c for c in df.columns if c not in _NON_PARAM_COLS
              and c not in {"mouse", "sub_label"}]
    return extras[0] if extras else None


def fig_sub_segmentation(res: dict, out_dir: str,
                          mice_list: list | None = None,
                          ref_param: str | None = None) -> list:
    """
    For each mouse, draw a grid of tumour slices with:
      • Grayscale MRI background  — one of the CSV parameters normalised to [0,1]
        (auto-selected by priority: MTR > T2 > T2star > DTI-FA … or user-supplied)
      • Semi-transparent colour overlay  — one colour per sub-habitat (alpha ≈ 0.55)
      • Sub-habitat border contours  — thin vector LineCollection

    The full tumour ROI is always shown (all habitats in grayscale); only the
    sub-habitats of this meta-habitat receive the colour overlay.

    Returns list of (abs_path, mouse_name) tuples.
    """
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    spatial_df = res.get("spatial_df")
    if spatial_df is None or spatial_df.empty:
        return []

    sub_labels = res["sub_labels"]
    k_sub      = res["sub_k"]
    sub_stats  = res["sub_stats"]
    mh_label   = res["meta_stat"]["display_label"]

    df = spatial_df.copy().reset_index(drop=True)
    df["sub_label"] = sub_labels

    colors     = [_SUB_COLORS[s % len(_SUB_COLORS)] for s in range(k_sub)]
    color_rgba = [mcolors.to_rgba(c) for c in colors]   # alpha set per-pixel below

    # Pre-load full-tumour CSVs (background MRI values + full spatial extent)
    bg_frames: dict[str, pd.DataFrame] = {}
    detected_ref: str | None = None
    if mice_list is not None:
        for mouse in mice_list:
            try:
                bg = pd.read_csv(mouse["csv_path"])
                if "X" in bg.columns and "Y" in bg.columns:
                    bg_frames[mouse["name"]] = bg
                    if detected_ref is None:
                        detected_ref = _pick_ref_param(bg)
            except Exception:
                pass

    # Honour explicit override; fall back to auto-detection
    ref = ref_param if ref_param is not None else detected_ref

    outputs = []
    for mouse_name, df_m in df.groupby("mouse", sort=False):
        slices = sorted(df_m["Slice"].unique())
        n_sl   = len(slices)
        if n_sl == 0:
            continue

        ncols = min(6, n_sl)
        nrows = int(np.ceil(n_sl / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.8 * ncols, 2.8 * nrows + 1.1),
                                 squeeze=False)
        fig.patch.set_facecolor(_THEME["bg"])

        bg_df_mouse = bg_frames.get(mouse_name)
        has_ref = bg_df_mouse is not None and ref is not None and ref in bg_df_mouse.columns

        for si, sl in enumerate(slices):
            ax  = axes[si // ncols][si % ncols]
            dfs = df_m[df_m["Slice"] == sl].copy()

            # ── Spatial extent from full-tumour CSV ───────────────────────
            if bg_df_mouse is not None and "Slice" in bg_df_mouse.columns:
                bg_sl = bg_df_mouse[bg_df_mouse["Slice"] == sl]
            elif bg_df_mouse is not None:
                bg_sl = bg_df_mouse
            else:
                bg_sl = dfs

            all_x = bg_sl["X"].values if len(bg_sl) else dfs["X"].values
            all_y = bg_sl["Y"].values if len(bg_sl) else dfs["Y"].values

            if len(all_x) == 0:
                ax.axis("off")
                continue

            x_min, x_max = int(all_x.min()), int(all_x.max())
            y_min, y_max = int(all_y.min()), int(all_y.max())
            w = x_max - x_min + 1
            h = y_max - y_min + 1

            # ── Layer 1: grayscale MRI background ─────────────────────────
            if has_ref and len(bg_sl):
                ref_grid = np.full((h, w), np.nan, dtype=float)
                bxi = (bg_sl["X"].values.astype(int) - x_min)
                byi = (bg_sl["Y"].values.astype(int) - y_min)
                valid = (bxi >= 0) & (bxi < w) & (byi >= 0) & (byi < h)
                ref_grid[byi[valid], bxi[valid]] = bg_sl[ref].values[valid]

                finite = ref_grid[np.isfinite(ref_grid)]
                if len(finite):
                    v_lo = float(np.nanpercentile(finite, 2))
                    v_hi = float(np.nanpercentile(finite, 98))
                    v_rng = max(v_hi - v_lo, 1e-8)
                    ref_norm = np.where(np.isfinite(ref_grid),
                                        np.clip((ref_grid - v_lo) / v_rng, 0.0, 1.0),
                                        0.0)
                    ax.imshow(ref_norm, cmap="gray", vmin=0, vmax=1,
                              origin="upper", interpolation="nearest", aspect="equal")
                else:
                    has_ref = False   # fallback for this slice

            if not has_ref or not len(bg_sl):
                # Fallback: uniform light-grey canvas
                ax.set_facecolor("#DEDEDE")

            # ── Layer 2: sub-habitat RGBA overlay + label grid ────────────
            rgba_ov  = np.zeros((h, w, 4), dtype=float)
            sub_grid = np.full((h, w), -1, dtype=int)

            if len(dfs):
                xi_arr = (dfs["X"].values.astype(int) - x_min)
                yi_arr = (dfs["Y"].values.astype(int) - y_min)
                sl_arr = dfs["sub_label"].values.astype(int)
                valid  = (xi_arr >= 0) & (xi_arr < w) & (yi_arr >= 0) & (yi_arr < h)
                xi_arr, yi_arr, sl_arr = xi_arr[valid], yi_arr[valid], sl_arr[valid]

                for s in range(k_sub):
                    mask = sl_arr == s
                    r, g, b, _ = color_rgba[s]
                    rgba_ov[yi_arr[mask], xi_arr[mask]] = [r, g, b, 0.55]
                    sub_grid[yi_arr[mask], xi_arr[mask]] = s

            ax.imshow(rgba_ov, origin="upper", interpolation="nearest", aspect="equal")

            # ── Layer 3: sub-habitat border contours ──────────────────────
            u_labs = np.unique(sub_grid[sub_grid >= 0])
            if len(u_labs) > 1:
                segs = []
                rs, cs = np.where(sub_grid[:-1, :] != sub_grid[1:, :])
                if len(rs):
                    segs.append(np.stack([
                        np.column_stack([cs - 0.5, rs + 0.5]),
                        np.column_stack([cs + 0.5, rs + 0.5]),
                    ], axis=1))
                rs, cs = np.where(sub_grid[:, :-1] != sub_grid[:, 1:])
                if len(rs):
                    segs.append(np.stack([
                        np.column_stack([cs + 0.5, rs - 0.5]),
                        np.column_stack([cs + 0.5, rs + 0.5]),
                    ], axis=1))
                if segs:
                    ax.add_collection(LineCollection(
                        np.concatenate(segs, axis=0),
                        colors=[(0, 0, 0, 0.65)], linewidths=0.6,
                        antialiased=False))

            ax.set_title(f"Sl {int(sl)}", fontsize=7, pad=2)
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("#BBBBBB")

        # blank unused axes
        for si in range(n_sl, nrows * ncols):
            axes[si // ncols][si % ncols].axis("off")

        handles = [
            mpatches.Patch(facecolor=colors[s], alpha=0.80,
                           label=f"{sub_stats[s]['display_label']}  "
                                 f"({sub_stats[s]['n_voxels']:,} vox)")
            for s in range(k_sub)
        ]
        ref_lbl = f"Grayscale background: {ref}" if (has_ref and ref) else "Gray background"
        handles.append(mpatches.Patch(facecolor="#888888", label=ref_lbl))
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 4), fontsize=8,
                   bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

        safe_mouse = re.sub(r'[^\w\-]', '_', mouse_name)
        safe_mh    = re.sub(r'[^\w\-]', '_', mh_label)
        fname  = f"f10_seg_{safe_mh}_{safe_mouse}.png"
        fpath  = os.path.join(out_dir, fname)
        ref_title = f"  |  background: {ref}" if (has_ref and ref) else ""
        fig.suptitle(f"{mh_label} — sub-habitats | {mouse_name}{ref_title}",
                     fontsize=10, fontweight="bold", y=1.0)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(fpath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        outputs.append((os.path.abspath(fpath), mouse_name))

    return outputs


def fig_sub_delta_heatmap(sub_results: dict, out_dir: str) -> str | None:
    """
    Heatmap of Δ(sub_centroid − meta_centroid) for every sub-habitat across all
    active meta-habitats.  Rows = sub-habitats (MH1.1, MH1.2, …), cols = params.
    Separator lines mark boundaries between different meta-habitats.
    Returns path to the saved figure, or None when there are no active results.
    """
    rows_data  = []   # list of (display_label, mh_display_label, delta_dict)
    mri_params = None

    for res in sub_results.values():
        if res.get("skipped"):
            continue
        if mri_params is None:
            mri_params = res["mri_params"]
        for ss in res["sub_stats"]:
            rows_data.append((ss["display_label"],
                              res["meta_stat"]["display_label"],
                              ss["delta"]))

    if not rows_data or mri_params is None:
        return None

    n_rows = len(rows_data)
    n_cols = len(mri_params)
    data   = np.array([[r[2].get(p, 0.0) for p in mri_params] for r in rows_data])
    labels = [r[0] for r in rows_data]
    mh_of_row = [r[1] for r in rows_data]
    vmax   = max(float(np.abs(data).max()), 0.3)

    fig_h = max(3.0, n_rows * 0.60 + 1.8)
    fig_w = max(6.0, n_cols * 1.15 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(_THEME["bg"])

    im = ax.imshow(data, cmap=_DIVMAP, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mri_params, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        "Level-2 sub-habitat profiles  (Δ = sub-centroid − meta-centroid)\n"
        "Red = above MH mean  ·  Blue = below MH mean  ·  IQR units",
        fontsize=10, fontweight="bold",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(v) > vmax * 0.55 else "black")

    # Separator lines between meta-habitats
    prev_mh = None
    for i, mh in enumerate(mh_of_row):
        if prev_mh is not None and mh != prev_mh:
            ax.axhline(i - 0.5, color="#444444", linewidth=1.5, linestyle="--")
        prev_mh = mh

    fig.colorbar(im, ax=ax, label="Δ centroid vs meta-centroid (IQR units)",
                 shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, "f11_sub_delta_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_sub_radar(sub_results: dict, out_dir: str) -> str | None:
    """
    Radar charts for sub-habitats, one subplot per active meta-habitat.
    Each sub-habitat is a coloured filled trace; the meta-centroid is a thick
    dashed black reference.  Uses the same _radar_setup/_radar_trace helpers
    as the main-discovery F6/F7 figures.
    Returns path to the saved figure, or None when no active results exist.
    """
    active = {k: v for k, v in sub_results.items() if not v.get("skipped")}
    if not active:
        return None

    n_mh  = len(active)
    ncols = min(3, n_mh)
    nrows = int(np.ceil(n_mh / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.8, nrows * 4.8),
        subplot_kw=dict(polar=True),
        squeeze=False,
    )
    fig.patch.set_facecolor(_THEME["bg"])

    for plot_idx, (_, res) in enumerate(active.items()):
        r, c       = divmod(plot_idx, ncols)
        ax         = axes[r][c]
        mh_label   = res["meta_stat"]["display_label"]
        mri_params = res["mri_params"]
        sub_stats  = res["sub_stats"]
        k_sub      = res["sub_k"]
        meta_c     = res["meta_centroid"]

        angles = _radar_setup(ax, mri_params)

        # Meta-centroid reference — thick dashed black
        meta_disp  = (np.clip(meta_c, _RADAR_VMIN, _RADAR_VMAX) - _RADAR_VMIN).tolist()
        a_closed   = angles + [angles[0]]
        d_closed   = meta_disp + [meta_disp[0]]
        ax.plot(a_closed, d_closed, color="#222222", linewidth=2.8,
                linestyle="--", zorder=6, label="Meta-centroid")

        # One trace per sub-habitat
        for s, ss in enumerate(sub_stats):
            color = _SUB_COLORS[s % len(_SUB_COLORS)]
            vals  = [ss["centroid"].get(p, 0.0) for p in mri_params]
            _radar_trace(ax, angles, vals, color,
                         label=f"{ss['display_label']}  ({ss['n_voxels']:,} vox)",
                         alpha=0.22, lw=2.2)

        ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.18),
                  fontsize=7.5, framealpha=0.9)
        ax.set_title(
            f"{mh_label}  —  K_sub = {k_sub}",
            fontsize=9, fontweight="bold", pad=14,
        )

    for plot_idx in range(n_mh, nrows * ncols):
        r, c = divmod(plot_idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Level-2 radar profiles — sub-habitats within each meta-habitat\n"
        "(IQR units — dashed ring = tumour median  [0]  — dashed black = meta-centroid)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "f12_sub_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_subclust_pdf(sub_results: dict, out_dir: str, n_mice: int,
                           mice_list: list | None = None) -> str | None:
    """
    Generate all Level-2 figures and assemble a standalone PDF.

    Per non-skipped meta-habitat:
      • bar-chart page   — sub-habitat centroid profiles vs. meta-centroid
      • segmentation pages — one page per mouse showing sub-habitat labels
                             on the actual tumour slices

    K=1 always evaluated — skipped MHs are listed on the title page.
    Returns PDF path, or None if every meta-habitat was skipped.
    """
    active = {k: v for k, v in sub_results.items() if not v.get("skipped")}
    if not active:
        return None

    bar_paths = fig_sub_profiles(sub_results, out_dir)
    bar_map   = {mh_label: fpath for fpath, mh_label, _ in bar_paths}

    delta_path = fig_sub_delta_heatmap(sub_results, out_dir)
    radar_path = fig_sub_radar(sub_results, out_dir)

    pdf_path = os.path.join(out_dir, "multi_mouse_discovery_level2_report.pdf")
    folder   = os.path.basename(os.path.abspath(out_dir))
    now      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def _img_page(pdf_obj, fpath, caption):
        if not fpath or not os.path.exists(fpath):
            return
        img  = mpimg.imread(fpath)
        h, w = img.shape[:2]
        fw   = 11.0
        fh   = min(9.0, fw * h / w) + 0.5
        fig  = plt.figure(figsize=(fw, fh))
        tf   = 0.5 / fh
        ax   = fig.add_axes([0, tf, 1, 1 - tf])
        ax.imshow(img); ax.axis("off")
        fig.text(0.5, tf * 0.4, caption,
                 ha="center", va="center", fontsize=9, color="#6B7280")
        pdf_obj.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(pdf_path) as pdf:

        # ── Title page ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.96, folder, transform=ax.transAxes,
                fontsize=16, fontweight="bold", ha="center", va="top",
                color="#1C1C1E")
        ax.text(0.5, 0.90, "Level 2 — Intra-habitat sub-clustering",
                transform=ax.transAxes, fontsize=13, ha="center", color="#2E75B6")
        ax.text(0.5, 0.85,
                f"Generated: {now}    Mice: {n_mice}\n"
                "K=1 always tested — split only when BIC confirms sub-structure.\n"
                "F11 — Δ heatmap (all sub-habitats)  ·  F12 — Radar profiles\n"
                "Per MH: centroid bar chart  +  segmentation maps per mouse.",
                transform=ax.transAxes, fontsize=8.5, ha="center",
                color="#555555", style="italic")

        lines = ""
        for mh_idx, res in sub_results.items():
            ms = res["meta_stat"]
            if res.get("skipped"):
                lines += f"  {ms['display_label']:6s}: SKIP — {res.get('skip_reason', '')}\n"
            else:
                lines += (f"  {ms['display_label']:6s}: K_sub={res['sub_k']}"
                          f"  ({len(res['X']):,} voxels)\n")
                for ss in res["sub_stats"]:
                    delta_str = "  ".join(
                        f"{p.replace('DTI-', '')}: {v:+.2f}"
                        for p, v in ss["delta"].items()
                    )
                    lines += (f"    {ss['display_label']:10s}  "
                              f"{ss['n_voxels']:>6,} vox   Δ {delta_str}\n")

        ax.text(0.06, 0.76, lines, transform=ax.transAxes,
                fontsize=8, va="top", fontfamily="monospace", color="#1C1C1E")
        pdf.savefig(fig); plt.close(fig)

        # ── Overview figures (all MHs together) ─────────────────────
        _img_page(pdf, delta_path,
                  "F11 — Δ heatmap: sub-centroid − meta-centroid  (all sub-habitats)")
        _img_page(pdf, radar_path,
                  "F12 — Radar profiles: sub-habitats within each meta-habitat")

        # ── Per meta-habitat pages ──────────────────────────────────
        for mh_idx, res in active.items():
            mh_label = res["meta_stat"]["display_label"]
            k_sub    = res["sub_k"]

            # Bar chart
            _img_page(pdf, bar_map.get(mh_label),
                      f"{mh_label} — centroid profiles  (K_sub={k_sub})")

            # Segmentation maps — one page per mouse
            seg_paths = fig_sub_segmentation(res, out_dir, mice_list=mice_list)
            for seg_fpath, mouse_name in seg_paths:
                _img_page(pdf, seg_fpath,
                          f"{mh_label} — sub-habitats on slices  |  {mouse_name}")

    return pdf_path


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def run_discovery(csv_paths: list, k_override: int | None,
                   k_range, n_init: int, method: str,
                   output_dir: str, log,
                   criterion: str = 'elbow',
                   merge_mode: str = 'suggest',
                   cosine_threshold: float = 0.92,
                   sub_clust: bool = False,
                   k_sub_max: int = 4,
                   sub_min_voxels: int = 200) -> dict:
    """
    Full pipeline.  log(str) is called for progress messages.
    Returns a result dict or raises ValueError on unrecoverable errors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    log("Loading CSVs…")
    mice_list = []
    for p in csv_paths:
        m = load_mouse_data(p)
        if m is None:
            log(f"  [!] Skipped (no valid clusters): {os.path.basename(p)}")
        else:
            mice_list.append(m)
            log(f"  OK  {m['name']}  ({len(m['clusters'])} clusters)")

    if len(mice_list) < 2:
        raise ValueError(
            "Need at least 2 valid mice.  "
            "Check that the CSVs have a 'Habitat' column and enough voxels."
        )

    # ── Parameters ───────────────────────────────────────────────────────
    common_params = find_common_parameters(mice_list)
    if not common_params:
        raise ValueError("No parameter in common across all selected mice.")
    log(f"Common parameters ({len(common_params)}): {', '.join(common_params)}")

    # ── Centroid matrix ───────────────────────────────────────────────────
    X, row_meta = build_centroid_matrix(mice_list, common_params)
    n_total_clusters = len(X)
    log(f"Centroid pool: {n_total_clusters} clusters from {len(mice_list)} mice")

    if n_total_clusters < 3:
        raise ValueError("Too few clusters to run meta-clustering (need ≥ 3).")

    # ── K selection ──────────────────────────────────────────────────────
    if k_override and k_override >= 2:
        chosen_k  = min(k_override, n_total_clusters - 1)
        meta_lbl  = _fit_predict(X, chosen_k, n_init, RANDOM_SEED, method)
        k_scores  = {chosen_k: _silhouette_safe(X, meta_lbl)}
        log(f"K forced to {chosen_k}  (silhouette={k_scores[chosen_k]:.3f})")
    else:
        safe_kmax = min(max(k_range), n_total_clusters - 1)
        safe_range = range(min(k_range), safe_kmax + 1)
        log(f"Searching optimal K in {list(safe_range)}…")
        chosen_k, k_scores = find_optimal_meta_k(
            X, safe_range, n_init, RANDOM_SEED, method, log,
            criterion=criterion,
        )
        crit_tag = "Elbow" if criterion == 'elbow' else "Max silhouette"
        log(f"→ {crit_tag} K = {chosen_k}  (silhouette={k_scores[chosen_k]:.3f})")

    # ── Meta-clustering ───────────────────────────────────────────────────
    log(f"Running {method.upper()} meta-clustering (K={chosen_k}, n_init={n_init})…")
    meta_labels = run_meta_clustering(X, chosen_k, n_init, RANDOM_SEED, method)

    # ── Statistics ────────────────────────────────────────────────────────
    meta_stats = compute_meta_stats(
        X, meta_labels, row_meta, common_params, chosen_k, len(mice_list)
    )
    log("Meta-habitats discovered:")
    for s in meta_stats:
        log(f"  {s['display_label']:5s}  {s['n_mice']:2d}/{len(mice_list)} mice  "
            f"{s['n_clusters']:2d} clusters  "
            f"presence {s['presence_rate']*100:.0f}%")

    # ── Second pass: directional merge ────────────────────────────────────
    sim_matrix          = None
    merge_groups        = []
    near_zero           = None
    pre_merge_stats     = meta_stats   # used for cosine figure
    original_k          = chosen_k
    merge_group_labels  = []

    if merge_mode in ("suggest", "auto"):
        log(f"\n-- 2nd pass: directional merge  (cosine ≥ {cosine_threshold}) --")
        sim_matrix, merge_groups, near_zero = compute_cosine_similarity(
            meta_stats, common_params, cosine_threshold)

        if merge_groups:
            for g in merge_groups:
                g_labels = [meta_stats[i]["display_label"] for i in g]
                log(f"  → Merge candidate: {' + '.join(g_labels)}")
            merge_group_labels = [
                [meta_stats[i]["display_label"] for i in g]
                for g in merge_groups
            ]
        else:
            log("  → No candidates above threshold.")

        if merge_mode == "auto" and merge_groups:
            pre_merge_stats = list(meta_stats)   # snapshot before merge
            meta_labels, meta_stats = apply_directional_merge(
                merge_groups, meta_stats, meta_labels, X, row_meta,
                common_params, len(mice_list))
            log(f"  → K reduced: {original_k} → {len(meta_stats)}  (auto-merge applied)")
            log("  Merged meta-habitats:")
            for s in meta_stats:
                log(f"  {s['display_label']:5s}  {s['n_mice']:2d}/{len(mice_list)} mice  "
                    f"{s['n_clusters']:2d} clusters")

    # ── Figures ───────────────────────────────────────────────────────────
    log("Generating figures…")
    n_mice = len(mice_list)
    fig_cohort_tumor_profile(mice_list, output_dir);                    log("  F0.0 done (cohort tumor profile)")
    fig_mouse_index(mice_list, output_dir);                              log("  F0 done")
    fig_summary_table(meta_stats, common_params, n_mice, output_dir);   log("  F1 done")
    fig_k_selection(k_scores, chosen_k, output_dir, criterion=criterion); log("  F2 done")
    fig_pca_scatter(X, meta_labels, meta_stats, output_dir);            log("  F3 done")
    fig_profiles_heatmap(meta_stats, common_params, output_dir);        log("  F4 done")
    fig_meta_profiles_bar(meta_stats, common_params, n_mice, output_dir); log("  F5 done")
    fig_radar_per_mh(meta_stats, common_params, n_mice,
                     X, meta_labels, row_meta, output_dir);             log("  F6 done")
    fig_radar_combined(meta_stats, common_params, output_dir);          log("  F7 done")
    _f8_groups = merge_groups if merge_mode == "suggest" else []
    fig_crosstab(row_meta, meta_labels, meta_stats, output_dir,
                 merge_groups=_f8_groups);                              log("  F8 done")

    if merge_mode != "off" and sim_matrix is not None:
        fig_cosine_heatmap(
            sim_matrix, pre_merge_stats, near_zero, merge_groups,
            cosine_threshold, common_params, output_dir, merge_mode)
        log("  F_cos done")

    seg_figs = fig_segmentation_maps(
        mice_list, row_meta, meta_labels, meta_stats, output_dir)
    log(f"  F9 done  ({len(seg_figs)} mouse map(s))")

    # ── Exports ───────────────────────────────────────────────────────────
    csv_out = export_assignment_csv(row_meta, meta_labels, meta_stats, output_dir)
    log(f"Assignment CSV → {csv_out}")

    pdf_out = generate_pdf(
        meta_stats, common_params, len(mice_list),
        chosen_k, method, k_scores, output_dir,
        seg_figs=seg_figs,
        criterion=criterion,
        merge_mode=merge_mode,
        cosine_threshold=cosine_threshold,
        merge_group_labels=merge_group_labels,
        original_k=original_k,
    )
    log(f"PDF report     → {pdf_out}")

    # ── Level-2 sub-clustering (optional) ────────────────────────────────
    sub_results = {}
    if sub_clust:
        log("\n-- Level 2: intra-habitat sub-clustering --")
        sub_results = run_sub_clustering(
            mice_list, row_meta, meta_labels, meta_stats, common_params,
            k_sub_max=k_sub_max, min_voxels=sub_min_voxels, log=log)
        pdf2 = generate_subclust_pdf(sub_results, output_dir, len(mice_list),
                                     mice_list=mice_list)
        if pdf2:
            log(f"Level-2 PDF    → {pdf2}")
        else:
            log("Level 2: no meta-habitats had sufficient voxels for sub-clustering.")

    log("Done.")

    return {
        "meta_stats"       : meta_stats,
        "common_params"    : common_params,
        "chosen_k"         : chosen_k,
        "k_scores"         : k_scores,
        "output_dir"       : output_dir,
        "merge_groups"     : merge_groups,
        "merge_group_labels": merge_group_labels,
        "sub_results"      : sub_results,
    }


# =============================================================================
# 9. TKINTER UI
# =============================================================================

class DiscoveryApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Multi-Mouse Habitat Discovery")
        self.configure(bg=_THEME["bg"])
        self.resizable(True, True)
        self.minsize(780, 580)
        self._queue  = queue.Queue()
        self._thread = None
        self._build_ui()
        self._poll()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Left: file list
        left = tk.Frame(self, bg=_THEME["bg"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=8)

        tk.Label(left, text="Selected habitats_result.csv files",
                 font=("Helvetica", 10, "bold"),
                 bg=_THEME["bg"], fg=_THEME["text"]).pack(anchor="w")

        list_frame = tk.Frame(left, bg=_THEME["separator"], bd=1, relief=tk.FLAT)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._lb = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED,
            bg=_THEME["bg_card"], fg=_THEME["text"],
            font=("Helvetica", 8), relief=tk.FLAT, bd=0,
            highlightthickness=0, yscrollcommand=sb.set,
        )
        sb.configure(command=self._lb.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(left, bg=_THEME["bg"])
        btn_row.pack(fill=tk.X)
        for label, cmd in [
            ("Add CSV(s)",   self._add_files),
            ("Add folder",   self._add_folder),
            ("Remove sel.",  self._remove_sel),
            ("Clear all",    self._clear),
        ]:
            self._btn(btn_row, label, cmd).pack(side=tk.LEFT, padx=2, pady=2)

        self._count_lbl = tk.Label(
            left, text="0 file(s) selected",
            bg=_THEME["bg"], fg=_THEME["text_sub"], font=("Helvetica", 8),
        )
        self._count_lbl.pack(anchor="w", pady=(2, 0))

        # Right: settings + log
        right = tk.Frame(self, bg=_THEME["bg"], width=290)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 12), pady=8)
        right.pack_propagate(False)

        # ── Settings ─────────────────────────────────────────────────────
        self._section(right, "Settings")

        self._label(right, "Method")
        self._method = tk.StringVar(value="gmm")
        mf = tk.Frame(right, bg=_THEME["bg"])
        mf.pack(anchor="w", padx=6)
        for m in ("gmm", "kmeans"):
            tk.Radiobutton(
                mf, text=m.upper(), variable=self._method, value=m,
                bg=_THEME["bg"], fg=_THEME["text"],
                selectcolor=_THEME["bg_card"], font=("Helvetica", 9),
            ).pack(side=tk.LEFT, padx=4)

        self._label(right, "K selection criterion")
        self._k_criterion = tk.StringVar(value='elbow')
        kc_frame = tk.Frame(right, bg=_THEME["bg"])
        kc_frame.pack(anchor="w", padx=6)
        tk.Radiobutton(
            kc_frame, text="Elbow  (recommended — avoids K=2 outlier trap)",
            variable=self._k_criterion, value='elbow',
            bg=_THEME["bg"], fg=_THEME["text"],
            selectcolor=_THEME["bg_card"], font=("Helvetica", 9),
        ).pack(anchor="w")
        tk.Radiobutton(
            kc_frame, text="Max silhouette  (highest separation score)",
            variable=self._k_criterion, value='max_silhouette',
            bg=_THEME["bg"], fg=_THEME["text"],
            selectcolor=_THEME["bg_card"], font=("Helvetica", 9),
        ).pack(anchor="w")

        self._label(right, "2nd pass — directional merge")
        self._merge_mode = tk.StringVar(value="suggest")
        mf2 = tk.Frame(right, bg=_THEME["bg"])
        mf2.pack(anchor="w", padx=6)
        for val, txt in [
            ("off",     "Off"),
            ("suggest", "Suggest  (default — report only)"),
            ("auto",    "Auto-merge"),
        ]:
            tk.Radiobutton(
                mf2, text=txt, variable=self._merge_mode, value=val,
                bg=_THEME["bg"], fg=_THEME["text"],
                selectcolor=_THEME["bg_card"], font=("Helvetica", 9),
            ).pack(anchor="w")

        thr_row = tk.Frame(right, bg=_THEME["bg"])
        thr_row.pack(anchor="w", padx=6, pady=(2, 0))
        tk.Label(thr_row, text="Cosine threshold", font=("Helvetica", 9),
                 bg=_THEME["bg"], fg=_THEME["text"]).pack(side=tk.LEFT)
        self._cosine_thr = tk.DoubleVar(value=0.92)
        tk.Spinbox(thr_row, from_=0.70, to=0.99, increment=0.01,
                   textvariable=self._cosine_thr, format="%.2f",
                   width=6, font=("Helvetica", 9),
                   bg=_THEME["bg_input"], relief=tk.FLAT, bd=1,
                   ).pack(side=tk.LEFT, padx=6)

        # ── Level 2 — sub-clustering ──────────────────────────────────────
        ttk.Separator(right).pack(fill="x", padx=4, pady=6)
        self._label(right, "Level 2 — intra-habitat sub-clustering")
        self._sub_clust = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Enable sub-clustering  (separate PDF)",
                       variable=self._sub_clust,
                       bg=_THEME["bg"], fg=_THEME["text"],
                       activebackground=_THEME["bg"],
                       font=("Helvetica", 9),
                       selectcolor=_THEME["bg_input"]).pack(anchor="w", padx=6)
        sub_k_row = tk.Frame(right, bg=_THEME["bg"])
        sub_k_row.pack(anchor="w", padx=6, pady=(2, 0))
        tk.Label(sub_k_row, text="Max K_sub  (K=1 = no split, always tested)",
                 font=("Helvetica", 9),
                 bg=_THEME["bg"], fg=_THEME["text"]).pack(side=tk.LEFT)
        self._sub_kmax = tk.IntVar(value=4)
        tk.Spinbox(sub_k_row, from_=2, to=8, textvariable=self._sub_kmax,
                   width=3, font=("Helvetica", 9),
                   bg=_THEME["bg_input"], relief=tk.FLAT, bd=1,
                   ).pack(side=tk.LEFT, padx=6)
        sub_v_row = tk.Frame(right, bg=_THEME["bg"])
        sub_v_row.pack(anchor="w", padx=6, pady=(2, 0))
        tk.Label(sub_v_row, text="Min voxels per MH", font=("Helvetica", 9),
                 bg=_THEME["bg"], fg=_THEME["text"]).pack(side=tk.LEFT)
        self._sub_min_vox = tk.IntVar(value=200)
        tk.Spinbox(sub_v_row, from_=50, to=2000, increment=50,
                   textvariable=self._sub_min_vox, width=6,
                   font=("Helvetica", 9),
                   bg=_THEME["bg_input"], relief=tk.FLAT, bd=1,
                   ).pack(side=tk.LEFT, padx=6)

        self._label(right, "Force K  (0 = automatic selection)")
        self._k_force = tk.IntVar(value=0)
        tk.Spinbox(right, from_=0, to=20, textvariable=self._k_force,
                   width=5, font=("Helvetica", 9),
                   bg=_THEME["bg_input"], relief=tk.FLAT, bd=1,
                   ).pack(anchor="w", padx=6, pady=2)

        self._label(right, "K search range  (min – max)")
        krf = tk.Frame(right, bg=_THEME["bg"])
        krf.pack(anchor="w", padx=6)
        self._kmin = tk.IntVar(value=2)
        self._kmax = tk.IntVar(value=10)
        for var, txt in ((self._kmin, "min"), (self._kmax, "max")):
            tk.Label(krf, text=txt, bg=_THEME["bg"], fg=_THEME["text_sub"],
                     font=("Helvetica", 8)).pack(side=tk.LEFT)
            tk.Spinbox(krf, from_=2, to=25, textvariable=var,
                       width=4, font=("Helvetica", 9),
                       bg=_THEME["bg_input"], relief=tk.FLAT, bd=1,
                       ).pack(side=tk.LEFT, padx=(2, 8))

        self._label(right, "N restarts  (n_init)")
        self._ninit = tk.IntVar(value=30)
        tk.Spinbox(right, from_=1, to=100, textvariable=self._ninit,
                   width=5, font=("Helvetica", 9),
                   bg=_THEME["bg_input"], relief=tk.FLAT, bd=1,
                   ).pack(anchor="w", padx=6, pady=2)

        # ── Output ───────────────────────────────────────────────────────
        self._section(right, "Output directory")
        out_row = tk.Frame(right, bg=_THEME["bg"])
        out_row.pack(fill=tk.X, padx=6, pady=2)
        self._out_dir = tk.StringVar(value=os.path.expanduser("~"))
        tk.Entry(out_row, textvariable=self._out_dir,
                 bg=_THEME["bg_input"], fg=_THEME["text"],
                 font=("Helvetica", 8), relief=tk.FLAT, bd=1,
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._btn(out_row, "…", self._pick_outdir).pack(side=tk.LEFT, padx=2)

        # ── Run ──────────────────────────────────────────────────────────
        self._run_btn = tk.Button(
            right, text="▶  Run Discovery",
            command=self._run,
            bg=_THEME["accent"], fg="white",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, pady=9, cursor="hand2",
            activebackground="#2B5180", activeforeground="white",
        )
        self._run_btn.pack(fill=tk.X, padx=6, pady=10)

        # ── Log ──────────────────────────────────────────────────────────
        self._section(right, "Log")
        self._log_box = scrolledtext.ScrolledText(
            right, height=12,
            bg="#F0F2F5", fg=_THEME["text"],
            font=("Courier", 7), relief=tk.FLAT, bd=1,
            state=tk.DISABLED,
        )
        self._log_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=_THEME["bg_card"], fg=_THEME["text"],
            font=("Helvetica", 8), relief=tk.FLAT, bd=1,
            padx=7, pady=3, cursor="hand2",
            activebackground=_THEME["accent"], activeforeground="white",
        )

    def _section(self, parent, text):
        tk.Frame(parent, bg=_THEME["separator"], height=1).pack(
            fill=tk.X, padx=6, pady=(8, 2))
        tk.Label(parent, text=text.upper(),
                 font=("Helvetica", 7, "bold"),
                 bg=_THEME["bg"], fg=_THEME["text_sub"],
                 ).pack(anchor="w", padx=6)

    def _label(self, parent, text):
        tk.Label(parent, text=text, font=("Helvetica", 9),
                 bg=_THEME["bg"], fg=_THEME["text"],
                 ).pack(anchor="w", padx=6, pady=(4, 0))

    # ── File management ───────────────────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select habitats_result.csv files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        self._insert(paths)

    def _add_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder to scan (recursive) for habitats_result.csv")
        if not folder:
            return
        found = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f == "habitats_result.csv":
                    found.append(os.path.join(root, f))
        self._insert(sorted(found))
        self._log(f"Scanned folder → {len(found)} CSV(s) found")

    def _insert(self, paths):
        existing = set(self._lb.get(0, tk.END))
        for p in paths:
            if p not in existing:
                self._lb.insert(tk.END, p)
        self._update_count()

    def _remove_sel(self):
        for i in reversed(self._lb.curselection()):
            self._lb.delete(i)
        self._update_count()

    def _clear(self):
        self._lb.delete(0, tk.END)
        self._update_count()

    def _pick_outdir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self._out_dir.set(d)

    def _update_count(self):
        n = self._lb.size()
        self._count_lbl.configure(text=f"{n} file(s) selected")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        csv_paths = list(self._lb.get(0, tk.END))
        if len(csv_paths) < 2:
            messagebox.showwarning(
                "Not enough files",
                "Please add at least 2 habitats_result.csv files.",
            )
            return

        k_force      = int(self._k_force.get()) or None
        k_range      = range(int(self._kmin.get()), int(self._kmax.get()) + 1)
        n_init       = int(self._ninit.get())
        method       = self._method.get()
        criterion    = self._k_criterion.get()
        merge_mode   = self._merge_mode.get()
        cosine_thr   = float(self._cosine_thr.get())
        sub_clust    = bool(self._sub_clust.get())
        sub_k_max    = int(self._sub_kmax.get())
        sub_min_vox  = int(self._sub_min_vox.get())
        out_dir      = self._out_dir.get().strip()

        if not out_dir:
            messagebox.showwarning("No output directory", "Please select an output directory.")
            return

        self._run_btn.configure(state=tk.DISABLED, text="Running…")
        self._log("=" * 44)
        self._log(f"Starting  {len(csv_paths)} mice  method={method.upper()}  "
                  f"criterion={criterion}  merge={merge_mode}")

        def _worker():
            try:
                run_discovery(csv_paths, k_force, k_range, n_init,
                              method, out_dir, self._q_log,
                              criterion=criterion,
                              merge_mode=merge_mode,
                              cosine_threshold=cosine_thr,
                              sub_clust=sub_clust,
                              k_sub_max=sub_k_max,
                              sub_min_voxels=sub_min_vox)
            except Exception as exc:
                self._queue.put(("error", str(exc)))
            finally:
                self._queue.put(("done", None))

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def _q_log(self, msg: str):
        self._queue.put(("log", msg))

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "error":
                    self._log(f"[ERROR] {payload}")
                    messagebox.showerror("Error", payload)
                    self._run_btn.configure(state=tk.NORMAL, text="▶  Run Discovery")
                elif kind == "done":
                    self._run_btn.configure(state=tk.NORMAL, text="▶  Run Discovery")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _log(self, msg: str):
        self._log_box.configure(state=tk.NORMAL)
        self._log_box.insert(tk.END, msg + "\n")
        self._log_box.see(tk.END)
        self._log_box.configure(state=tk.DISABLED)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = DiscoveryApp()
    app.mainloop()
