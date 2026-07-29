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
                  out_dir: str) -> str:
    """
    Counts how many clusters of each user label end up in each meta-habitat.
    This is purely for post-hoc validation; labels are NOT used in discovery.
    """
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

    # Normalise row-wise so we read "given a user label, where does it land?"
    row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    matrix_norm = matrix / row_sums

    cell_h  = max(0.35, min(0.7, 10.0 / max(n_ul, 1)))
    fig_h   = max(3.5, n_ul * cell_h + 1.8)
    fig, ax = plt.subplots(figsize=(max(5, n_mh * 1.0 + 1.5), fig_h))

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

    for r in range(n_ul):
        for c in range(n_mh):
            v = matrix_norm[r, c]
            raw = matrix[r, c]
            ax.text(c, r, f"{raw}\n({v*100:.0f}%)",
                    ha="center", va="center", fontsize=6,
                    color="white" if v > 0.55 else "#333")

    fig.colorbar(im, ax=ax, label="Fraction of clusters from that user label",
                 shrink=0.8)
    fig.tight_layout()
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
# 7. PDF REPORT
# =============================================================================

def generate_pdf(meta_stats: list, common_params: list,
                  n_mice: int, chosen_k: int, method: str,
                  k_scores: dict, out_dir: str,
                  seg_figs: list | None = None,
                  criterion: str = 'elbow') -> str:
    """
    Build the discovery PDF.
    seg_figs : optional list of (abs_path, caption) tuples — per-mouse
               segmentation maps appended after the standard figures.
    """
    _FIG_TITLES = [
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
        text = (
            f"Multi-Mouse Habitat Discovery\n"
            f"{'=' * 45}\n\n"
            f"Date            : {now}\n"
            f"Mice analysed   : {n_mice}\n"
            f"Method          : {method.upper()}\n"
            f"K selection     : {criterion_lbl}\n"
            f"Chosen K        : {chosen_k}\n"
            f"Parameters      : {', '.join(common_params)}\n\n"
            f"Discovery is purely data-driven — no biological\n"
            f"labels are used during meta-clustering.\n"
            f"User labels appear ONLY in F8 (validation).\n\n"
            f"Silhouette scores by K:\n{k_line}\n"
            f"{elbow_note}\n"
            f"Meta-habitats (sorted by prevalence):\n"
            f"{summary_lines}\n"
            f"Figures:\n"
            f"  F1 — Meta-habitat summary table\n"
            f"  F2 — K selection curve\n"
            f"  F3 — PCA scatter (centroid space)\n"
            f"  F4 — Mean centroid heatmap\n"
            f"  F5 — Per-MH MRI bar profiles\n"
            f"  F6 — Radar profiles (per meta-habitat)\n"
            f"  F7 — Radar profiles (all overlaid)\n"
            f"  F8 — User label vs MH cross-tabulation\n"
            f"  F9 — Segmentation maps\n"
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
# 8. MAIN PIPELINE
# =============================================================================

def run_discovery(csv_paths: list, k_override: int | None,
                   k_range, n_init: int, method: str,
                   output_dir: str, log,
                   criterion: str = 'elbow') -> dict:
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

    # ── Figures ───────────────────────────────────────────────────────────
    log("Generating figures…")
    n_mice = len(mice_list)
    fig_mouse_index(mice_list, output_dir);                            log("  F0 done")
    fig_summary_table(meta_stats, common_params, n_mice, output_dir);  log("  F1 done")
    fig_k_selection(k_scores, chosen_k, output_dir, criterion=criterion); log("  F2 done")
    fig_pca_scatter(X, meta_labels, meta_stats, output_dir);           log("  F3 done")
    fig_profiles_heatmap(meta_stats, common_params, output_dir);       log("  F4 done")
    fig_meta_profiles_bar(meta_stats, common_params, n_mice, output_dir); log("  F5 done")
    fig_radar_per_mh(meta_stats, common_params, n_mice,
                     X, meta_labels, row_meta, output_dir);            log("  F6 done")
    fig_radar_combined(meta_stats, common_params, output_dir);         log("  F7 done")
    fig_crosstab(row_meta, meta_labels, meta_stats, output_dir);       log("  F8 done")
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
    )
    log(f"PDF report     → {pdf_out}")
    log("Done.")

    return {
        "meta_stats"   : meta_stats,
        "common_params": common_params,
        "chosen_k"     : chosen_k,
        "k_scores"     : k_scores,
        "output_dir"   : output_dir,
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

        k_force   = int(self._k_force.get()) or None
        k_range   = range(int(self._kmin.get()), int(self._kmax.get()) + 1)
        n_init    = int(self._ninit.get())
        method    = self._method.get()
        criterion = self._k_criterion.get()
        out_dir   = self._out_dir.get().strip()

        if not out_dir:
            messagebox.showwarning("No output directory", "Please select an output directory.")
            return

        self._run_btn.configure(state=tk.DISABLED, text="Running…")
        self._log("=" * 44)
        self._log(f"Starting  {len(csv_paths)} mice  method={method.upper()}  "
                  f"criterion={criterion}")

        def _worker():
            try:
                run_discovery(csv_paths, k_force, k_range, n_init,
                              method, out_dir, self._q_log,
                              criterion=criterion)
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
