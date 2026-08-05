"""
comparative_pipeline.py
Runs Classic + Discovery + Joint on the same cohort and produces a
multi-page PDF comparison report.  No Tkinter dependency.
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

from pipeline.loading    import load_data, robust_scale, resolve_dti_parameters
from pipeline.spatial    import prepare_features, inject_dtopo_centroids
from pipeline.clustering import run_clustering, run_clustering_with_size_guard
from pipeline.selection  import select_optimal_k
from pipeline.joint_clustering import run_joint_pipeline
from multi_mouse_discovery import (
    find_common_parameters,
    build_centroid_matrix, run_meta_clustering, find_optimal_meta_k,
    SIZE_FLOOR as _DISC_SIZE_FLOOR,
)
from config import (
    HABITAT_COLORS,
    GMM_SILHOUETTE_GUARD,
    DTOPO_MIN_NECROSIS_FRACTION,
    N_JOBS_GAP, RANDOM_SEED, SPATIAL_WEIGHT,
)

# ── colour helpers ──────────────────────────────────────────────────────────

_METHOD_COLORS = {
    "classic"  : "#4A90D9",
    "discovery": "#E67E22",
    "joint"    : "#27AE60",
}

def _hcol(hab_id, alpha=1.0):
    r, g, b, _ = HABITAT_COLORS[int(hab_id) % len(HABITAT_COLORS)]
    return (r / 255, g / 255, b / 255, alpha)


# ── Semantic habitat colours (Compare PDF only) ──────────────────────────────

_SIG_BULK = (0.13, 0.59, 0.95, 1.0)   # blue  — bulk (most voxels)
_SIG_NECT = (0.90, 0.23, 0.23, 1.0)   # red   — necrotic (highest MD/AD/RD)
_SIG_GRAY = (0.55, 0.55, 0.55, 1.0)   # fallback

# Must not contain blue or red
_DEFAULT_POOL = [
    (1.00, 0.60, 0.00),  # orange
    (0.30, 0.69, 0.31),  # green
    (0.00, 0.74, 0.83),  # teal
    (1.00, 0.76, 0.03),  # amber
    (0.55, 0.27, 0.07),  # brown
    (0.96, 0.50, 0.19),  # deep orange
    (0.61, 0.80, 0.40),  # lime
    (0.62, 0.27, 0.62),  # violet (reserved for future astrocyte type)
]


def _hcol_mapped(hab_id, color_map, alpha=1.0):
    """Use color_map if available, else fall back to HABITAT_COLORS."""
    if color_map:
        c = color_map.get(int(hab_id))
        if c is not None:
            return (c[0], c[1], c[2], alpha)
    return _hcol(int(hab_id), alpha)


def _diff_indices(params):
    """Indices of diffusion parameters (MD / AD / RD, DTI-prefixed or bare)."""
    out = []
    for i, p in enumerate(params):
        pu = p.upper().replace("_", "-")
        if (pu.startswith("DTI-") and pu.endswith(("MD", "AD", "RD"))) \
                or pu in ("MD", "ADC"):
            out.append(i)
    return out


def _signature_assignment(centroids, labels, params):
    """
    Blue  → habitat with most voxels (bulk tumour).
    Red   → habitat with highest diffusion norm (necrosis), if DTI features present.
    Returns (color_map, assigned_set).
    """
    k      = centroids.shape[0]
    counts = np.array([np.sum(labels == i) for i in range(k)])
    cmap   = {}
    done   = set()

    bulk = int(np.argmax(counts))
    cmap[bulk] = _SIG_BULK
    done.add(bulk)

    if k > 1:
        diff_idx = _diff_indices(params)
        if diff_idx:
            norms = np.linalg.norm(centroids[:, diff_idx], axis=1).copy()
            for a in done:
                norms[a] = -np.inf
            nec = int(np.argmax(norms))
            if nec not in done:
                cmap[nec] = _SIG_NECT
                done.add(nec)

    return cmap, done


def _build_cross_color_maps(disc_cents, disc_labels, disc_params,
                             jt_cents,   jt_labels,   jt_params,
                             threshold=0.65):
    """
    Consistent color maps for Discovery and Joint.
    Signature colours (bulk=blue, necrotic=red) are assigned independently.
    Non-signature habitats are matched across methods by cosine similarity.
    Returns (disc_color_map, jt_color_map).
    """
    k_d = len(disc_cents) if disc_cents is not None else 0
    k_j = len(jt_cents)   if jt_cents  is not None else 0

    pool_iter = iter(_DEFAULT_POOL)

    d_map, d_sig, free_d = {}, set(), []
    if k_d > 0:
        d_map, d_sig = _signature_assignment(disc_cents, disc_labels, disc_params)
        free_d = [i for i in range(k_d) if i not in d_sig]

    j_map, j_sig, free_j = {}, set(), []
    if k_j > 0:
        j_map, j_sig = _signature_assignment(jt_cents, jt_labels, jt_params)
        free_j = [i for i in range(k_j) if i not in j_sig]

    # Cosine matching of non-signature habitats
    matches = {}
    if free_d and free_j and disc_cents is not None and jt_cents is not None:
        common = [p for p in disc_params if p in jt_params]
        if common:
            di = [disc_params.index(p) for p in common]
            ji = [jt_params.index(p)   for p in common]
            CD   = disc_cents[np.array(free_d)][:, di]
            CJ   = jt_cents  [np.array(free_j)][:, ji]
            CD_n = CD / (np.linalg.norm(CD, axis=1, keepdims=True) + 1e-10)
            CJ_n = CJ / (np.linalg.norm(CJ, axis=1, keepdims=True) + 1e-10)
            sim  = CD_n @ CJ_n.T
            used_j = set()
            for ai, d_h in enumerate(free_d):
                best_score, best_bi = threshold, None
                for bi, j_h in enumerate(free_j):
                    if bi not in used_j and sim[ai, bi] > best_score:
                        best_score, best_bi = sim[ai, bi], bi
                if best_bi is not None:
                    matches[d_h] = free_j[best_bi]
                    used_j.add(best_bi)

    # Discovery free habitats → sequential default colours
    for d_h in free_d:
        col = next(pool_iter, _SIG_GRAY[:3])
        d_map[d_h] = (*col, 1.0)

    # Matched Joint habitats inherit Discovery colour
    matched_j = set(matches.values())
    for d_h, j_h in matches.items():
        j_map[j_h] = d_map[d_h]

    # Unmatched Joint habitats continue from the same pool
    for j_h in free_j:
        if j_h not in matched_j:
            col = next(pool_iter, _SIG_GRAY[:3])
            j_map[j_h] = (*col, 1.0)

    return d_map, j_map


def _single_color_map(centroids, labels, params):
    """Signature + default palette for one method (no cross-method matching)."""
    k      = centroids.shape[0]
    c_map, done = _signature_assignment(centroids, labels, params)
    pool_iter   = iter(_DEFAULT_POOL)
    for i in range(k):
        if i not in c_map:
            col = next(pool_iter, _SIG_GRAY[:3])
            c_map[i] = (*col, 1.0)
    return c_map


# ── Classic per-mouse ───────────────────────────────────────────────────────

def run_classic_for_mouse(csv_files, method, k_range, k_override,
                           n_init, n_refs, out_dir, log):
    """
    Lightweight single-mouse classic clustering (no hierarchical / MRF).
    Returns dict with keys: labels, features, parameters, df, centroids, k
    or None on failure.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        parameters, df_all  = load_data(csv_files)
        parameters, _       = resolve_dti_parameters(parameters)
        df_scaled, feat_sc, _ = robust_scale(df_all, parameters)

        features_clust, dtopo_meta = prepare_features(
            feat_sc, df_all, parameters, n_init_gmm=n_init)

        k_ov = int(k_override) if str(k_override).strip().isdigit() else None

        if k_ov is not None and k_ov >= 2:
            chosen_k  = k_ov
            gmm_cache = {}
            log(f"  Classic K forced = {chosen_k}")
        else:
            chosen_k, gmm_cache = select_optimal_k(
                features_clust,
                method               = method,
                df_all               = df_scaled,
                k_range              = k_range,
                n_refs               = n_refs,
                n_jobs               = N_JOBS_GAP,
                n_init_gmm           = n_init,
                gmm_silhouette_guard = GMM_SILHOUETTE_GUARD if method == "gmm" else None,
                kmeans_gap_guard     = (method == "kmeans"),
            )

        labels, _, best_k = run_clustering_with_size_guard(
            features_clust, feat_sc, df_scaled, chosen_k,
            method        = method,
            spatial_weight= SPATIAL_WEIGHT,
            n_init_gmm    = n_init,
            parameters    = parameters,
            gmm_cache     = gmm_cache,
        )

        df_scaled["Habitat"] = labels

        # Save only resolved parameters — df_scaled is a copy of df_all which may
        # still contain DTI-MD even when AD+RD are present (resolve_dti_parameters
        # removes it from `parameters` but not from the DataFrame).  Saving the
        # full DataFrame would cause load_mouse_data to pick up DTI-MD and produce
        # a 7-feature centroid matrix while cr["features"] only has 6, breaking
        # the PCA transform in _draw_pca.
        csv_path  = os.path.join(out_dir, "habitats_result.csv")
        save_cols = ['X', 'Y', 'Slice'] + [p for p in parameters if p in df_scaled.columns] + ['Habitat']
        df_scaled[save_cols].to_csv(csv_path, index=False)

        feat_cols  = parameters
        centroids  = df_scaled.groupby("Habitat")[feat_cols].mean().values

        # Build enriched centroid dicts for discovery (mirrors standalone app):
        # inject d_topo_norm so meta-clustering uses the same 7D space as the
        # standalone Discovery tab instead of the 6D CSV-derived space.
        centroids_dict = {
            int(c): {param: float(feat_sc[labels == c, pi].mean())
                     for pi, param in enumerate(parameters)}
            for c in range(int(best_k))
        }
        inject_dtopo_centroids(centroids_dict, labels, dtopo_meta)
        total_vox = max(len(labels), 1)
        clusters_disc = []
        for c in range(int(best_k)):
            mask  = labels == c
            n_vox = int(mask.sum())
            if n_vox < _DISC_SIZE_FLOOR:
                continue
            clusters_disc.append({
                "cluster_id": c,
                "user_label": "Unknown",
                "n_voxels"  : n_vox,
                "proportion": n_vox / total_vox,
                "centroid"  : centroids_dict[c],
            })

        log(f"  Classic k={best_k}")
        return {
            "labels"      : labels,
            "features"    : feat_sc,
            "parameters"  : feat_cols,
            "df"          : df_scaled,
            "centroids"   : centroids,
            "k"           : int(best_k),
            "csv_path"    : csv_path,
            "clusters_disc": clusters_disc,
        }
    except Exception as e:
        import traceback
        log(f"  [!] Classic failed: {e}")
        log(f"  [!] {traceback.format_exc()}")
        return None


# ── Discovery meta-clustering ───────────────────────────────────────────────

def run_discovery_meta(classic_results, method, k_range, k_override, n_init, log):
    """
    Meta-cluster per-mouse classic centroids and map labels back to voxel level.

    Uses the pre-computed clusters_disc from each classic result, which mirrors
    the standalone Discovery tab: centroids include d_topo_norm (via
    inject_dtopo_centroids) and small clusters (<SIZE_FLOOR voxels) are dropped.
    This aligns the comparative pipeline's discovery with the standalone app.

    Returns {mouse_name: {labels, features, parameters, centroids, common_params, k}}
    or None on failure.
    """
    try:
        mice_list = []
        for name, cr in classic_results.items():
            clusters_disc = cr.get("clusters_disc")
            if not clusters_disc:
                log(f"  [Discovery] {name}: no valid clusters — skipped")
                continue
            mice_list.append({
                "name"    : name,
                "csv_path": cr.get("csv_path", ""),
                "clusters": clusters_disc,
            })

        if len(mice_list) < 2:
            log("  [Discovery] Need ≥ 2 valid mice — skipped")
            return None

        common_params = find_common_parameters(mice_list)
        if not common_params:
            log("  [Discovery] No common parameters — skipped")
            return None

        log(f"  [Discovery] {len(mice_list)} mice, {len(common_params)} common params")
        X, row_meta = build_centroid_matrix(mice_list, common_params)
        log(f"  [Discovery] Centroid pool: {len(X)} rows")

        k_ov_str = str(k_override).strip()
        if k_ov_str.isdigit() and int(k_ov_str) >= 2:
            chosen_k = min(int(k_ov_str), len(X) - 1)
            log(f"  [Discovery] K forced = {chosen_k}")
        else:
            # Always search 2–10 for meta-K (mirrors standalone Discovery tab),
            # regardless of the Classic k_range set by the user.
            safe_kmax  = min(10, len(X) - 1)
            if safe_kmax < 2:
                log(f"  [Discovery] Not enough clusters for K search "
                    f"(pool={len(X)}) — forcing K=2")
                chosen_k = min(2, len(X) - 1)
            else:
                chosen_k, _ = find_optimal_meta_k(
                    X, range(2, safe_kmax + 1), n_init, RANDOM_SEED, method, log)

        chosen_k = max(2, min(chosen_k, len(X) - 1))
        log(f"  [Discovery] Meta-clustering K={chosen_k}")
        meta_labels = run_meta_clustering(X, chosen_k, n_init, RANDOM_SEED, method)

        meta_centroids = np.array([
            X[meta_labels == i].mean(axis=0) if (meta_labels == i).any()
            else np.zeros(X.shape[1])
            for i in range(chosen_k)
        ])

        # Map (mouse_name, classic_cluster_id) → meta habitat id
        cluster_to_meta = {
            (rm["mouse"], rm["cluster_id"]): int(meta_labels[i])
            for i, rm in enumerate(row_meta)
        }

        # cluster_to_meta keys use the name from classic_results (same key used
        # when building mice_list above), so look up directly by name.
        out = {}
        for name, cr in classic_results.items():
            if not cr.get("clusters_disc"):
                continue
            hab_col   = cr["df"]["Habitat"].values
            disc_lbls = np.array([
                cluster_to_meta.get((name, int(h)), 0) for h in hab_col
            ])

            mouse_params = list(cr["parameters"])

            # Per-mouse centroids: mean of this mouse's own voxels per meta-habitat.
            # Used for the per-page heatmap so each page reflects the actual feature
            # profile of that specific mouse rather than the global meta-centroids.
            feats_m = cr["features"]
            cents_pm = np.array([
                feats_m[disc_lbls == c].mean(axis=0)
                if (disc_lbls == c).any()
                else np.zeros(len(mouse_params))
                for c in range(chosen_k)
            ])

            out[name] = {
                "labels"       : disc_lbls,
                "features"     : feats_m,
                "parameters"   : mouse_params,
                "centroids"    : cents_pm,
                "common_params": common_params,
                "k"            : chosen_k,
            }
        # Expose global meta-centroids so run_comparative_pdf can build cross-
        # method colour maps without iterating per-mouse results.
        out["_meta"] = {
            "centroids": meta_centroids,
            "params"   : common_params,
        }
        log(f"  [Discovery] Done — {len(out)} mice mapped", "ok")
        return out

    except Exception as e:
        import traceback
        log(f"  [Discovery] Error: {e}")
        log(f"  [Discovery] {traceback.format_exc()}")
        return None


# ── Per-axis drawing helpers ────────────────────────────────────────────────

def _draw_pca(ax, features, labels, centroids, title, header_color,
              color_map=None):
    if features is None or len(features) == 0:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888888")
        ax.set_title(title, fontsize=9, color="#888888", fontweight="bold")
        return

    if features.shape[1] > 2:
        pca       = PCA(n_components=2, random_state=RANDOM_SEED)
        pts       = pca.fit_transform(features)
        cents_2d  = pca.transform(centroids) if centroids is not None else None
        var1, var2 = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({var1:.0f}%)", fontsize=7, color="#cccccc")
        ax.set_ylabel(f"PC2 ({var2:.0f}%)", fontsize=7, color="#cccccc")
    else:
        pts, cents_2d = features, centroids

    for hid in sorted(np.unique(labels)):
        mask = labels == hid
        ax.scatter(pts[mask, 0], pts[mask, 1],
                   s=4, alpha=0.4, linewidths=0,
                   color=_hcol_mapped(hid, color_map, 0.55), label=f"H{hid}")

    if cents_2d is not None:
        present = set(np.unique(labels).tolist())
        for hid, c in enumerate(cents_2d):
            if hid not in present:
                continue
            ax.scatter(c[0], c[1], marker="*", s=220,
                       color=_hcol_mapped(hid, color_map), edgecolors="white",
                       linewidths=0.7, zorder=5)

    ax.set_title(title, fontsize=9, fontweight="bold", color=header_color)
    ax.tick_params(labelsize=6, colors="#aaaaaa")
    ax.legend(fontsize=6, markerscale=2, framealpha=0.5,
              loc="upper right", handlelength=1,
              labelcolor="#cccccc", facecolor="#2a2a3e",
              edgecolor="#555577")


def _draw_proportion_bar(ax, labels, k, color_map=None):
    ids    = np.arange(k)
    counts = np.array([np.sum(labels == i) for i in ids])
    total  = max(counts.sum(), 1)
    left   = 0.0
    for hid, count in zip(ids, counts):
        frac = count / total
        ax.barh(0, frac, left=left, color=_hcol_mapped(hid, color_map),
                height=0.6, linewidth=0)
        if frac > 0.04:
            ax.text(left + frac / 2, 0,
                    f"H{hid}\n{frac*100:.0f}%",
                    ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold")
        left += frac
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=6, color="#aaaaaa")
    ax.tick_params(axis="x", length=2, colors="#aaaaaa")


def _draw_centroid_heatmap(ax, centroids, parameters, k, color):
    if centroids is None or len(centroids) == 0:
        ax.set_visible(False)
        return
    mat   = np.atleast_2d(centroids)[:k]
    nfeat = min(mat.shape[1], len(parameters))
    mat   = mat[:, :nfeat]
    plbls = [p.replace("DTI-", "").replace("T2star", "T2*")
             for p in parameters[:nfeat]]
    vabs  = max(np.nanmax(np.abs(mat)), 1e-6)
    im    = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                      vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(nfeat))
    ax.set_xticklabels(plbls, rotation=45, ha="right",
                       fontsize=6, color="#cccccc")
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([f"H{i}" for i in range(len(mat))],
                       fontsize=6, color="#cccccc")
    ax.set_title("Feature centroids (normalised)", fontsize=8,
                 fontweight="bold", color=color)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors="#cccccc")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if abs(v) < vabs * 0.6 else "white")


# ── Per-mouse PDF page ──────────────────────────────────────────────────────

def _mouse_page(pdf, mouse_name, classic, discovery, joint, date_str,
               settings_str, color_maps=None):
    color_maps = color_maps or {}
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#1a1d23")

    # Header band
    fig.text(0.01, 0.96, f"Mouse: {mouse_name}",
             fontsize=12, fontweight="bold", color="white", va="top")
    fig.text(0.01, 0.935, settings_str,
             fontsize=7, color="#888899", va="top")
    fig.text(0.99, 0.96, date_str,
             fontsize=8, color="#888899", va="top", ha="right")

    gs = fig.add_gridspec(3, 3,
                          height_ratios=[5, 1, 3],
                          left=0.04, right=0.98,
                          top=0.92, bottom=0.06,
                          hspace=0.45, wspace=0.3)

    cols = [
        ("Classic",   classic,   _METHOD_COLORS["classic"],   "classic"),
        ("Discovery", discovery, _METHOD_COLORS["discovery"], "discovery"),
        ("Joint",     joint,     _METHOD_COLORS["joint"],     "joint"),
    ]

    for col, (label, res, hdr_col, cmap_key) in enumerate(cols):
        ax_pca  = fig.add_subplot(gs[0, col])
        ax_bar  = fig.add_subplot(gs[1, col])
        ax_heat = fig.add_subplot(gs[2, col])

        for ax in (ax_pca, ax_bar, ax_heat):
            ax.set_facecolor("#2a2a3e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#555577")

        if res is None:
            for ax in (ax_pca, ax_bar, ax_heat):
                ax.set_visible(False)
            ax_pca.set_visible(True)
            ax_pca.text(0.5, 0.5, f"{label}\nnot available",
                        transform=ax_pca.transAxes,
                        ha="center", va="center",
                        fontsize=10, color="#888888")
            continue

        lbls    = res["labels"]
        feats   = res["features"]
        cents   = res["centroids"]
        params  = res.get("parameters") or res.get("common_params") or []
        k       = res["k"]
        title   = f"{label}  (K={k})"
        c_map   = color_maps.get(cmap_key)

        _draw_pca(ax_pca, feats, lbls, cents, title, hdr_col, color_map=c_map)
        _draw_proportion_bar(ax_bar, lbls, k, color_map=c_map)
        ax_bar.set_title("Habitat proportions", fontsize=7, color="#aaaaaa")
        _draw_centroid_heatmap(ax_heat, cents, params, k, hdr_col)

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Segmentation maps page ──────────────────────────────────────────────────

def _segmentation_page(pdf, mouse_name, classic, discovery, joint,
                       color_maps=None):
    """
    One landscape page per mouse: per-slice habitat maps.
    Layout: 3 columns (Classic | Discovery | Joint) × n_slices rows.
    Classic and Discovery share spatial coords from classic["df"].
    Joint uses its own "df" key (X, Y, Slice from df_pooled subset).
    """
    if classic is None:
        return

    color_maps = color_maps or {}
    df_cl   = classic["df"]
    slices  = sorted(df_cl["Slice"].unique())
    n_sl    = len(slices)

    col_defs = [
        ("Classic",   classic,   _METHOD_COLORS["classic"],
         df_cl,       color_maps.get("classic")),
        ("Discovery", discovery, _METHOD_COLORS["discovery"],
         df_cl,       color_maps.get("discovery")),
        ("Joint",     joint,     _METHOD_COLORS["joint"],
         joint["df"] if (joint is not None and "df" in joint) else df_cl,
         color_maps.get("joint")),
    ]

    row_h   = 2.4
    fig_h   = max(5, n_sl * row_h + 1.2)
    fig     = plt.figure(figsize=(16, fig_h))
    fig.patch.set_facecolor("#1a1d23")

    fig.text(0.01, 0.99,
             f"Mouse: {mouse_name} — Habitat segmentation",
             fontsize=11, fontweight="bold", color="white", va="top")

    gs = fig.add_gridspec(n_sl, 3,
                          left=0.06, right=0.98,
                          top=0.94, bottom=0.05,
                          hspace=0.12, wspace=0.06)

    for row, sl in enumerate(slices):
        for col, (label, res, hdr_col, df_sp, c_map) in enumerate(col_defs):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#0d0d18")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333355")

            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold",
                             color=hdr_col, pad=3)

            if col == 0:
                ax.set_ylabel(f"Slice {int(sl)}", fontsize=7,
                              color="#aaaaaa", rotation=0,
                              labelpad=28, va="center")

            ax.set_xticks([])
            ax.set_yticks([])

            if res is None:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=10, color="#555566")
                continue

            lbls_all = res["labels"]
            xs_all   = df_sp["X"].values.astype(int)
            ys_all   = df_sp["Y"].values.astype(int)
            sl_all   = df_sp["Slice"].values

            sl_mask = sl_all == sl
            if not sl_mask.any():
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=10, color="#555566")
                continue

            xs   = xs_all[sl_mask] - xs_all[sl_mask].min()
            ys   = ys_all[sl_mask] - ys_all[sl_mask].min()
            lbls = lbls_all[sl_mask]

            W = xs.max() + 1
            H = ys.max() + 1

            img  = np.zeros((H, W, 4), dtype=np.float32)
            rgba = np.array([_hcol_mapped(int(l), c_map) for l in lbls],
                            dtype=np.float32)
            img[ys, xs, :3] = rgba[:, :3]
            img[ys, xs,  3] = 1.0

            ax.imshow(img, origin="lower", aspect="equal",
                      interpolation="nearest")

    # Colour legend — one row per method
    import matplotlib.patches as mpatches
    leg_ax = fig.add_axes([0.06, 0.0, 0.88, 0.04])
    leg_ax.set_facecolor("#1a1d23")
    leg_ax.axis("off")
    x_cursor = 0.0
    for label, res, hdr_col, df_sp, c_map in col_defs:
        if res is None:
            continue
        unique_labs = sorted(np.unique(res["labels"]))
        n_labs      = len(unique_labs)
        block_w     = 1.0 / 3
        slot        = block_w / max(n_labs, 1)
        for i, hab in enumerate(unique_labs):
            col_rgba = _hcol_mapped(hab, c_map)
            x0 = x_cursor + i * slot
            leg_ax.add_patch(mpatches.Rectangle(
                (x0 + slot * 0.05, 0.15), slot * 0.25, 0.70,
                color=col_rgba[:3], transform=leg_ax.transAxes, clip_on=False))
            leg_ax.text(x0 + slot * 0.35, 0.5, f"H{hab}",
                        transform=leg_ax.transAxes, va="center",
                        fontsize=7, color="#cccccc")
        leg_ax.text(x_cursor + block_w * 0.5, -0.3, label,
                    transform=leg_ax.transAxes, va="top",
                    ha="center", fontsize=6, color=hdr_col)
        x_cursor += block_w

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Cover page ──────────────────────────────────────────────────────────────

def _cover_page(pdf, mice_names, method, k_range, k_override, norm, date_str):
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#1a1d23")
    ax.set_facecolor("#1a1d23")
    ax.axis("off")

    ax.text(0.5, 0.88, "Comparative Habitat Analysis",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=22, fontweight="bold", color="white")
    ax.text(0.5, 0.80, "Classic  |  Discovery  |  Joint",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=14, color="#4A90D9")
    ax.text(0.5, 0.73, date_str,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color="#888899")

    k_txt = (f"K forced = {k_override}" if str(k_override).strip().isdigit()
             else f"K range = {min(k_range)}–{max(k_range)}")
    settings = (f"Method: {method.upper()}    {k_txt}    "
                f"Joint norm: {norm}")
    ax.text(0.5, 0.66, settings,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#aaaaaa")

    # Mice list
    ax.text(0.5, 0.57, f"Mice ({len(mice_names)})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color="white")
    for i, name in enumerate(mice_names):
        ax.text(0.5, 0.50 - i * 0.055, f"• {name}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="#cccccc")

    # Legend
    for j, (label, col) in enumerate([
        ("Classic — per-mouse independent clustering", _METHOD_COLORS["classic"]),
        ("Discovery — meta-clustering of classic centroids", _METHOD_COLORS["discovery"]),
        ("Joint — single clustering on pooled voxels", _METHOD_COLORS["joint"]),
    ]):
        y = 0.18 - j * 0.07
        ax.add_patch(plt.Rectangle((0.18, y - 0.015), 0.03, 0.03,
                                    transform=ax.transAxes,
                                    color=col, clip_on=False))
        ax.text(0.23, y, label,
                transform=ax.transAxes, va="center",
                fontsize=9, color="#cccccc")

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main entry point ────────────────────────────────────────────────────────

def run_comparative_pdf(mice_data, method, k_range, k_override,
                         n_init, n_refs, norm,
                         dtopo_weight=0.0,
                         out_dir="", pdf_name="comparative_analysis", log=print):
    """
    Parameters
    ----------
    mice_data  : list of {'name': str, 'files': [csv_paths]}
    method     : 'gmm' or 'kmeans'
    k_range    : range object
    k_override : str — digit string to force K, else ''
    n_init     : int
    n_refs     : int
    norm       : 'robust_global', 'robust', 'cl_hybrid', 'cl_global', or 'cl'  (for Joint step)
    out_dir    : directory where PDF is saved
    pdf_name   : filename without extension
    log        : callable(str[, tag])

    Returns
    -------
    dict with 'pdf_path', 'n_mice', 'classic_ks', 'joint_k', 'disc_k'
    """
    os.makedirs(out_dir, exist_ok=True)

    date_str    = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    k_txt       = (f"K={k_override}" if str(k_override).strip().isdigit()
                   else f"K {min(k_range)}–{max(k_range)}")
    settings_str = f"method={method.upper()}  {k_txt}  n_init={n_init}  joint_norm={norm}"

    # ── Step 1: Classic ──────────────────────────────────────────────────────
    log("Step 1/3 — Classic analysis")
    classic_results  = {}
    result_csv_paths = {}

    for mouse in mice_data:
        name  = mouse["name"]
        files = mouse["files"]
        log(f"  [{name}]")
        c_out = os.path.join(out_dir, "classic", name)
        res   = run_classic_for_mouse(
            files, method, k_range, k_override, n_init, n_refs, c_out, log)
        if res:
            classic_results[name]  = res
            result_csv_paths[name] = os.path.join(c_out, "habitats_result.csv")

    # ── Step 2: Discovery ────────────────────────────────────────────────────
    log("Step 2/3 — Discovery meta-clustering")
    disc_per_mouse = None
    if len(classic_results) >= 2:
        disc_per_mouse = run_discovery_meta(
            classic_results,
            method, k_range, k_override, n_init, log)
    else:
        log("  Discovery skipped (need ≥ 2 mice)")

    # ── Step 3: Joint ────────────────────────────────────────────────────────
    log("Step 3/3 — Joint clustering")
    joint_per_mouse = {}
    cents_arr  = None   # global joint centroids (k, n_params) for colour mapping
    vis_params = []     # parameter names matching cents_arr columns
    try:
        mice_list_jt = [{"name": m["name"], "files": m["files"]} for m in mice_data]
        jt_out       = os.path.join(out_dir, "joint")
        res_jt       = run_joint_pipeline(
            mice_list_jt, method, k_range, n_init, n_refs,
            output_dir     = jt_out,
            log_fn         = log,
            k_override     = (int(k_override) if str(k_override).strip().isdigit() else None),
            normalization  = norm,
            dtopo_weight   = dtopo_weight,
            dtopo_min_frac = DTOPO_MIN_NECROSIS_FRACTION,
        )
        if res_jt and "df_pooled" in res_jt:
            df_jt     = res_jt["df_pooled"]
            jt_params = res_jt.get("parameters", [])
            jt_cents  = res_jt.get("centroids")   # dict {cluster_id: {param: value}}
            jt_k      = res_jt.get("best_k",
                                    int(df_jt["Habitat"].nunique()))

            # Strip d_topo_norm — derived spatial feature, not a raw MRI param.
            # Also convert centroids dict-of-dicts → numpy array (k, n_params)
            # so PCA gets consistent (k, n_params) input instead of a dict.
            vis_params = [p for p in jt_params if p != "d_topo_norm"]
            if jt_cents is not None and vis_params:
                cents_arr = np.array([
                    [jt_cents[c].get(p, 0.0) for p in vis_params]
                    for c in range(jt_k)
                    if c in jt_cents
                ])
            else:
                cents_arr = None

            for mouse in mice_data:
                name = mouse["name"]
                mask = df_jt["mouse_name"] == name
                if not mask.any():
                    continue
                sub     = df_jt[mask].reset_index(drop=True)
                feats_m = sub[vis_params].values if vis_params else np.empty((0, 0))
                labs_m  = sub["Habitat"].values
                # Per-mouse centroids: mean of each mouse's own voxels per habitat
                cents_m = np.array([
                    feats_m[labs_m == c].mean(axis=0)
                    if (labs_m == c).any() else np.zeros(len(vis_params))
                    for c in range(jt_k)
                ]) if vis_params else cents_arr
                joint_per_mouse[name] = {
                    "labels"    : labs_m,
                    "features"  : feats_m,
                    "parameters": vis_params,
                    "centroids" : cents_m,
                    "k"         : jt_k,
                    "df"        : sub[["X", "Y", "Slice"]].copy(),
                }
    except Exception as e:
        log(f"  [Joint] Error: {e}")

    # ── Step 4: Build semantic colour maps ───────────────────────────────────
    # Discovery global centroids come from the _meta key added by run_discovery_meta.
    disc_meta    = (disc_per_mouse or {}).pop("_meta", None)
    disc_col_map = {}
    jt_col_map   = {}

    try:
        disc_cents_g  = disc_meta["centroids"] if disc_meta else None
        disc_params_g = disc_meta["params"]    if disc_meta else []
        disc_labels_g = (
            np.concatenate([disc_per_mouse[n]["labels"]
                            for n in disc_per_mouse])
            if disc_per_mouse else None
        )

        jt_cents_g  = cents_arr if cents_arr is not None else None
        jt_params_g = vis_params
        jt_labels_g = (
            np.concatenate([joint_per_mouse[n]["labels"]
                            for n in joint_per_mouse])
            if joint_per_mouse else None
        )

        if (disc_cents_g is not None and disc_labels_g is not None
                and jt_cents_g is not None and jt_labels_g is not None):
            disc_col_map, jt_col_map = _build_cross_color_maps(
                disc_cents_g, disc_labels_g, disc_params_g,
                jt_cents_g,   jt_labels_g,   jt_params_g,
            )
        elif disc_cents_g is not None and disc_labels_g is not None:
            disc_col_map = _single_color_map(
                disc_cents_g, disc_labels_g, disc_params_g)
        elif jt_cents_g is not None and jt_labels_g is not None:
            jt_col_map = _single_color_map(
                jt_cents_g, jt_labels_g, jt_params_g)
    except Exception as _ce:
        log(f"  [colour] Semantic mapping skipped: {_ce}")

    # ── Step 5: Generate PDF ─────────────────────────────────────────────────
    log("Generating PDF…")
    pdf_path = os.path.join(out_dir, f"{pdf_name}.pdf")

    mice_names = [m["name"] for m in mice_data]

    with PdfPages(pdf_path) as pdf:
        _cover_page(pdf, mice_names, method, k_range, k_override, norm, date_str)

        for mouse in mice_data:
            name = mouse["name"]
            log(f"  Page: {name}")
            cl  = classic_results.get(name)
            dis = (disc_per_mouse or {}).get(name)
            jt  = joint_per_mouse.get(name)

            # Classic: per-mouse signature colours (K may differ per mouse)
            cl_col_map = {}
            if cl is not None:
                try:
                    cl_col_map = _single_color_map(
                        cl["centroids"], cl["labels"], cl["parameters"])
                except Exception:
                    pass

            color_maps = {
                "classic"  : cl_col_map,
                "discovery": disc_col_map,
                "joint"    : jt_col_map,
            }

            _mouse_page(
                pdf, name,
                classic      = cl,
                discovery    = dis,
                joint        = jt,
                date_str     = date_str,
                settings_str = settings_str,
                color_maps   = color_maps,
            )
            _segmentation_page(pdf, name, cl, dis, jt, color_maps=color_maps)

    log(f"PDF saved → {pdf_path}", "ok")

    classic_ks = {name: r["k"] for name, r in classic_results.items()}
    disc_k     = next(iter((disc_per_mouse or {}).values()), {}).get("k")
    joint_k    = next(iter(joint_per_mouse.values()), {}).get("k")

    return {
        "pdf_path"  : pdf_path,
        "n_mice"    : len(mice_data),
        "classic_ks": classic_ks,
        "disc_k"    : disc_k,
        "joint_k"   : joint_k,
    }
