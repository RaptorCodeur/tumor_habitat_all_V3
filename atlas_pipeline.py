"""
atlas_pipeline.py
Inter-animal habitat comparison via frozen weighted-GMM atlas.

Pipeline
--------
1.  Load + normalise all mice  (reuses pipeline/joint_loading.py)
2.  Select K via weighted-BIC (or use k_override)
3.  Fit frozen weighted-GMM atlas  (1/n_i weight per voxel)
4.  Bootstrap stability check
5.  Soft-assign each mouse → π, σ², residuals
6.  Statistical tests
    a. Dirichlet regression on π
    b. PERMANOVA + PERMDISP on Aitchison distances of π
    c. PERMANOVA + PERMDISP on MMD pairwise distances (full distribution)
    d. Conditional MMD per habitat
7.  Figures + PDF report

Entry point
-----------
    results = run_atlas_pipeline(mice_data, group_labels, group_names, ...)
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from pipeline.joint_loading  import load_joint_dataset
from pipeline.habitat_atlas  import WeightedAtlas, select_atlas_k
from pipeline.inter_animal_stats import (
    mmd_pairwise_matrix, permanova, permdisp,
    aitchison_dist_matrix, clr_transform,
    dirichlet_regression, mmd_per_habitat,
    witness_2d, top_witness_pairs,
    pi_per_habitat_tests,
)
from config import RANDOM_SEED, GMM_SILHOUETTE_GUARD

# ── Palette ──────────────────────────────────────────────────────────────────

_GROUP_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF", "#888888",
]

_HAB_COLORS = [
    "#E85D24", "#3B8BD4", "#2EAA72", "#9B59B6",
    "#F39C12", "#FF69B4", "#7B3F00", "#E91E8C",
]


def _gcol(g: int) -> str:
    return _GROUP_COLORS[int(g) % len(_GROUP_COLORS)]


def _hcol(h: int) -> str:
    return _HAB_COLORS[int(h) % len(_HAB_COLORS)]


def _sig_stars(p: float) -> str:
    if np.isnan(p):   return "n/a"
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

def _fig_atlas_profiles(atlas: WeightedAtlas,
                         group_labels: np.ndarray,
                         mice_descs: list[dict],
                         mouse_names: list[str],
                         group_names: dict) -> plt.Figure:
    """
    Atlas centroid heatmap (K × n_params) with bootstrap stability bars,
    and a per-mouse π composition strip beneath.
    """
    k      = atlas.k
    params = atlas.parameters
    cents  = atlas.centroids_
    stab   = atlas.stability_

    fig    = plt.figure(figsize=(max(10, len(params) * 1.2 + 2), 7))
    gs     = mgrid.GridSpec(2, 1, figure=fig, height_ratios=[3, 1.5],
                            hspace=0.45)

    # ── Centroid heatmap ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    vabs = float(np.nanmax(np.abs(cents))) or 1.0
    im   = ax1.imshow(cents, aspect="auto", cmap="RdBu_r",
                      vmin=-vabs, vmax=vabs)
    ax1.set_xticks(range(len(params)))
    ax1.set_xticklabels(params, rotation=40, ha="right", fontsize=8)
    ax1.set_yticks(range(k))
    ax1.set_yticklabels([f"H{h}" for h in range(k)], fontsize=9)
    ax1.set_title("Atlas centroids  (robust-scaled)  ·  stability bars = bootstrap ±1σ",
                  fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)

    for h in range(k):
        for p_idx in range(len(params)):
            v = cents[h, p_idx]
            ax1.text(p_idx, h, f"{v:+.2f}", ha="center", va="center",
                     fontsize=6.5,
                     color="white" if abs(v) > vabs * 0.6 else "black")

    # Bootstrap stability error bars (one per habitat, over the param axis)
    if "mean_shift" in stab:
        ms = stab["mean_shift"]         # (K,)
        for h in range(k):
            ax1.annotate(f"Δ={ms[h]:.3f}", xy=(-0.7, h),
                         xycoords=("data", "data"),
                         fontsize=6, color="#555555", va="center", ha="right")

    # ── π composition bar per mouse ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    n   = len(mouse_names)
    bar_w = 0.7
    for mi, desc in enumerate(mice_descs):
        pi   = desc["pi"]
        left = 0.0
        for h in range(k):
            ax2.barh(mi, pi[h], left=left, color=_hcol(h),
                     height=bar_w, linewidth=0)
            if pi[h] > 0.06:
                ax2.text(left + pi[h] / 2, mi, f"H{h}\n{pi[h]*100:.0f}%",
                         ha="center", va="center", fontsize=5.5,
                         color="white", fontweight="bold")
            left += pi[h]

    g_arr = group_labels.astype(int)
    for mi, name in enumerate(mouse_names):
        gc  = _gcol(g_arr[mi])
        gn  = group_names.get(int(g_arr[mi]), f"G{g_arr[mi]}")
        ax2.text(-0.01, mi, f"{name} [{gn}]", ha="right", va="center",
                 fontsize=7, color=gc, transform=ax2.get_yaxis_transform())

    ax2.set_xlim(0, 1)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([""] * n)
    ax2.set_xlabel("Habitat proportion (soft composition π)", fontsize=8)
    ax2.set_title("Per-mouse soft composition", fontsize=9, fontweight="bold")
    ax2.invert_yaxis()

    fig.tight_layout()
    return fig


def _fig_intra_variance(atlas: WeightedAtlas,
                         mice_descs: list[dict],
                         mouse_names: list[str],
                         group_labels: np.ndarray,
                         group_names: dict) -> plt.Figure:
    """σ² (intra-habitat heterogeneity) and atlas residuals as heatmaps."""
    k = atlas.k
    n = len(mouse_names)
    g_arr = group_labels.astype(int)

    sigma2_mat = np.array([d["sigma2"]   for d in mice_descs])   # (n, K)
    resid_mat  = np.array([d["residual"] for d in mice_descs])    # (n, K)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, k * 1.6 + 4), max(4, n * 0.45 + 1.5)))

    for ax, mat, title, cmap in [
        (axes[0], sigma2_mat, "σ²  Intra-habitat heterogeneity\n(high = voxels spread, biology heterogeneous)", "YlOrRd"),
        (axes[1], resid_mat,  "Atlas residual — mean dist to centroid\n(high = mouse biology doesn't fit atlas)", "PuRd"),
    ]:
        vmax = float(np.nanmax(np.abs(mat))) or 1.0
        im   = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"H{h}" for h in range(k)], fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f"{nm}  [{group_names.get(int(g_arr[mi]), '')}]"
             for mi, nm in enumerate(mouse_names)],
            fontsize=7)
        ax.set_title(title, fontsize=8, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04).ax.tick_params(labelsize=7)

        for i in range(n):
            for j in range(k):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if v > vmax * 0.6 else "black")

    fig.suptitle("Intra-habitat heterogeneity & atlas fit quality",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_mmd_matrix(D_sq: np.ndarray,
                     mouse_names: list[str],
                     group_labels: np.ndarray,
                     group_names: dict,
                     perm_res: dict) -> plt.Figure:
    """MMD² pairwise distance matrix + PERMANOVA summary."""
    n     = len(mouse_names)
    g_arr = group_labels.astype(int)
    D_dist = np.sqrt(np.maximum(D_sq, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.6 + 2)),
                              gridspec_kw={"width_ratios": [2, 1]})

    # Heatmap
    ax = axes[0]
    im = ax.imshow(D_dist, cmap="viridis", aspect="auto")
    labels_tick = [f"{nm}\n[{group_names.get(int(g_arr[i]), '')}]"
                   for i, nm in enumerate(mouse_names)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_tick, rotation=45,
                                                  ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_tick, fontsize=7)
    ax.set_title("MMD pairwise distance matrix\n√MMD²(P_i, P_j)",
                 fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04).ax.tick_params(labelsize=7)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{D_dist[i,j]:.3f}", ha="center", va="center",
                    fontsize=5.5, color="white" if D_dist[i,j] > D_dist.max()*0.5
                    else "black")

    # Draw group borders
    borders = []
    cur_g   = g_arr[0]
    start   = 0
    for idx in range(1, n + 1):
        if idx == n or g_arr[idx] != cur_g:
            borders.append((start - 0.5, idx - 0.5, _gcol(cur_g)))
            cur_g  = g_arr[idx] if idx < n else cur_g
            start  = idx
    for x0, x1, col in borders:
        for a in (ax,):
            a.add_patch(plt.Rectangle((x0, x0), x1-x0, x1-x0,
                                       fill=False, edgecolor=col, lw=2.0))

    # PERMANOVA summary table
    ax2 = axes[1]
    ax2.axis("off")
    F, p, R2 = perm_res["F"], perm_res["p_value"], perm_res["R2"]
    rows = [
        ["Test",       "PERMANOVA on MMD"],
        ["F statistic", f"{F:.3f}"],
        ["p-value",     f"{p:.4f}  {_sig_stars(p)}"],
        ["R²",          f"{R2:.3f}  ({R2*100:.1f}% variance)"],
        ["n perm",      str(perm_res["n_perm"])],
    ]
    tbl = ax2.table(cellText=rows, loc="center", cellLoc="left",
                    colWidths=[0.45, 0.55])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl[(0, 0)].set_facecolor("#1F3864"); tbl[(0, 0)].set_text_props(color="white")
    tbl[(0, 1)].set_facecolor("#1F3864"); tbl[(0, 1)].set_text_props(color="white")

    color_row = "#c0e0ff" if p < 0.05 else "#ffffff"
    for row in range(1, len(rows)):
        for col in range(2):
            tbl[(row, col)].set_facecolor(color_row if row == 2 else "#f5f5f5")

    ax2.set_title("PERMANOVA results", fontsize=9, fontweight="bold")
    fig.suptitle("MMD pairwise distances between mouse distributions",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_dirichlet(dirich_res: dict,
                    atlas: WeightedAtlas,
                    group_names: dict) -> plt.Figure:
    """
    Per-habitat group coefficients from Dirichlet regression.
    Positive coeff → group has MORE of that habitat vs reference.
    """
    coef   = dirich_res["group_coef"]   # (n_groups-1, K)
    k      = atlas.k
    unique = dirich_res["unique_groups"]
    ref_g  = unique[-1]
    n_g    = coef.shape[0]

    fig, ax = plt.subplots(figsize=(max(8, k * 1.5 + 2), 4 + n_g * 0.8))
    x  = np.arange(k)
    bw = 0.7 / max(n_g, 1)

    for gi in range(n_g):
        g_id  = unique[gi]
        color = _gcol(int(g_id))
        gname = group_names.get(int(g_id), f"G{g_id}")
        off   = (gi - n_g / 2 + 0.5) * bw
        ax.bar(x + off, coef[gi], width=bw * 0.9,
               color=color, alpha=0.85, label=f"{gname}  (vs G{ref_g})")

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(k)], fontsize=9)
    ax.set_ylabel("Log-concentration coefficient  (log α ratio)", fontsize=9)
    ax.set_title(
        f"Dirichlet regression — per-habitat group effects\n"
        f"(LR p={dirich_res['p_value']:.4f}  {_sig_stars(dirich_res['p_value'])}  "
        f"df={dirich_res['df']})",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_conditional_mmd(cond_results: list[dict],
                           group_names: dict,
                           group_a: list[int],
                           group_b: list[int]) -> plt.Figure:
    """Bar chart of conditional MMD² per habitat with significance stars."""
    k       = len(cond_results)
    habs    = [r["habitat"]  for r in cond_results]
    mmd2s   = [r["mmd_sq"]   for r in cond_results]
    pvals   = [r["p_value"]  for r in cond_results]
    n_as    = [r["n_a"]      for r in cond_results]
    n_bs    = [r["n_b"]      for r in cond_results]

    ga_name = group_names.get(group_a[0], f"G{group_a[0]}") if group_a else "A"
    gb_name = group_names.get(group_b[0], f"G{group_b[0]}") if group_b else "B"

    fig, ax = plt.subplots(figsize=(max(7, k * 1.4 + 1.5), 4))
    x       = np.arange(k)
    colors  = [_hcol(h) for h in habs]

    bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in mmd2s],
                  color=colors, alpha=0.85, edgecolor="black", lw=0.5)

    for i, (m, p) in enumerate(zip(mmd2s, pvals)):
        if np.isnan(m):
            ax.text(i, 0.005, "n/a", ha="center", va="bottom", fontsize=7)
        else:
            stars = _sig_stars(p)
            ax.text(i, m + max(mmd2s) * 0.02, stars,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="red" if p < 0.05 else "#888888")
        # nA / nB annotation
        ax.text(i, -max([v for v in mmd2s if not np.isnan(v)] or [0.01]) * 0.12,
                f"n={n_as[i]}/{n_bs[i]}",
                ha="center", va="top", fontsize=6, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in habs], fontsize=9)
    ax.set_ylabel("Conditional MMD²", fontsize=9)
    ax.set_title(
        f"Conditional MMD per habitat  —  {ga_name} vs {gb_name}\n"
        "Tests whether same atlas habitat has same biology across groups",
        fontsize=9, fontweight="bold",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="#cccccc", label="ns"),
        plt.Rectangle((0, 0), 1, 1, fc="#ff4444", label="p<0.05 *"),
    ], fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def _fig_pi_habitat_tests(pi_results: list[dict],
                           pi_matrix: np.ndarray,
                           group_labels: np.ndarray,
                           group_names: dict) -> plt.Figure:
    """
    Per-habitat Mann-Whitney on π with Bonferroni correction.
    Grouped bars (one per group) + significance stars.
    """
    k        = len(pi_results)
    groups   = np.asarray(group_labels)
    unique_g = np.unique(groups)
    n_groups = len(unique_g)
    x        = np.arange(k)
    bw       = 0.7 / n_groups

    fig, ax  = plt.subplots(figsize=(max(7, k * 1.5 + 2), 5))

    for gi, g in enumerate(unique_g):
        medians = [r["median_per_group"].get(int(g), 0.0) for r in pi_results]
        vals    = [pi_matrix[groups == g, h] for h in range(k)]
        offset  = (gi - n_groups / 2 + 0.5) * bw
        gname   = group_names.get(int(g), f"G{g}")

        bars = ax.bar(x + offset, medians, width=bw * 0.9,
                      color=_gcol(int(g)), alpha=0.85, label=gname)

        # Individual mouse dots
        for h in range(k):
            jitter = np.random.default_rng(h + gi).uniform(
                -bw * 0.2, bw * 0.2, size=len(vals[h]))
            ax.scatter(x[h] + offset + jitter, vals[h],
                       color=_gcol(int(g)), s=18, zorder=3,
                       edgecolors="white", linewidths=0.5)

    # Significance stars above each habitat
    y_max = pi_matrix.max() * 1.05
    for r in pi_results:
        h     = r["habitat"]
        p_adj = r["p_adj"]
        p_raw = r["p_raw"]
        if p_adj < 0.001:   stars = "***"
        elif p_adj < 0.01:  stars = "**"
        elif p_adj < 0.05:  stars = "*"
        elif p_raw < 0.05:  stars = f"(~{p_raw:.2f})"
        else:               stars = f"ns\n({p_raw:.2f})"
        color = "red" if p_adj < 0.05 else ("#888888" if p_raw < 0.05 else "#aaaaaa")
        ax.text(h, y_max, stars, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(k)], fontsize=9)
    ax.set_ylabel("Soft proportion π", fontsize=9)
    ax.set_ylim(0, y_max * 1.18)
    ax.set_title(
        f"Per-habitat Mann-Whitney on π  —  Bonferroni corrected (×{k})\n"
        "Bars = median per group  ·  dots = individual mice  "
        "·  * p_adj<0.05  ·  (~) p_raw<0.05 before correction",
        fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_habitat_maps(df_pooled, habitat_labels: np.ndarray,
                       mouse_names: list[str],
                       g_arr: np.ndarray,
                       group_names: dict,
                       k: int) -> list[tuple]:
    """
    Spatial habitat maps — one figure per mouse.
    Each slice is rendered as a filled 2-D grid (imshow) so habitat regions
    appear as solid colour blocks rather than tiny dots.
    Returns list of (fig, caption).
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    hab_colors = [_hcol(h) for h in range(k)]
    cmap       = ListedColormap(hab_colors)

    figs = []
    mouse_ids_col = df_pooled["mouse_id"].values

    for i, name in enumerate(mouse_names):
        mask   = mouse_ids_col == i
        df_m   = df_pooled[mask]
        labs_m = habitat_labels[mask]

        slices   = sorted(df_m["Slice"].unique())
        n_slices = len(slices)

        fig, axes = plt.subplots(1, n_slices,
                                  figsize=(max(n_slices * 3.2, 5), 3.8),
                                  squeeze=False)

        for col, sl in enumerate(slices):
            ax     = axes[0, col]
            s_mask = df_m["Slice"].values == sl
            xs     = df_m["X"].values[s_mask].astype(int)
            ys     = df_m["Y"].values[s_mask].astype(int)
            ls     = labs_m[s_mask]

            # Build a 2-D grid: -1 = background (no voxel)
            x0, y0 = xs.min(), ys.min()
            grid   = np.full((ys.max() - y0 + 1, xs.max() - x0 + 1), -1, dtype=float)
            grid[ys - y0, xs - x0] = ls

            # Show background as light grey, habitats with the palette
            bg = np.ma.masked_where(grid >= 0, grid)
            ax.imshow(np.zeros_like(grid), cmap="gray", vmin=0, vmax=1,
                      origin="upper", aspect="equal")
            ax.imshow(np.ma.masked_where(grid < 0, grid),
                      cmap=cmap, vmin=0, vmax=k - 1,
                      origin="upper", aspect="equal", interpolation="nearest")

            ax.set_title(f"Slice {int(sl)}", fontsize=8)
            ax.axis("off")

        present = sorted(np.unique(labs_m).astype(int))
        legend_patches = [Patch(facecolor=_hcol(h), label=f"H{h}") for h in present]
        axes[0, -1].legend(handles=legend_patches, loc="lower right",
                            fontsize=7, framealpha=0.8)

        gname = group_names.get(int(g_arr[i]), f"G{g_arr[i]}")
        fig.suptitle(f"{name}  [{gname}]", fontsize=9, fontweight="bold")
        fig.tight_layout()
        figs.append((fig, f"Habitat map — {name} [{gname}]"))

    return figs


def _fig_witness(witness_data: list[dict],
                  ga_name: str, gb_name: str) -> plt.Figure:
    """Top-N witness function 2D projections."""
    n   = len(witness_data)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 4.5))
    if n == 1:
        axes = [axes]

    for ax, wd in zip(axes, witness_data):
        w   = wd["witness"]
        vabs = float(np.nanmax(np.abs(w))) or 1.0
        norm = TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)
        ax.contourf(wd["GI"], wd["GJ"], w, levels=30, cmap="RdBu_r", norm=norm)
        ax.contour(wd["GI"], wd["GJ"], w, levels=[0], colors="black", lw=0.8)
        ax.set_xlabel(wd["param_i"], fontsize=9)
        ax.set_ylabel(wd["param_j"], fontsize=9)
        ax.set_title(f"Witness: {wd['param_i']} × {wd['param_j']}\n"
                     f"Blue={ga_name} enriched  ·  Red={gb_name} enriched",
                     fontsize=8, fontweight="bold")

    fig.suptitle("MMD witness function  —  discriminating feature regions",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_permanova_summary(perm_aitch: dict,
                            perm_mmd: dict,
                            pdisp_aitch: dict,
                            pdisp_mmd: dict,
                            dirich_res: dict) -> plt.Figure:
    """Single-page summary table of all statistical results."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    def _row(name, F, p, R2=None, note=""):
        r2_str = f"{R2:.3f}" if R2 is not None else "—"
        return [name, f"{F:.3f}", f"{p:.4f}", _sig_stars(p), r2_str, note]

    rows_data = [
        _row("PERMANOVA · Aitchison (π)",
             perm_aitch["F"], perm_aitch["p_value"], perm_aitch["R2"],
             "Composition location shift"),
        _row("PERMDISP · Aitchison (π)",
             pdisp_aitch["F"], pdisp_aitch["p_value"],
             note=pdisp_aitch["interpretation"]),
        _row("PERMANOVA · MMD (full dist.)",
             perm_mmd["F"], perm_mmd["p_value"], perm_mmd["R2"],
             "Distributional location shift"),
        _row("PERMDISP · MMD",
             pdisp_mmd["F"], pdisp_mmd["p_value"],
             note=pdisp_mmd["interpretation"]),
        ["Dirichlet regression",
         f"LR={dirich_res['lr_stat']:.2f}",
         f"{dirich_res['p_value']:.4f}",
         _sig_stars(dirich_res["p_value"]),
         f"df={dirich_res['df']}",
         "Habitat-specific group effects"],
    ]

    col_labels = ["Test", "F / LR", "p-value", "Sig.", "R² / df", "Interpretation"]
    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#1F3864")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        tbl[(0, j)].set_height(0.12)

    for row_idx in range(1, len(rows_data) + 1):
        p_val = float(rows_data[row_idx - 1][2])
        bg    = "#c8e6c9" if p_val < 0.05 else "#f5f5f5"
        for j in range(len(col_labels)):
            tbl[(row_idx, j)].set_facecolor(bg)
            tbl[(row_idx, j)].set_height(0.10)

    ax.set_title("Statistical test summary", fontsize=11,
                 fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PDF report
# ═════════════════════════════════════════════════════════════════════════════

def _save_fig(fig: plt.Figure, path: str) -> str:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_pdf(pdf_path: str, fig_paths: list[tuple[str, str]],
               cover_text: str):
    import matplotlib.image as mpimg
    with PdfPages(pdf_path) as pdf:
        # Cover
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.9, "Atlas-Based Inter-Animal Habitat Comparison",
                ha="center", fontsize=18, fontweight="bold",
                transform=ax.transAxes, color="#1F3864")
        ax.text(0.08, 0.78, cover_text,
                va="top", fontsize=10, family="monospace",
                transform=ax.transAxes, color="#333333")
        pdf.savefig(fig); plt.close(fig)

        for fpath, caption in fig_paths:
            if not os.path.exists(fpath):
                continue
            img   = mpimg.imread(fpath)
            h, w  = img.shape[:2]
            fw    = 11.0
            fh    = min(8.5, fw * h / w) + 0.4
            fig2  = plt.figure(figsize=(fw, fh))
            ax2   = fig2.add_axes([0, 0.05, 1, 0.92])
            ax2.imshow(img); ax2.axis("off")
            fig2.text(0.5, 0.01, caption, ha="center",
                      fontsize=8, color="#555555")
            pdf.savefig(fig2, bbox_inches="tight"); plt.close(fig2)


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_atlas_pipeline(
    mice_data      : list[dict],
    group_labels   : list[int],
    group_names    : dict,
    k_range        : range           = range(2, 8),
    k_override     : int | None      = None,
    normalization  : str             = "robust",
    n_init         : int             = 20,
    n_bootstrap    : int             = 30,
    n_perm         : int             = 999,
    n_perm_mmd_cond: int             = 199,
    mmd_max_voxels : int             = 2000,
    out_dir        : str             = "atlas_comparison",
    covariates     : np.ndarray | None = None,
    covar_names    : list[str]       = (),
    log                              = print,
) -> dict:
    """
    Full atlas comparison pipeline.

    Parameters
    ----------
    mice_data      : list of {'name': str, 'files': [csv_paths]}
    group_labels   : list of int, one per mouse (same order as mice_data)
    group_names    : {int: str}  human-readable group labels
    k_range        : K candidates for BIC selection
    k_override     : force K (skips BIC selection)
    normalization  : 'robust' or 'cl'
    n_init         : GMM restarts
    n_bootstrap    : bootstrap runs for stability check
    n_perm         : permutations for PERMANOVA / PERMDISP
    n_perm_mmd_cond: permutations for conditional MMD per habitat
    mmd_max_voxels : max voxels per mouse for MMD (subsampled if larger)
    out_dir        : output directory

    Returns
    -------
    dict with keys: atlas, pi_matrix, sigma2_matrix, residual_matrix,
                    mmd_sq_matrix, permanova_aitch, permanova_mmd,
                    permdisp_aitch, permdisp_mmd, dirichlet, mmd_per_hab,
                    pdf_path, out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    g_arr      = np.array(group_labels, dtype=int)
    mouse_names = [m["name"] for m in mice_data]
    n_mice     = len(mice_data)
    date_str   = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    log("═══ Step 1 / 6 — Load & normalise ═══")
    common_params, df_pooled = load_joint_dataset(
        mice_data, normalization=normalization, log_fn=log)
    features   = df_pooled[common_params].values.astype(np.float32)
    mouse_ids  = df_pooled["mouse_id"].values.astype(int)
    log(f"  {len(features)} total voxels  ×  {len(common_params)} params")

    log("═══ Step 2 / 6 — Select K ═══")
    if k_override and k_override >= 2:
        best_k   = k_override
        bic_dict = {}
        log(f"  K forced = {best_k}")
    else:
        best_k, bic_dict = select_atlas_k(
            features, mouse_ids, k_range, n_init,
            seed=RANDOM_SEED,
            gmm_silhouette_guard=GMM_SILHOUETTE_GUARD,
            out_dir=out_dir,
            log=log)

    log(f"═══ Step 3 / 6 — Fit frozen atlas  (K={best_k}) ═══")
    atlas = WeightedAtlas(k=best_k, n_init=n_init, seed=RANDOM_SEED)
    atlas.fit(features, mouse_ids, common_params)

    log(f"═══ Step 3b — Bootstrap stability  ({n_bootstrap} runs) ═══")
    atlas.bootstrap_stability(features, mouse_ids, n_bootstrap, log=log)

    log("═══ Step 4 / 6 — Soft-assign all mice ═══")
    mice_features  = []
    mice_posteriors = []
    mice_descs     = []

    for i, mouse in enumerate(mice_data):
        mask  = mouse_ids == i
        feats = features[mask]
        mice_features.append(feats)
        post = atlas.posteriors(feats)
        mice_posteriors.append(post)
        desc = atlas.describe_mouse(feats)
        mice_descs.append(desc)
        log(f"  [{mouse['name']}]  π = {np.round(desc['pi'], 3)}")

    pi_matrix       = np.array([d["pi"]       for d in mice_descs])  # (n, K)
    sigma2_matrix   = np.array([d["sigma2"]   for d in mice_descs])
    residual_matrix = np.array([d["residual"] for d in mice_descs])
    habitat_labels  = atlas.gmm_.predict(features)   # hard label per voxel

    log("═══ Step 5 / 6 — Statistical tests ═══")

    # 5a. Aitchison distances + PERMANOVA + PERMDISP
    log("  [5a] Aitchison PERMANOVA …")
    D_aitch     = aitchison_dist_matrix(pi_matrix)
    perm_aitch  = permanova(D_aitch, g_arr, n_perm=n_perm)
    pdisp_aitch = permdisp(D_aitch, g_arr, n_perm=n_perm)
    log(f"    PERMANOVA  F={perm_aitch['F']:.3f}  p={perm_aitch['p_value']:.4f}  R²={perm_aitch['R2']:.3f}")
    log(f"    PERMDISP   F={pdisp_aitch['F']:.3f}  p={pdisp_aitch['p_value']:.4f}")

    # 5b. MMD pairwise matrix + PERMANOVA + PERMDISP
    log("  [5b] MMD pairwise matrix …")
    D_mmd_sq, sigma_mmd = mmd_pairwise_matrix(
        mice_features, max_voxels=mmd_max_voxels, log=log)
    D_mmd      = np.sqrt(np.maximum(D_mmd_sq, 0.0))
    perm_mmd   = permanova(D_mmd, g_arr, n_perm=n_perm)
    pdisp_mmd  = permdisp(D_mmd, g_arr, n_perm=n_perm)
    log(f"    PERMANOVA  F={perm_mmd['F']:.3f}  p={perm_mmd['p_value']:.4f}  R²={perm_mmd['R2']:.3f}")
    log(f"    PERMDISP   F={pdisp_mmd['F']:.3f}  p={pdisp_mmd['p_value']:.4f}")

    # 5c. Dirichlet regression on π
    log("  [5c] Dirichlet regression …")
    if covariates is not None:
        cnames_str = ", ".join(covar_names) if covar_names else "covariables"
        log(f"  Dirichlet avec covariables : {cnames_str}")
    dirich_res = dirichlet_regression(pi_matrix, g_arr,
                                      covariates=covariates, log=log)

    # 5c2. Per-habitat Mann-Whitney on π (mouse-level, Bonferroni corrected)
    log("  [5c2] Per-habitat Mann-Whitney on π …")
    unique_g = np.unique(g_arr)
    pi_mw_results = pi_per_habitat_tests(pi_matrix, g_arr, log=log)

    # 5d. Conditional MMD per habitat
    log("  [5d] Conditional MMD per habitat …")
    if len(unique_g) == 2:
        grp_a = [i for i, g in enumerate(g_arr) if g == unique_g[0]]
        grp_b = [i for i, g in enumerate(g_arr) if g == unique_g[1]]
    else:
        grp_a = [i for i, g in enumerate(g_arr) if g == unique_g[0]]
        grp_b = [i for i, g in enumerate(g_arr) if g != unique_g[0]]

    cond_mmd = mmd_per_habitat(
        mice_features, mice_posteriors, grp_a, grp_b,
        k=best_k, sigma=sigma_mmd,
        n_perm=n_perm_mmd_cond, max_voxels=mmd_max_voxels, log=log)

    # 5e. Witness function (top 3 discriminating pairs)
    log("  [5e] Witness function …")
    if grp_a and grp_b:
        Xa = np.vstack([mice_features[i] for i in grp_a])
        Xb = np.vstack([mice_features[i] for i in grp_b])
        top_pairs  = top_witness_pairs(Xa, Xb, sigma_mmd, common_params, top_n=3)
        witness_figs = [
            witness_2d(Xa, Xb, sigma_mmd, pi, pj, common_params)
            for pi, pj, _ in top_pairs
        ]
    else:
        witness_figs = []

    log("═══ Step 6 / 6 — Figures & PDF ═══")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig_paths = []

    def _save(name, fig, caption):
        p = _save_fig(fig, os.path.join(fig_dir, name))
        fig_paths.append((p, caption))

    _save("f1_atlas_profiles.png",
          _fig_atlas_profiles(atlas, g_arr, mice_descs, mouse_names, group_names),
          "F1 — Atlas centroids & per-mouse composition π")

    _save("f2_heterogeneity.png",
          _fig_intra_variance(atlas, mice_descs, mouse_names, g_arr, group_names),
          "F2 — Intra-habitat heterogeneity σ² and atlas residuals")

    _save("f3_mmd_matrix.png",
          _fig_mmd_matrix(D_mmd_sq, mouse_names, g_arr, group_names, perm_mmd),
          "F3 — MMD pairwise distance matrix + PERMANOVA")

    _save("f4_dirichlet.png",
          _fig_dirichlet(dirich_res, atlas, group_names),
          "F4 — Dirichlet regression: per-habitat group effects")

    _save("f4b_pi_mw.png",
          _fig_pi_habitat_tests(pi_mw_results, pi_matrix, g_arr, group_names),
          "F4b — Per-habitat Mann-Whitney on π (mouse-level, Bonferroni corrected)")

    _save("f5_conditional_mmd.png",
          _fig_conditional_mmd(cond_mmd, group_names,
                               [int(unique_g[0])], [int(unique_g[-1])]),
          "F5 — Conditional MMD per habitat")

    _save("f6_stat_summary.png",
          _fig_permanova_summary(perm_aitch, perm_mmd, pdisp_aitch, pdisp_mmd, dirich_res),
          "F6 — Statistical test summary")

    if witness_figs:
        ga_n = group_names.get(int(unique_g[0]), f"G{unique_g[0]}")
        gb_n = group_names.get(int(unique_g[-1]), f"G{unique_g[-1]}")
        _save("f7_witness.png",
              _fig_witness(witness_figs, ga_n, gb_n),
              f"F7 — MMD witness function  ({ga_n} vs {gb_n})")

    log("  Generating habitat spatial maps…")
    for i, (fig_map, caption_map) in enumerate(
            _fig_habitat_maps(df_pooled, habitat_labels,
                              mouse_names, g_arr, group_names, best_k)):
        safe_name = mouse_names[i].replace(" ", "_").replace("/", "-")
        _save(f"fmap_{i:02d}_{safe_name}.png", fig_map, caption_map)

    # Cover text
    k_line = (f"K={best_k}  (BIC selection)" if not k_override
              else f"K={best_k}  (forced)")
    cover = (
        f"Date           : {date_str}\n"
        f"Mice           : {n_mice}   Groups: {len(unique_g)}\n"
        f"Normalization  : {normalization}\n"
        f"{k_line}\n"
        f"Parameters     : {', '.join(common_params)}\n"
        f"Atlas stable   : {atlas.stability_.get('stable', 'not checked')}\n\n"
        f"PERMANOVA (Aitchison π)  F={perm_aitch['F']:.3f}  "
        f"p={perm_aitch['p_value']:.4f}  {_sig_stars(perm_aitch['p_value'])}\n"
        f"PERMANOVA (MMD dist.)    F={perm_mmd['F']:.3f}  "
        f"p={perm_mmd['p_value']:.4f}  {_sig_stars(perm_mmd['p_value'])}\n"
        f"Dirichlet regression     LR={dirich_res['lr_stat']:.2f}  "
        f"p={dirich_res['p_value']:.4f}  {_sig_stars(dirich_res['p_value'])}\n"
        f"\nGroups:\n"
        + "".join(
            f"  {group_names.get(int(g), f'G{g}')}: "
            f"{(g_arr == g).sum()} mice — "
            f"{', '.join(mouse_names[i] for i, gl in enumerate(g_arr) if gl == g)}\n"
            for g in unique_g
        )
    )

    pdf_path = os.path.join(out_dir, "atlas_comparison_report.pdf")
    _build_pdf(pdf_path, fig_paths, cover)
    log(f"PDF saved → {pdf_path}", "ok")

    return {
        "atlas"           : atlas,
        "pi_matrix"       : pi_matrix,
        "sigma2_matrix"   : sigma2_matrix,
        "residual_matrix" : residual_matrix,
        "mmd_sq_matrix"   : D_mmd_sq,
        "mmd_sigma"       : sigma_mmd,
        "permanova_aitch" : perm_aitch,
        "permanova_mmd"   : perm_mmd,
        "permdisp_aitch"  : pdisp_aitch,
        "permdisp_mmd"    : pdisp_mmd,
        "dirichlet"       : dirich_res,
        "mmd_per_hab"     : cond_mmd,
        "bic_dict"        : bic_dict,
        "common_params"   : common_params,
        "mouse_names"     : mouse_names,
        "group_labels"    : g_arr,
        "pdf_path"        : pdf_path,
        "out_dir"         : out_dir,
    }
