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


def _build_interpretation_pages(
    pi_matrix, sigma2_matrix, residual_matrix,
    perm_aitch, perm_mmd, pdisp_aitch, pdisp_mmd,
    dirich_res, pi_mw_results, cond_mmd,
    group_names, g_arr, common_params,
    best_k, atlas, n_mice, out_dir
) -> list[tuple[str, str]]:
    """
    Generate one or two A4-landscape pages of plain-language interpretation
    of all Atlas statistical results.  Saved as PNG, returned as (path, caption)
    tuples ready to be inserted in fig_paths.
    """
    import math as _math
    import textwrap as _tw

    unique_g   = sorted(set(g_arr.tolist()))
    n_groups   = len(unique_g)
    stability  = getattr(atlas, "stability_", {}) or {}
    stable     = stability.get("stable", True)
    mean_shifts = np.asarray(stability.get("mean_shift", np.zeros(best_k)), dtype=float)
    max_shift   = float(np.max(mean_shifts)) if len(mean_shifts) else 0.0

    p_aitch  = float(perm_aitch.get("p_value", 1.0))
    p_mmd    = float(perm_mmd.get("p_value",   1.0))
    p_dir    = float(dirich_res.get("p_value",  1.0))
    r2_aitch = float(perm_aitch.get("R2", 0.0))
    r2_mmd   = float(perm_mmd.get("R2",   0.0))
    p_pdisp_a = float(pdisp_aitch.get("p_value", 1.0))
    p_pdisp_m = float(pdisp_mmd.get("p_value",   1.0))

    def _sig(p):
        if p < 0.001: return f"p < 0.001  ***"
        if p < 0.01:  return f"p = {p:.3f}  **"
        if p < 0.05:  return f"p = {p:.3f}  *"
        if p < 0.10:  return f"p = {p:.3f}  (tendance)"
        return        f"p = {p:.3f}  n.s."

    def _col(p):
        if p < 0.05:  return "#1B5E20"   # dark green
        if p < 0.10:  return "#E65100"   # orange
        return "#616161"                  # grey

    def _gname(g):
        return group_names.get(int(g), f"G{g}")

    # -----------------------------------------------------------------------
    # Build a list of "line" records:
    #   ("text", fontsize, bold, color, left_margin_fraction)
    # -----------------------------------------------------------------------
    Record = lambda txt, fs=9, bold=False, col="#222222", lm=0.04: \
        (txt, fs, bold, col, lm)
    SEP  = lambda: Record("", fs=3)
    HEAD = lambda t: Record(t, fs=11.5, bold=True, col="#1A237E", lm=0.03)
    RULE = lambda: Record("─" * 105, fs=6.5, col="#BDBDBD", lm=0.03)

    page1, page2 = [], []

    def add1(txt, fs=9, bold=False, col="#222222", lm=0.04):
        page1.append(Record(txt, fs, bold, col, lm))

    def add2(txt, fs=9, bold=False, col="#222222", lm=0.04):
        page2.append(Record(txt, fs, bold, col, lm))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 1  — Stabilité · Tests globaux · Composition par habitat
    # ═══════════════════════════════════════════════════════════════════════

    add1("RAPPORT D'INTERPRÉTATION — ATLAS MULTI-ANIMAL", fs=14,
         bold=True, col="#1A237E", lm=0.03)
    add1(f"Cohorte : {n_mice} souris · {n_groups} groupes · {best_k} habitats universels  "
         f"({', '.join(_gname(g) for g in unique_g)})", fs=9, col="#555555")
    page1.append(RULE())
    page1.append(SEP())

    # ── ① Stabilité de l'atlas ─────────────────────────────────────────────
    page1.append(HEAD("①  QUALITÉ ET STABILITÉ DE L'ATLAS"))
    page1.append(RULE())

    if stable:
        add1(f"L'atlas est STABLE : les {best_k} habitats restent cohérents d'un bootstrap à l'autre "
             f"(déplacement moyen maximal : {max_shift:.3f} unités IQR < seuil 0.10).",
             col="#1B5E20")
        add1("  → Les résultats per-habitat peuvent être interprétés avec confiance.",
             col="#1B5E20", lm=0.06)
    else:
        worst_h = int(np.argmax(mean_shifts))
        add1(f"⚠  INSTABILITÉ DÉTECTÉE : l'Habitat H{worst_h} présente un déplacement bootstrap "
             f"élevé ({max_shift:.3f} > 0.10 unités IQR).",
             bold=True, col="#B71C1C")
        add1(f"  → Le nombre K = {best_k} habitats est peut-être trop grand pour cette cohorte. "
             f"Essayer K = {best_k - 1} via 'K forcé' dans l'interface.",
             col="#E65100", lm=0.06)
        add1("  → Les conclusions par habitat doivent être interprétées avec prudence.",
             col="#E65100", lm=0.06)

    for h in range(best_k):
        ms  = float(mean_shifts[h]) if h < len(mean_shifts) else 0.0
        qlt = ("excellent (< 0.05)" if ms < 0.05 else
               "bon (< 0.10)"       if ms < 0.10 else
               "moyen (< 0.20)"     if ms < 0.20 else
               "faible (≥ 0.20)  ⚠")
        col = "#1B5E20" if ms < 0.10 else ("#E65100" if ms < 0.20 else "#B71C1C")
        add1(f"     H{h} : déplacement moyen bootstrap = {ms:.4f}  [{qlt}]",
             fs=8.5, col=col, lm=0.06)

    page1.append(SEP())

    # ── ② Tests globaux ────────────────────────────────────────────────────
    page1.append(HEAD("②  TESTS STATISTIQUES GLOBAUX"))
    page1.append(RULE())

    # PERMANOVA Aitchison
    add1(f"PERMANOVA (compositions Aitchison) :  F = {perm_aitch.get('F', 0):.2f}  |  "
         f"R² = {r2_aitch:.3f} ({r2_aitch:.0%})  |  {_sig(p_aitch)}",
         bold=True, col=_col(p_aitch))
    if p_aitch < 0.05:
        add1(f"  → Les PROPORTIONS d'habitats diffèrent significativement entre groupes. "
             f"Le groupe explique {r2_aitch:.0%} de la variance compositionnelle totale.",
             col="#1B5E20", lm=0.06)
    elif p_aitch < 0.10:
        add1("  → Tendance à une différence de composition (borderline). "
             "Augmenter la cohorte pourrait révéler un effet réel.", col="#E65100", lm=0.06)
    else:
        add1("  → Aucune différence significative de composition : les tumeurs ont des proportions "
             "d'habitats similaires entre groupes.", col="#616161", lm=0.06)

    page1.append(SEP())

    # PERMANOVA MMD
    add1(f"PERMANOVA (distributions complètes — MMD) :  F = {perm_mmd.get('F', 0):.2f}  |  "
         f"R² = {r2_mmd:.3f} ({r2_mmd:.0%})  |  {_sig(p_mmd)}",
         bold=True, col=_col(p_mmd))
    if p_mmd < 0.05:
        add1(f"  → Les DISTRIBUTIONS DE VOXELS diffèrent entre groupes. Ce test capture également "
             f"les différences intra-habitat (structure, texture MRI), pas seulement les proportions.",
             col="#1B5E20", lm=0.06)
    elif p_mmd < 0.10:
        add1("  → Signal MMD borderline. La distribution complète diffère légèrement mais le résultat "
             "n'est pas encore fiable.", col="#E65100", lm=0.06)
    else:
        add1("  → Distributions voxel-level non différentes entre groupes.", col="#616161", lm=0.06)

    # Concordance entre les deux tests
    if p_aitch < 0.05 and p_mmd < 0.05:
        add1("  ✔  Les deux approches concordent : la différence est robuste "
             "(composition ET distribution de voxels).", bold=True, col="#1B5E20", lm=0.06)
    elif p_aitch >= 0.05 and p_mmd < 0.05:
        add1("  ⚡  Dissociation : la composition globale n'est pas significative, mais la distribution "
             "l'est. Les groupes partagent les mêmes proportions d'habitats mais ont des biologies "
             "voxel-level distinctes. Chercher un effet sur la BIOLOGIE INTERNE des habitats (voir §⑥).",
             col="#E65100", lm=0.06)
    elif p_aitch < 0.05 and p_mmd >= 0.05:
        add1("  ⚡  La composition diffère mais pas la distribution complète : les groupes ont des "
             "proportions d'habitats différentes mais une biologie voxel-level similaire au sein de "
             "chaque habitat. Effet purement quantitatif (volume) et non qualitatif (biologie).",
             col="#E65100", lm=0.06)
    page1.append(SEP())

    # PERMDISP
    if p_pdisp_a < 0.05 or p_pdisp_m < 0.05:
        parts = []
        if p_pdisp_a < 0.05: parts.append(f"π Aitchison : {_sig(p_pdisp_a)}")
        if p_pdisp_m < 0.05: parts.append(f"MMD : {_sig(p_pdisp_m)}")
        add1("PERMDISP (dispersion inégale) :  " + "  |  ".join(parts),
             bold=True, col="#E65100")
        add1("  → Un groupe est PLUS HÉTÉROGÈNE que l'autre (variabilité inter-souris plus grande). "
             "Le PERMANOVA peut détecter cette dispersion plutôt qu'un vrai shift de centre. "
             "Interpréter avec précaution : l'effet peut être dû à l'hétérogénéité d'un groupe.",
             col="#E65100", lm=0.06)
        add1("  → En oncologie, une dispersion élevée = hétérogénéité tumorale inter-individuelle "
             "plus grande dans ce groupe (par ex. : tumeurs à des stades évolutifs différents).",
             col="#E65100", lm=0.06)
    else:
        add1("PERMDISP : pas de différence de dispersion inter-groupe "
             f"(π: {_sig(p_pdisp_a)}  |  MMD: {_sig(p_pdisp_m)}).", col="#616161")
        add1("  → Le PERMANOVA reflète bien un vrai shift de position, pas un effet de dispersion.",
             col="#1B5E20", lm=0.06)
    page1.append(SEP())

    # Dirichlet
    lr = float(dirich_res.get("lr_stat", 0))
    add1(f"Dirichlet regression (modèle multivarié) :  LR = {lr:.2f}  |  {_sig(p_dir)}",
         bold=True, col=_col(p_dir))
    if p_dir < 0.05:
        add1("  → La composition d'habitats est globalement modifiée par le groupe dans un modèle "
             "Dirichlet (adapté aux données compositionnelles). Confirme le PERMANOVA Aitchison.",
             col="#1B5E20", lm=0.06)
        coef = dirich_res.get("group_coef")
        if coef is not None and len(coef):
            try:
                coef_arr = np.asarray(coef)
                for h in range(min(best_k, coef_arr.shape[1] if coef_arr.ndim > 1 else len(coef_arr))):
                    c_val = float(coef_arr[0, h] if coef_arr.ndim > 1 else coef_arr[h])
                    fold  = float(np.exp(c_val))
                    dirn  = "plus" if c_val > 0 else "moins"
                    add1(f"     H{h} : coefficient = {c_val:+.3f}  →  {_gname(unique_g[-1])} a "
                         f"{fold:.2f}× {dirn} d'H{h} que {_gname(unique_g[0])}.",
                         fs=8.5, col="#333333", lm=0.06)
            except Exception:
                pass
    else:
        add1("  → Pas de modification globale de la composition selon la Dirichlet regression.",
             col="#616161", lm=0.06)

    page1.append(SEP())

    # ── ③ Composition par habitat ──────────────────────────────────────────
    page1.append(HEAD("③  ANALYSE DE COMPOSITION PAR HABITAT (π)"))
    page1.append(RULE())
    add1("π_h = proportion moyenne du volume tumoral assignée à l'Habitat h (soft-assignment GMM). "
         "Somme des π sur tous les habitats = 1 par souris.", fs=8, col="#555555")
    page1.append(SEP())

    for h in range(best_k):
        pi_means  = {g: float(np.mean(pi_matrix[g_arr == g, h])) for g in unique_g}
        pi_sds    = {g: float(np.std( pi_matrix[g_arr == g, h], ddof=1)) for g in unique_g}
        mw        = pi_mw_results[h] if h < len(pi_mw_results) else {}
        p_adj     = float(mw.get("p_adj", 1.0))
        reject    = bool(mw.get("reject", False))

        max_g = max(pi_means, key=pi_means.get)
        min_g = min(pi_means, key=pi_means.get)
        diff  = pi_means[max_g] - pi_means[min_g]
        ratio = pi_means[max_g] / max(pi_means[min_g], 1e-6)

        stat_str = f"Mann-Whitney Bonferroni : {_sig(p_adj)}"
        comp_str = "  |  ".join(
            f"{_gname(g)} : {pi_means[g]:.1%} ± {pi_sds[g]:.1%}" for g in unique_g)
        add1(f"H{h}  [{comp_str}]   [{stat_str}]",
             fs=8.5, bold=True, col=_col(p_adj))

        if reject:
            add1(f"  → H{h} est significativement plus présent dans {_gname(max_g)} "
                 f"(+{diff:.1%}, ratio {ratio:.1f}×). "
                 f"Ce groupe présente davantage de tissu à la signature de cet habitat.",
                 col="#1B5E20", lm=0.06)
        else:
            if diff > 0.10:
                add1(f"  → Tendance non significative : {_gname(max_g)} > {_gname(min_g)} "
                     f"(Δ = {diff:.1%}) mais non confirmée avec n = {n_mice}. "
                     "Augmenter la cohorte.", col="#E65100", lm=0.06)
            else:
                add1(f"  → Proportion équivalente entre groupes (Δ = {diff:.1%} — effet négligeable).",
                     col="#616161", lm=0.06)

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 2  — Hétérogénéité · Résidus · MMD conditionnel · Conclusions
    # ═══════════════════════════════════════════════════════════════════════

    add2("RAPPORT D'INTERPRÉTATION — ATLAS MULTI-ANIMAL  (suite)", fs=13,
         bold=True, col="#1A237E", lm=0.03)
    page2.append(RULE())
    page2.append(SEP())

    # ── ④ Hétérogénéité intra-habitat ─────────────────────────────────────
    page2.append(HEAD("④  HÉTÉROGÉNÉITÉ INTRA-HABITAT  (σ²)"))
    page2.append(RULE())
    add2("σ²_h mesure la dispersion des voxels autour du centroïde de leur habitat dans chaque souris. "
         "Grande valeur = habitat biologiquement hétéroclite (voxels très dispersés). "
         "Faible valeur = habitat compact, biologie uniforme.", fs=8, col="#555555")
    add2("Référence : σ² < 1.0 = compact · 1.0–2.0 = modéré · > 2.0 = très hétérogène.", fs=8, col="#555555")
    page2.append(SEP())

    for h in range(best_k):
        s2_means = {g: float(np.mean(sigma2_matrix[g_arr == g, h])) for g in unique_g}
        s2_sds   = {g: float(np.std( sigma2_matrix[g_arr == g, h], ddof=1)) for g in unique_g}
        max_g = max(s2_means, key=s2_means.get)
        min_g = min(s2_means, key=s2_means.get)
        ratio = s2_means[max_g] / max(s2_means[min_g], 1e-6)
        max_val = s2_means[max_g]

        comp_str = "  |  ".join(
            f"{_gname(g)} : {s2_means[g]:.3f} ± {s2_sds[g]:.3f}" for g in unique_g)
        add2(f"H{h}  [{comp_str}]", fs=8.5, bold=True)

        if max_val > 2.0 and ratio > 1.4:
            add2(f"  → σ² ÉLEVÉ dans {_gname(max_g)} (> 2.0, ratio {ratio:.1f}×) : "
                 f"l'Habitat H{h} capture des biologies très variées dans ce groupe. "
                 "Possible transition entre états tissulaires ou biologie mal définie pour ces souris. "
                 "Envisager une sous-analyse (Discovery Level 2) pour affiner.",
                 col="#B71C1C", lm=0.06)
        elif ratio > 1.5:
            add2(f"  → H{h} est {ratio:.1f}× plus hétérogène dans {_gname(max_g)} "
                 f"(σ² = {max_val:.3f}). "
                 "La biologie de cet habitat est moins homogène dans ce groupe "
                 "(ex : stades tumoraux multiples, hétérogénéité intra-tumorale).",
                 col="#E65100", lm=0.06)
        elif ratio > 1.2:
            add2(f"  → Légère différence d'hétérogénéité dans H{h} : "
                 f"{_gname(max_g)} est {ratio:.1f}× plus dispersé.", col="#E65100", lm=0.06)
        else:
            add2(f"  → σ² comparable entre groupes — H{h} est uniformément homogène.",
                 col="#616161", lm=0.06)

    page2.append(SEP())

    # ── ⑤ Résidus atlas ────────────────────────────────────────────────────
    page2.append(HEAD("⑤  FIDÉLITÉ À L'ATLAS  (résidu euclidien)"))
    page2.append(RULE())
    add2("Le résidu mesure la distance euclidienne moyenne entre les voxels d'une souris "
         "et le centroïde de l'atlas pour cet habitat. "
         "Grand résidu = la souris 'dépasse' l'atlas — biologie atypique ou sous-type rare.",
         fs=8, col="#555555")
    add2("Référence : résidu < 0.8 = bien aligné · 0.8–1.5 = modéré · > 1.5 = biologie atypique.",
         fs=8, col="#555555")
    page2.append(SEP())

    for h in range(best_k):
        r_means = {g: float(np.mean(residual_matrix[g_arr == g, h])) for g in unique_g}
        r_sds   = {g: float(np.std( residual_matrix[g_arr == g, h], ddof=1)) for g in unique_g}
        max_g   = max(r_means, key=r_means.get)
        min_g   = min(r_means, key=r_means.get)
        ratio   = r_means[max_g] / max(r_means[min_g], 1e-6)
        max_val = r_means[max_g]

        comp_str = "  |  ".join(
            f"{_gname(g)} : {r_means[g]:.3f} ± {r_sds[g]:.3f}" for g in unique_g)
        add2(f"H{h}  [{comp_str}]", fs=8.5, bold=True)

        if max_val > 1.5:
            add2(f"  → Résidu ÉLEVÉ dans {_gname(max_g)} pour H{h} (> 1.5) : "
                 "ces souris s'écartent significativement du centroïde atlas. "
                 "Leur biologie dans cet habitat est atypique — sous-type non représenté "
                 "dans la moyenne de cohorte, ou artefact technique (qualité du scan).",
                 col="#B71C1C", lm=0.06)
        elif ratio > 1.4:
            add2(f"  → {_gname(max_g)} s'écarte davantage de l'atlas dans H{h} "
                 f"(résidu {ratio:.1f}× plus grand). "
                 "La biologie de cet habitat est légèrement plus éloignée du prototype moyen.",
                 col="#E65100", lm=0.06)
        else:
            add2(f"  → Les deux groupes sont bien alignés avec l'atlas dans H{h}.",
                 col="#616161", lm=0.06)

    page2.append(SEP())

    # ── ⑥ MMD conditionnel ─────────────────────────────────────────────────
    page2.append(HEAD("⑥  BIOLOGIE INTERNE DES HABITATS — MMD CONDITIONNEL"))
    page2.append(RULE())
    add2("Le MMD conditionnel teste si, pour les voxels ASSIGNÉS à un même habitat (argmax), "
         "la distribution multivariée MRI est identique entre groupes. "
         "C'est la question : 'H{h} a-t-il la même signature MRI dans les deux groupes ?'",
         fs=8, col="#555555")
    add2("Différence significative = même label, biologie différente = sous-type latent non capturé par l'atlas.",
         fs=8, col="#555555")
    page2.append(SEP())

    n_cond_sig = 0
    for h in range(best_k):
        cm = cond_mmd[h] if h < len(cond_mmd) else {}
        mmd_sq = cm.get("mmd_sq", float("nan"))
        p_h    = cm.get("p_value", float("nan"))
        n_a    = int(cm.get("n_a", 0))
        n_b    = int(cm.get("n_b", 0))
        sigma  = float(cm.get("sigma", float("nan")))

        if _math.isnan(mmd_sq):
            add2(f"H{h}  — MMD² non calculable (voxels insuffisants : "
                 f"n = {n_a} + {n_b} < 10 par groupe).", fs=8.5, col="#AAAAAA")
            add2("  → Augmenter le nombre de souris ou vérifier que cet habitat est bien représenté "
                 "dans les deux groupes.", col="#AAAAAA", lm=0.06)
            continue

        if not _math.isnan(p_h) and p_h < 0.05:
            n_cond_sig += 1

        p_str = _sig(p_h) if not _math.isnan(p_h) else "non disponible"
        sigma_str = f"σ = {sigma:.4f}" if not _math.isnan(sigma) else ""
        add2(f"H{h}  — MMD² = {mmd_sq:.5f}  |  {p_str}  "
             f"(n_A = {n_a}, n_B = {n_b}{', ' + sigma_str if sigma_str else ''})",
             fs=8.5, bold=True, col=_col(p_h) if not _math.isnan(p_h) else "#616161")

        if not _math.isnan(p_h) and p_h < 0.05:
            add2(f"  → BIOLOGIE INTERNE DIFFÉRENTE dans H{h} entre groupes : "
                 "même en ayant le même 'label', les voxels H{h} ont une signature MRI "
                 "distincte selon le groupe. L'atlas ne sépare pas ce sous-type.",
                 col="#1B5E20", lm=0.06)
            add2(f"  → Recommandation : lancer une analyse Discovery (Level 2) ciblée sur H{h} "
                 "pour identifier la structure latente. Ou enrichir l'atlas avec plus de souris "
                 f"pour permettre un K plus élevé.", col="#1565C0", lm=0.06)
        elif not _math.isnan(p_h):
            add2(f"  → Biologie interne équivalente dans H{h} : l'effet entre groupes dans cet "
                 "habitat est purement quantitatif (proportion différente, pas la biologie).",
                 col="#616161", lm=0.06)

    page2.append(SEP())

    # ── ⑦ Conclusion ───────────────────────────────────────────────────────
    page2.append(HEAD("⑦  CONCLUSION GLOBALE ET RECOMMANDATIONS"))
    page2.append(RULE())

    n_sig = sum(1 for p in [p_aitch, p_mmd, p_dir] if p < 0.05)
    n_hab_sig = sum(1 for h in range(best_k)
                    if h < len(pi_mw_results) and pi_mw_results[h].get("reject", False))

    if n_sig == 3:
        add2("CONCLUSION FORTE : Les trois tests globaux sont significatifs. "
             "La différence entre groupes est robuste et confirmée à deux niveaux "
             "(composition des habitats ET distribution voxel-level).",
             bold=True, col="#1B5E20")
    elif n_sig == 2:
        add2("CONCLUSION MODÉRÉE : Deux tests globaux sur trois sont significatifs. "
             "Il existe une différence réelle entre groupes, confirmée par deux approches indépendantes.",
             bold=True, col="#2E7D32")
    elif n_sig == 1:
        add2("SIGNAL FAIBLE : Un seul test global significatif. "
             "La différence existe mais est visible à un seul niveau d'analyse. "
             "Augmenter n ou vérifier la qualité des données.",
             bold=True, col="#E65100")
    else:
        add2("PAS DE DIFFÉRENCE GLOBALE DÉTECTÉE. "
             "Les groupes ont des habitats tumoraux similaires (composition ET distribution). "
             "Causes possibles : effet trop subtil, cohorte insuffisante, ou vraie similarité biologique.",
             bold=True, col="#616161")

    page2.append(SEP())

    if n_hab_sig:
        add2(f"• {n_hab_sig} habitat(s) sur {best_k} montrent une différence de proportion significative "
             f"(Mann-Whitney Bonferroni). Ce sont les candidats biologiques prioritaires à valider "
             f"par immunohistochimie ou marqueurs moléculaires.")
    if n_cond_sig:
        add2(f"• {n_cond_sig} habitat(s) ont une biologie interne différente (MMD conditionnel). "
             "Priorité absolue pour sous-analyse Discovery (Level 2) ou validation histologique.")
    if not stable:
        add2(f"• ⚠  Atlas instable — les analyses par habitat sont indicatives, pas définitives. "
             f"Essayer K = {best_k - 1} ou augmenter la cohorte.", col="#B71C1C")
    if n_mice < 6:
        add2(f"• PUISSANCE LIMITÉE : n = {n_mice} souris. "
             "Les tests de permutation sont peu fiables sous 6 animaux par groupe. "
             "Valider sur une cohorte plus large avant conclusion.", col="#E65100")
    if p_pdisp_a < 0.05 or p_pdisp_m < 0.05:
        add2("• Dispersion inégale détectée (PERMDISP) : interpréter les tests PERMANOVA "
             "avec prudence — l'effet peut refléter une hétérogénéité inter-souris plus grande "
             "dans un groupe (ex : stades tumoraux mixtes) plutôt qu'un vrai shift de biologie moyenne.",
             col="#E65100")

    page2.append(SEP())

    # ── ⑧ Notes méthodologiques ────────────────────────────────────────────
    page2.append(HEAD("⑧  NOTES MÉTHODOLOGIQUES ET LIMITES"))
    page2.append(RULE())

    notes = [
        "PERMANOVA est sensible à la dispersion inégale entre groupes. "
        "Toujours vérifier PERMDISP avant d'interpréter un PERMANOVA significatif.",

        "La distance d'Aitchison (CLR + Euclidien) est adaptée aux compositions (somme = 1) "
        "et invariante aux redimensionnements. Elle est plus robuste que la distance euclidienne brute sur π.",

        "Le MMD (Maximum Mean Discrepancy) utilise un noyau RBF dont la bande passante σ est estimée "
        "par la médiane des distances voxel-level. Sur des cohortes très hétérogènes, σ peut être biaisé "
        "vers des valeurs trop grandes, réduisant la sensibilité.",

        "La Dirichlet regression suppose que π suit une loi Dirichlet. Si un habitat est absent "
        "dans une souris (π ≈ 0), un pseudo-compte ε est ajouté — les coefficients log-ratio "
        "peuvent alors être légèrement biaisés vers zéro.",

        f"Le nombre K = {best_k} habitats a été sélectionné par vote majoritaire (BIC / ICL / Silhouette). "
        "Ce K est optimal pour cette cohorte et cette normalisation — il peut différer si la cohorte ou "
        "la normalisation changent.",

        "L'atlas est construit sur un sous-échantillon équilibré de 500 voxels par souris (tirage avec "
        "remplacement), garantissant un poids égal indépendamment de la taille tumorale. "
        "Les tumeurs très petites (< 500 voxels) contribuent avec plus de répétitions.",

        "Le soft-assignment (π) est calculé par la probabilité a posteriori GMM, plus stable que "
        "l'assignement dur (argmax). Le MMD conditionnel utilise l'assignement dur — ses résultats "
        "dépendent de la netteté des frontières entre habitats.",

        "Les corrections multiples (Bonferroni × K) sur les tests Mann-Whitney per-habitat sont "
        "conservatrices — un p_adj borderline (0.05–0.15) mérite attention même s'il est non significatif.",
    ]
    for i, note in enumerate(notes, 1):
        wrapped = _tw.fill(f"{'•' if i > 0 else ' '}  {note}", width=120)
        for j, line in enumerate(wrapped.split("\n")):
            add2(line, fs=7.8, col="#444444", lm=0.04 if j == 0 else 0.07)
        page2.append(SEP())

    # -----------------------------------------------------------------------
    # Render pages to PNG
    # -----------------------------------------------------------------------
    def _render_page(records, out_path):
        fig = plt.figure(figsize=(13.0, 9.5))
        fig.patch.set_facecolor("white")
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="white", zorder=0))

        y    = 0.975
        dy_base = 0.013          # baseline line height for fs=9
        for (txt, fs, bold, col, lm) in records:
            if not txt:
                y -= dy_base * 0.35
                continue
            dy  = dy_base * (fs / 9.0)
            fw  = "bold" if bold else "normal"
            ax.text(lm, y, txt, fontsize=fs, fontweight=fw, color=col,
                    va="top", ha="left", transform=ax.transAxes,
                    fontfamily="monospace",
                    clip_on=True)
            y  -= dy
            if y < 0.005:
                break

        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)

    p1_path = os.path.join(out_dir, "interp_page1.png")
    p2_path = os.path.join(out_dir, "interp_page2.png")
    _render_page(page1, p1_path)
    _render_page(page2, p2_path)

    return [
        (p1_path, "Rapport d'interprétation — Page 1 : stabilité · tests globaux · composition"),
        (p2_path, "Rapport d'interprétation — Page 2 : hétérogénéité · résidus · MMD conditionnel · conclusions"),
    ]


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
    dtopo_weight   : float           = 0.0,
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

    if dtopo_weight > 0:
        log(f"  [d_topo] weight={dtopo_weight} — note: d_topo two-pass is not yet "
            f"implemented in the Atlas pipeline. Parameter accepted but ignored.")
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

    # Interpretation pages
    log("  Generating interpretation report…")
    try:
        interp_pages = _build_interpretation_pages(
            pi_matrix       = pi_matrix,
            sigma2_matrix   = sigma2_matrix,
            residual_matrix = residual_matrix,
            perm_aitch      = perm_aitch,
            perm_mmd        = perm_mmd,
            pdisp_aitch     = pdisp_aitch,
            pdisp_mmd       = pdisp_mmd,
            dirich_res      = dirich_res,
            pi_mw_results   = pi_mw_results,
            cond_mmd        = cond_mmd,
            group_names     = group_names,
            g_arr           = g_arr,
            common_params   = common_params,
            best_k          = best_k,
            atlas           = atlas,
            n_mice          = n_mice,
            out_dir         = out_dir,
        )
        fig_paths.extend(interp_pages)
        log("  Interpretation pages added to PDF.", "ok")
    except Exception as _ie:
        log(f"  [!] Interpretation page skipped: {_ie}")

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
