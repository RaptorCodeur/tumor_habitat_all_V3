"""
visualize_joint_boundaries.py
Visualize 7D joint-clustering results per mouse:
  1. Global PCA — all voxels in the pooled 7D space projected to 2D
  2. Per-mouse PCA grid — each mouse in the same global PCA space
  3. Parallel coordinates — per-mouse habitat profiles vs. global centroids
  4. Centroid deviation heatmap — how far each mouse's habitat mean is
     from the global centroid (per feature and in total Euclidean distance)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA

# ── colours ────────────────────────────────────────────────────────────────
HABITAT_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                  '#1ABC9C', '#E67E22', '#2980B9']


def load_results(result_dir):
    centroids_path = os.path.join(result_dir, 'centroids.json')
    with open(centroids_path) as f:
        raw = json.load(f)

    mouse_dirs = sorted([
        d for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d))
    ])

    dfs = []
    for m in mouse_dirs:
        csv = os.path.join(result_dir, m, 'habitats_result.csv')
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df['mouse_name'] = m
            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Parameter list: all numeric columns except metadata
    skip = {'X', 'Y', 'Slice', 'Habitat', 'confidence', 'Habitat_label', 'mouse_name'}
    params = [c for c in df_all.columns if c not in skip]

    k = len(raw)
    centroid_mat = np.array([[raw[str(c)].get(p, 0.0) for p in params]
                              for c in range(k)])

    return df_all, params, centroid_mat, k


# ── 1. Global PCA ───────────────────────────────────────────────────────────

def plot_global_pca(ax, features_2d, labels, centroid_2d, k, pca):
    colors = HABITAT_COLORS[:k]
    for c in range(k):
        mask = labels == c
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=colors[c], s=4, alpha=0.35, linewidths=0, label=f'H{c}')

    # centroids
    for c in range(k):
        ax.scatter(*centroid_2d[c], c=colors[c], s=180, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(f'H{c}', centroid_2d[c],
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=9, fontweight='bold', color=colors[c])

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({100*var[0]:.1f}% var)', fontsize=9)
    ax.set_ylabel(f'PC2 ({100*var[1]:.1f}% var)', fontsize=9)
    ax.set_title('Global — all mice pooled', fontsize=10, fontweight='bold')
    ax.legend(markerscale=2, fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.3)


# ── 2. Per-mouse PCA grid ───────────────────────────────────────────────────

def plot_per_mouse_pca(pdf, df_all, features_2d, params, centroid_2d, k, pca):
    mice  = sorted(df_all['mouse_name'].unique())
    n_m   = len(mice)
    n_col = min(3, n_m)
    n_row = int(np.ceil(n_m / n_col))
    colors = HABITAT_COLORS[:k]

    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(6 * n_col, 5 * n_row),
                             squeeze=False)
    var = pca.explained_variance_ratio_

    for i, mouse in enumerate(mice):
        ax   = axes[i // n_col][i % n_col]
        mask = df_all['mouse_name'].values == mouse
        lbs  = df_all['Habitat'].values[mask].astype(int)
        f2d  = features_2d[mask]

        # all other mice as grey background
        ax.scatter(features_2d[~mask, 0], features_2d[~mask, 1],
                   c='#cccccc', s=2, alpha=0.15, linewidths=0)

        # this mouse's voxels
        for c in range(k):
            m2 = lbs == c
            if m2.sum() > 0:
                ax.scatter(f2d[m2, 0], f2d[m2, 1],
                           c=colors[c], s=12, alpha=0.7, linewidths=0,
                           label=f'H{c} (n={m2.sum()})')

        # global centroids
        for c in range(k):
            ax.scatter(*centroid_2d[c], c=colors[c], s=160, marker='*',
                       edgecolors='black', linewidths=0.7, zorder=5)

        # per-mouse centroid crosses
        feats_m = df_all[mask][params].values
        for c in range(k):
            m2 = lbs == c
            if m2.sum() > 0:
                # project per-mouse centroid
                mu_m  = feats_m[m2].mean(axis=0, keepdims=True)
                mu_2d = pca.transform(mu_m)[0]
                ax.scatter(*mu_2d, c=colors[c], s=90, marker='X',
                           edgecolors='black', linewidths=0.7, zorder=6)
                # line from global centroid to per-mouse centroid
                ax.plot([centroid_2d[c, 0], mu_2d[0]],
                        [centroid_2d[c, 1], mu_2d[1]],
                        color=colors[c], lw=1.2, linestyle='--', alpha=0.7)

        short = mouse[:25] + '…' if len(mouse) > 25 else mouse
        ax.set_title(short, fontsize=8, fontweight='bold')
        ax.set_xlabel(f'PC1 ({100*var[0]:.1f}%)', fontsize=7)
        ax.set_ylabel(f'PC2 ({100*var[1]:.1f}%)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.legend(fontsize=6, markerscale=1.5, framealpha=0.6,
                  loc='upper right')

    # hide unused axes
    for j in range(n_m, n_row * n_col):
        axes[j // n_col][j % n_col].set_visible(False)

    # legend: star = global centroid, X = per-mouse centroid
    star_h = mpatches.Patch(color='none', label='★ global centroid')
    x_h    = mpatches.Patch(color='none', label='✕ per-mouse centroid')
    fig.legend(handles=[star_h, x_h], loc='lower center',
               ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0))

    fig.suptitle('Per-mouse voxels in global PCA space\n'
                 '(★ = global centroid  ✕ = per-mouse centroid  '
                 'dashed line = shift between them)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── 3. Parallel coordinates ─────────────────────────────────────────────────

def plot_parallel_coords(pdf, df_all, params, centroid_mat, k):
    mice   = sorted(df_all['mouse_name'].unique())
    colors = HABITAT_COLORS[:k]
    n_p    = len(params)
    x      = np.arange(n_p)

    # one figure per mouse
    for mouse in mice:
        mask   = df_all['mouse_name'].values == mouse
        lbs    = df_all['Habitat'].values[mask].astype(int)
        feats  = df_all[mask][params].values

        fig, ax = plt.subplots(figsize=(max(10, n_p * 1.5), 5))

        for c in range(k):
            m2 = lbs == c
            if m2.sum() == 0:
                continue
            f_c = feats[m2]

            # individual voxels (thin, very transparent)
            for row in f_c:
                ax.plot(x, row, color=colors[c], alpha=0.04, lw=0.5)

            # per-mouse centroid (thick solid)
            mu_m = f_c.mean(axis=0)
            ax.plot(x, mu_m, color=colors[c], lw=2.5, alpha=0.9,
                    label=f'H{c} per-mouse (n={m2.sum()})', zorder=4)

            # global centroid (dashed black)
            ax.plot(x, centroid_mat[c], color=colors[c], lw=2.0,
                    linestyle='--', alpha=0.6, zorder=3)

        # global centroids legend entry (one dummy)
        ax.plot([], [], color='black', lw=2.0, linestyle='--',
                label='global centroid (dashed)')
        ax.axhline(0, color='grey', lw=0.8, linestyle=':', alpha=0.5,
                   label='reference = 0')

        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Normalised value', fontsize=9)
        short = mouse[:40] + '…' if len(mouse) > 40 else mouse
        ax.set_title(f'{short} — parallel coordinates\n'
                     'Solid = per-mouse mean  |  Dashed = global centroid',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0.8, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', linestyle='-', alpha=0.1)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ── 4. Centroid deviation heatmap ───────────────────────────────────────────

def plot_deviation_heatmap(pdf, df_all, params, centroid_mat, k):
    mice   = sorted(df_all['mouse_name'].unique())
    colors = HABITAT_COLORS[:k]

    # For each (mouse, habitat): compute per-feature deviation and Euclidean dist
    # Shape: n_mice × k × n_params  (signed deviation from global centroid)
    n_m = len(mice)
    n_p = len(params)
    dev = np.full((n_m, k, n_p), np.nan)
    euc = np.full((n_m, k), np.nan)

    for i, mouse in enumerate(mice):
        mask  = df_all['mouse_name'].values == mouse
        lbs   = df_all['Habitat'].values[mask].astype(int)
        feats = df_all[mask][params].values
        for c in range(k):
            m2 = lbs == c
            if m2.sum() == 0:
                continue
            mu_m      = feats[m2].mean(axis=0)
            delta     = mu_m - centroid_mat[c]
            dev[i, c] = delta
            euc[i, c] = float(np.linalg.norm(delta))

    # ── subplot layout: k rows (one per habitat) × n_params+1 columns
    fig = plt.figure(figsize=(max(14, (n_p + 2) * 1.4), 3.5 * k))
    gs  = gridspec.GridSpec(k, n_p + 2, figure=fig,
                            hspace=0.6, wspace=0.15)

    vmax_feat = np.nanmax(np.abs(dev))
    vmax_euc  = np.nanmax(euc)

    short_mice = [m[:18] + '…' if len(m) > 18 else m for m in mice]

    for c in range(k):
        # per-feature deviation columns
        for j, p in enumerate(params):
            ax = fig.add_subplot(gs[c, j])
            col = dev[:, c, j].reshape(-1, 1)
            im  = ax.imshow(col, cmap='RdBu_r',
                            vmin=-vmax_feat, vmax=vmax_feat,
                            aspect='auto')
            ax.set_xticks([0])
            ax.set_xticklabels([p], rotation=40, ha='right', fontsize=7)
            ax.set_yticks(range(n_m))
            if j == 0:
                ax.set_yticklabels(short_mice, fontsize=7)
                hab_label = ax.set_ylabel(
                    f'H{c}', fontsize=11, fontweight='bold',
                    color=colors[c], rotation=0, labelpad=35)
            else:
                ax.set_yticklabels([])

            # annotate cell value
            for i in range(n_m):
                v = dev[i, c, j]
                if not np.isnan(v):
                    txt_col = 'white' if abs(v) > vmax_feat * 0.55 else 'black'
                    ax.text(0, i, f'{v:+.2f}', ha='center', va='center',
                            fontsize=6, color=txt_col)

        # Euclidean distance column (last)
        ax = fig.add_subplot(gs[c, n_p])
        col_e = euc[:, c].reshape(-1, 1)
        ax.imshow(col_e, cmap='Oranges',
                  vmin=0, vmax=vmax_euc, aspect='auto')
        ax.set_xticks([0])
        ax.set_xticklabels(['‖Δ‖', ], rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([])
        for i in range(n_m):
            v = euc[i, c]
            if not np.isnan(v):
                ax.text(0, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color='black')

        # empty last column for colorbar breathing room
        fig.add_subplot(gs[c, n_p + 1]).axis('off')

    # colorbars
    cax1 = fig.add_axes([0.91, 0.55, 0.012, 0.35])
    fig.colorbar(ScalarMappable(norm=Normalize(-vmax_feat, vmax_feat),
                                cmap='RdBu_r'),
                 cax=cax1, label='Signed deviation\n(per-mouse mean − global centroid)')

    cax2 = fig.add_axes([0.91, 0.12, 0.012, 0.35])
    fig.colorbar(ScalarMappable(norm=Normalize(0, vmax_euc),
                                cmap='Oranges'),
                 cax=cax2, label='Euclidean distance\nto global centroid')

    fig.suptitle('Per-mouse centroid deviation from global centroid\n'
                 'Blue = below global  |  Red = above global  |  '
                 'Orange = total distance',
                 fontsize=12, fontweight='bold', y=1.01)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def main(result_dir):
    print(f"Loading results from: {result_dir}")
    df_all, params, centroid_mat, k = load_results(result_dir)

    features = df_all[params].values
    labels   = df_all['Habitat'].values.astype(int)

    print(f"  {len(df_all)} voxels  |  {len(params)} params  |  k={k}")
    print(f"  Params: {params}")

    # Global PCA fit
    pca         = PCA(n_components=2, random_state=0)
    features_2d = pca.fit_transform(features)
    centroid_2d = pca.transform(centroid_mat)
    print(f"  PCA variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}  "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")

    png_dir  = os.path.join(result_dir, 'joint_boundaries_pages')
    os.makedirs(png_dir, exist_ok=True)
    out_path = os.path.join(result_dir, 'joint_boundaries_analysis.pdf')
    _page    = [0]   # mutable counter shared across helpers via closure

    class _DualSaver:
        """Saves every savefig call to both PDF and a numbered PNG."""
        def __init__(self, pdf):
            self._pdf = pdf
        def savefig(self, fig, **kw):
            self._pdf.savefig(fig, **kw)
            n = _page[0]
            fig.savefig(os.path.join(png_dir, f'page_{n:02d}.png'),
                        dpi=120, bbox_inches='tight')
            _page[0] += 1

    with PdfPages(out_path) as _pdf:
        pdf = _DualSaver(_pdf)

        # ── page 1: global PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_global_pca(ax, features_2d, labels, centroid_2d, k, pca)
        fig.suptitle('Global PCA — all voxels from all mice\n'
                     '★ = global centroid', fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── pages 2+: per-mouse PCA grid
        plot_per_mouse_pca(pdf, df_all, features_2d, params, centroid_2d, k, pca)

        # ── pages 3+: parallel coordinates (one page per mouse)
        plot_parallel_coords(pdf, df_all, params, centroid_mat, k)

        # ── last page: deviation heatmap
        plot_deviation_heatmap(pdf, df_all, params, centroid_mat, k)

    print(f"\nPDF saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', nargs='?',
                        default=r'C:\Users\laboratorio\Desktop\Projet 1\dossier_test\test_2307\joint_clustering_12',
                        help='Path to joint_clustering_* output directory')
    args = parser.parse_args()
    main(args.result_dir)
