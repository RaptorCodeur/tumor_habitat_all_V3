# =================================================================
# pipeline/joint_clustering.py — Joint multi-mouse clustering pipeline
# =================================================================

import os
import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

from sklearn.mixture import GaussianMixture

from pipeline.joint_loading  import load_joint_dataset
from pipeline.selection      import select_optimal_k
from pipeline.clustering     import run_clustering
from pipeline.export         import export_results, save_centroids_json
from pipeline.labeling       import show_habitat_map
from pipeline.spatial        import _necrosis_guard, _compute_dtopo
from pipeline.habitat_atlas  import _balanced_sample, VOXELS_PER_MOUSE
from config                  import (N_JOBS_GAP, HABITAT_COLORS_HEX,
                                      GMM_SILHOUETTE_GUARD, KMEANS_GAP_GUARD,
                                      JOINT_MIN_CLUSTER_SIZE_RATIO,
                                      DTOPO_JOINT_WEIGHT,
                                      DTOPO_MIN_NECROSIS_FRACTION,
                                      RANDOM_SEED)

_NORM_YLABEL = {
    'cl'           : 'CL-normalized value  (0 = contralateral tissue)',
    'cl_hybrid'    : 'CL-hybrid value  (0 = contralateral tissue, unit = tumor IQR)',
    'robust'       : "Robust-scaled value  (0 = each mouse's tumor median)",
    'robust_global': 'Global robust-scaled value  (0 = global tumor median)',
}
_NORM_REFLABEL = {
    'cl'           : 'Contralateral reference (0)',
    'cl_hybrid'    : 'Contralateral reference (0)',
    'robust'       : 'Per-mouse tumor median (0)',
    'robust_global': 'Global tumor median (0)',
}
_NORM_TAG = {
    'cl'           : 'CL-normalized',
    'cl_hybrid'    : 'CL-hybrid',
    'robust'       : 'per-mouse robust',
    'robust_global': 'global robust',
}


# ------------------------------------------------------------------
# Global profile chart
# ------------------------------------------------------------------

def _plot_global_profiles(centroids, parameters, best_k, output_dir, normalization='cl'):
    """
    Bar chart of universal habitat centroid profiles.
    y-axis label adapts to the normalization used.
    Habitats are shown as H0, H1, … (no biological labels).
    """
    x     = np.arange(len(parameters))
    width = 0.8 / best_k

    ylabel   = _NORM_YLABEL.get(normalization, _NORM_YLABEL['robust'])
    reflabel = _NORM_REFLABEL.get(normalization, _NORM_REFLABEL['robust'])
    norm_tag = _NORM_TAG.get(normalization, normalization)
    title    = (f'Joint habitat profiles — {best_k} universal habitats'
                f'  [{norm_tag}]')

    fig, ax = plt.subplots(figsize=(max(10, len(parameters) * 1.8), 5))
    for c in range(best_k):
        vals   = [centroids[c].get(p, 0.0) for p in parameters]
        color  = HABITAT_COLORS_HEX[c % len(HABITAT_COLORS_HEX)]
        offset = (c - best_k / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=f"H{c}",
               color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', label=reflabel)
    ax.set_xticks(x)
    ax.set_xticklabels(parameters, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'joint_profiles.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Global profile chart saved: {out_path}")


# ------------------------------------------------------------------
# Multi-slice segmentation grid (all slices, one page per mouse)
# ------------------------------------------------------------------

def _plot_all_slices(df_mouse, labels_mouse, best_k, mouse_name, output_dir,
                     confidence_mouse=None):
    slices = sorted(df_mouse['Slice'].unique())
    n_sl   = len(slices)
    if n_sl == 0:
        return None

    n_cols   = min(n_sl, 5)
    n_rows   = int(np.ceil(n_sl / n_cols))
    colors   = HABITAT_COLORS_HEX[:best_k]
    seg_cmap = ListedColormap(colors)
    has_conf = confidence_mouse is not None

    n_blocks  = 2 if has_conf else 1
    fig_h     = n_rows * 3 * n_blocks + 0.8
    fig       = plt.figure(figsize=(n_cols * 3, fig_h))
    gs        = gridspec.GridSpec(n_blocks * n_rows + 1, n_cols,
                                  figure=fig,
                                  height_ratios=[3] * (n_blocks * n_rows) + [0.4],
                                  hspace=0.05, wspace=0.05)

    def _build_grid(df_s, values, dtype=int):
        x_vals = sorted(df_s['X'].unique())
        y_vals = sorted(df_s['Y'].unique())
        x_idx  = {x: i for i, x in enumerate(x_vals)}
        y_idx  = {y: i for i, y in enumerate(y_vals)}
        grid   = np.full((len(y_vals), len(x_vals)),
                         np.nan if dtype == float else -1, dtype=float)
        grid[df_s['Y'].map(y_idx).values,
             df_s['X'].map(x_idx).values] = values
        return grid

    for idx, sl in enumerate(slices):
        row  = idx // n_cols
        col  = idx % n_cols
        mask = df_mouse['Slice'].values == sl
        df_s = df_mouse[mask].copy()
        lb_s = labels_mouse[mask]

        # --- segmentation ---
        ax_seg = fig.add_subplot(gs[row, col])
        grid_s = _build_grid(df_s, lb_s.astype(int))
        masked = np.ma.masked_where(grid_s == -1, grid_s)
        ax_seg.imshow(masked, cmap=seg_cmap, vmin=0, vmax=best_k - 1,
                      interpolation='nearest', origin='upper')
        ax_seg.set_title(f'Slice {int(sl)}', fontsize=7, pad=2)
        ax_seg.axis('off')
        if col == 0 and row == 0:
            ax_seg.set_ylabel('Habitat', fontsize=8)

        # --- confidence ---
        if has_conf:
            ax_cf = fig.add_subplot(gs[n_rows + row, col])
            conf_s = confidence_mouse[mask]
            grid_c = _build_grid(df_s, conf_s, dtype=float)
            masked_c = np.ma.masked_where(np.isnan(grid_c), grid_c)
            im = ax_cf.imshow(masked_c, cmap='RdYlGn', vmin=0.0, vmax=1.0,
                              interpolation='nearest', origin='upper')
            ax_cf.axis('off')
            if col == 0 and row == 0:
                ax_cf.set_ylabel('Confidence', fontsize=8)

    # Hide unused cells
    for idx in range(n_sl, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        fig.add_subplot(gs[row, col]).axis('off')
        if has_conf:
            fig.add_subplot(gs[n_rows + row, col]).axis('off')

    # Habitat legend (bottom strip)
    ax_leg = fig.add_subplot(gs[-1, :])
    ax_leg.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c % len(colors)])
               for c in range(best_k)]
    labels_leg = [f'H{c}' for c in range(best_k)]
    if has_conf:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_leg, orientation='horizontal',
                            fraction=0.4, pad=0.05, aspect=30)
        cbar.set_label('GMM confidence (max posterior)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax_leg.legend(handles, labels_leg, loc='upper left',
                  ncol=best_k, fontsize=8, frameon=False)

    fig.suptitle(f'{mouse_name} — segmentation  |  confidence',
                 fontsize=11, fontweight='bold', y=1.01)

    out_path = os.path.join(output_dir, 'segmentation_all_slices.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


# ------------------------------------------------------------------
# Joint PDF report
# ------------------------------------------------------------------

def generate_joint_report_pdf(df_pooled, common_params, mice_list,
                               centroids, best_k, normalization,
                               method, output_dir, log_fn=print):
    """
    Multi-page PDF report for joint clustering results.
    Pages: title, global summary, global profile chart,
           then per-mouse segmentation (all slices) + habitat breakdown.
    """
    pdf_path  = os.path.join(output_dir, 'joint_report.pdf')
    doc_name  = os.path.basename(output_dir)          # e.g. "joint_clustering_5"
    norm_tag  = _NORM_TAG.get(normalization, normalization)
    n_mice    = len(mice_list)
    n_total   = len(df_pooled)
    labels    = df_pooled['Habitat'].values.astype(int)
    features  = np.nan_to_num(df_pooled[common_params].values, nan=0.0)

    def _make_text_page():
        """
        Create a full-bleed figure where the axes occupies the whole page.
        Returns (fig, ax) with xlim/ylim=[0,1] and axis off.
        Using add_axes([0,0,1,1]) instead of plt.subplots() avoids the default
        subplot margins that cause text to be mis-positioned when combined with
        bbox_inches='tight'.
        """
        fig = plt.figure(figsize=(11, 8.5))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig, ax

    def _header(ax, subtitle=''):
        """Running header: thin rule near top + small italic label."""
        header = doc_name if not subtitle else f"{doc_name}  —  {subtitle}"
        # Axes has xlim/ylim=[0,1] so data coords == axes fraction coords.
        ax.plot([0, 1], [0.965, 0.965], color='#cccccc', linewidth=0.5,
                transform=ax.transAxes)
        ax.text(0.5, 0.975, header,
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, color='#888888', style='italic')

    def _embed_image(fpath, title, subtitle=''):
        """Embed a PNG preserving its aspect ratio (no distortion)."""
        img  = mpimg.imread(fpath)
        h, w = img.shape[:2]
        fig_w = 11.0
        # Cap height at 9 inches; always leave 0.5 inch for the title bar.
        fig_h = min(9.0, fig_w * h / w) + 0.5
        fig  = plt.figure(figsize=(fig_w, fig_h))
        # Image axes: full width, leaving room at bottom for title text.
        title_frac = 0.5 / fig_h          # fraction of figure height for title
        ax   = fig.add_axes([0, title_frac, 1, 1 - title_frac])
        ax.imshow(img)
        ax.axis('off')
        _header(ax, subtitle)
        fig.text(0.5, title_frac * 0.5, title,
                 ha='center', va='center',
                 fontsize=12, fontweight='bold', color='#1F3864')
        return fig

    with PdfPages(pdf_path) as pdf:

        # ── Title page ───────────────────────────────────────────────
        fig, ax = _make_text_page()
        T = ax.transAxes
        ax.text(0.5, 0.88, doc_name,
                ha='center', fontsize=14, color='#888888', style='italic',
                transform=T)
        ax.text(0.5, 0.72, 'Joint Tumor Habitat Clustering',
                ha='center', fontsize=30, fontweight='bold', color='#1F3864',
                transform=T)
        ax.text(0.5, 0.60, 'Multi-mouse Analysis Report',
                ha='center', fontsize=20, color='#2E75B6', transform=T)
        ax.text(0.5, 0.50,
                f'Method : {method.upper()}  |  Universal habitats : {best_k}  |  '
                f'Normalization : {norm_tag}',
                ha='center', fontsize=13, color='#444444', transform=T)
        ax.text(0.5, 0.43,
                f'Mice ({n_mice}) : {", ".join(m["name"] for m in mice_list)}',
                ha='center', fontsize=11, color='#444444', transform=T)
        ax.text(0.5, 0.36,
                f'Total pooled voxels : {n_total:,}',
                ha='center', fontsize=11, color='#444444', transform=T)
        ax.text(0.5, 0.29,
                f'Parameters : {", ".join(common_params)}',
                ha='center', fontsize=10, color='#666666', transform=T)
        ax.text(0.5, 0.18, datetime.datetime.now().strftime('%B %d, %Y'),
                ha='center', fontsize=12, color='#888888', transform=T)
        _header(ax)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Global habitat summary ───────────────────────────────────
        fig, ax = _make_text_page()
        T = ax.transAxes
        norm_desc = ('0 = contralateral tissue level' if normalization == 'cl'
                     else "0 = each mouse's tumor median")
        ax.text(0.05, 0.92, 'Global Habitat Summary', fontsize=18,
                fontweight='bold', color='#1F3864', va='top', transform=T)
        ax.text(0.05, 0.87,
                f'Values are {norm_tag}  ({norm_desc})',
                fontsize=10, color='#888888', va='top', style='italic',
                transform=T)

        y = 0.81
        step = min(0.12, (y - 0.04) / max(best_k, 1))  # shrink if many habitats
        for c in range(best_k):
            n_c   = int((labels == c).sum())
            pct   = 100 * n_c / n_total
            color = HABITAT_COLORS_HEX[c % len(HABITAT_COLORS_HEX)]
            ax.add_patch(plt.Rectangle((0.03, y - 0.025), 0.012, 0.025,
                                       color=color, transform=T))
            ax.text(0.06, y, f'H{c}',
                    fontsize=13, fontweight='bold', color='#1F3864', va='top',
                    transform=T)
            ax.text(0.12, y, f'{n_c:,} voxels ({pct:.1f}% of total)',
                    fontsize=11, color='#444444', va='top', transform=T)
            profile_str = '   '.join(
                f"{p}: {centroids[c].get(p, 0.0):+.2f}"
                for p in common_params
            )
            ax.text(0.06, y - 0.04, profile_str,
                    fontsize=8, color='#555555', va='top', family='monospace',
                    transform=T)
            y -= step
        _header(ax, 'Global Habitat Summary')
        pdf.savefig(fig)
        plt.close(fig)

        # ── Global profile chart ─────────────────────────────────────
        profile_png = os.path.join(output_dir, 'joint_profiles.png')
        if os.path.exists(profile_png):
            fig = _embed_image(profile_png,
                               'Universal habitat profiles (all mice)',
                               'Universal Profiles')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ── Global centroid heatmap ───────────────────────────────────
        vis_params = [p for p in common_params if p != 'd_topo_norm']
        if centroids and vis_params:
            mat = np.array([
                [centroids[c].get(p, 0.0) for p in vis_params]
                for c in range(best_k) if c in centroids
            ])
            plbls = [p.replace('DTI-', '').replace('T2star', 'T2*')
                     for p in vis_params]
            vabs  = max(float(np.nanmax(np.abs(mat))), 1e-6)
            fig_w = max(8.0, len(vis_params) * 1.1)
            fig_h = max(3.5, best_k * 0.7 + 1.5)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                           vmin=-vabs, vmax=vabs)
            ax.set_xticks(range(len(vis_params)))
            ax.set_xticklabels(plbls, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(mat)))
            ax.set_yticklabels([f'H{c}' for c in range(len(mat))], fontsize=9)
            cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=8)
            cb.set_label('Normalized value', fontsize=9)
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat[r, c]
                    ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                            fontsize=8,
                            color='black' if abs(v) < vabs * 0.6 else 'white')
            ax.set_title(
                f'Universal habitat centroids — {best_k} habitats  [{norm_tag}]',
                fontsize=12, fontweight='bold', pad=10)
            plt.tight_layout()
            _header(ax, 'Universal Centroids Heatmap')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ── Per-mouse pages ──────────────────────────────────────────
        for i, mouse in enumerate(mice_list):
            name     = mouse['name']
            mask     = df_pooled['mouse_id'].values == i
            df_m     = df_pooled[mask].reset_index(drop=True)
            labels_m = labels[mask]
            feats_m  = features[mask]
            n_m      = int(mask.sum())

            # --- Mouse header page ---
            fig, ax = _make_text_page()
            T = ax.transAxes
            ax.text(0.5, 0.88, name, ha='center', fontsize=22,
                    fontweight='bold', color='#1F3864', transform=T)
            ax.text(0.5, 0.81, f'{n_m:,} voxels',
                    ha='center', fontsize=14, color='#444444', transform=T)

            y    = 0.72
            step = min(0.10, (y - 0.04) / max(best_k, 1))
            for c in range(best_k):
                n_c   = int((labels_m == c).sum())
                pct   = 100 * n_c / n_m if n_m > 0 else 0
                color = HABITAT_COLORS_HEX[c % len(HABITAT_COLORS_HEX)]
                ax.add_patch(plt.Rectangle((0.15, y - 0.022), 0.012, 0.022,
                                           color=color, transform=T))
                ax.text(0.18, y, f'H{c} : {n_c:,} voxels ({pct:.1f}%)',
                        fontsize=12, color='#333333', va='top', transform=T)
                if n_c > 0:
                    cen_m = {p: float(feats_m[labels_m == c, j].mean())
                             for j, p in enumerate(common_params)}
                    cen_str = '   '.join(f"{p}: {cen_m[p]:+.2f}"
                                         for p in common_params)
                    ax.text(0.18, y - 0.038, cen_str,
                            fontsize=8, color='#666666', va='top',
                            family='monospace', transform=T)
                y -= step

            _header(ax, name)
            pdf.savefig(fig)
            plt.close(fig)

            # --- All-slices segmentation grid ---
            mouse_dir  = os.path.join(output_dir, name)
            slices_png = os.path.join(mouse_dir, 'segmentation_all_slices.png')
            if os.path.exists(slices_png):
                fig = _embed_image(slices_png,
                                   f'{name} — segmentation (all slices)',
                                   name)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    log_fn(f"Joint PDF report saved: {pdf_path}")
    return pdf_path


# ------------------------------------------------------------------
# Two-pass d_topo helper
# ------------------------------------------------------------------

def _compute_joint_dtopo(df_pooled, labels, centroids, common_params,
                          min_necrosis_fraction=DTOPO_MIN_NECROSIS_FRACTION,
                          log_fn=print):
    """
    After Pass 1 clustering, compute per-mouse d_topo_norm using the necrosis
    habitat identified in the global centroids as the seed.

    Per-mouse guard: if a mouse has fewer than min_necrosis_fraction of its
    voxels in the necrosis cluster, d_topo is set to 0 for that mouse so it
    does not influence cluster assignment (avoids spatial mis-classification in
    mice with little or no real necrosis).

    Returns
    -------
    dtopo_all : float32 array aligned with df_pooled rows, or None
    report    : dict  {mouse_name: {'included': bool, 'fraction': float,
                                    'n_voxels': int, 'reason': str}}
    """
    passes, nec_id = _necrosis_guard(centroids, common_params)
    if not passes:
        log_fn("[d_topo] Necrosis guard failed on global centroids — d_topo skipped")
        return None, {}

    log_fn(f"[d_topo] Necrosis habitat: H{nec_id} — computing per-mouse dtopo…")

    mouse_ids = df_pooled['mouse_id'].values
    n_total   = len(df_pooled)
    dtopo_all = np.zeros(n_total, dtype=np.float32)   # 0 = neutral (no effect)
    report    = {}

    for mid in np.unique(mouse_ids):
        mask  = mouse_ids == mid
        df_m  = df_pooled[mask].reset_index(drop=True)
        lbs_m = labels[mask]
        name  = df_pooled['mouse_name'].values[mask][0]
        n_m   = int(mask.sum())

        nec_frac = float((lbs_m == nec_id).sum()) / n_m

        if nec_frac < min_necrosis_fraction:
            log_fn(f"  [{name}] d_topo SKIPPED — necrosis fraction "
                   f"{100*nec_frac:.1f}% < {100*min_necrosis_fraction:.0f}% threshold "
                   f"→ d_topo set to 0 for this mouse")
            report[name] = {
                'included' : False,
                'fraction' : nec_frac,
                'n_voxels' : n_m,
                'reason'   : (f"Necrosis fraction {100*nec_frac:.1f}% is below the "
                               f"{100*min_necrosis_fraction:.0f}% threshold — this mouse "
                               f"does not have enough necrotic tissue for d_topo to be "
                               f"meaningful. Setting d_topo=0 prevents spatially-driven "
                               f"misclassification."),
            }
            # dtopo_all[mask] stays 0 → no influence on clustering
            continue

        dtopo_m = _compute_dtopo(df_m, lbs_m, nec_id)
        valid   = ~np.isnan(dtopo_m)

        if valid.sum() == 0:
            log_fn(f"  [{name}] d_topo SKIPPED — no necrosis seed voxels in any slice")
            report[name] = {
                'included' : False,
                'fraction' : nec_frac,
                'n_voxels' : n_m,
                'reason'   : (f"Necrosis fraction {100*nec_frac:.1f}% passed the threshold "
                               f"but no necrosis seed voxels were found in any individual "
                               f"slice — topological distance cannot be computed."),
            }
            continue

        vals = dtopo_m[valid]
        med  = float(np.median(vals))
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr  = max(q75 - q25, 1e-8)
        dtopo_m[valid]  = (vals - med) / iqr
        dtopo_m[~valid] = 0.0   # slices without necrosis seed → neutral
        dtopo_all[mask] = dtopo_m

        log_fn(f"  [{name}] d_topo INCLUDED (necrosis {100*nec_frac:.1f}%, "
               f"{valid.sum()} voxels scaled, median={med:.3f}, IQR={iqr:.3f})")
        report[name] = {
            'included' : True,
            'fraction' : nec_frac,
            'n_voxels' : n_m,
            'reason'   : (f"Necrosis fraction {100*nec_frac:.1f}% ≥ threshold — "
                           f"d_topo computed from {valid.sum()} voxels across all slices "
                           f"and added to the feature matrix with weight "
                           f"{DTOPO_JOINT_WEIGHT}."),
        }

    n_active = int(np.any(dtopo_all != 0.0))
    included = [m for m, r in report.items() if r['included']]
    if not included:
        log_fn("[d_topo] No mice passed the necrosis guard — d_topo skipped entirely")
        return None, report

    log_fn(f"[d_topo] d_topo active for {len(included)}/{len(report)} mice: "
           f"{', '.join(included)}")
    return dtopo_all, report


# ------------------------------------------------------------------
# Balanced GMM helpers for per-mouse weighting
# ------------------------------------------------------------------

def _fit_balanced_gmm(features, mouse_ids, k, n_init):
    """Train a GMM on a balanced subsample (VOXELS_PER_MOUSE per mouse)."""
    bal = _balanced_sample(features, mouse_ids)
    gmm = GaussianMixture(
        n_components=k, n_init=n_init,
        covariance_type='full', random_state=RANDOM_SEED,
        max_iter=300, init_params='kmeans')
    gmm.fit(bal)
    return gmm


def _select_k_gmm_balanced(features, mouse_ids, k_range, n_init, log_fn=print):
    """
    BIC-based K selection using balanced-subsample GMMs.
    Training uses VOXELS_PER_MOUSE voxels per mouse; BIC is evaluated
    on all voxels (same strategy as the Atlas).
    Returns (best_k, gmm_cache).
    """
    bic_dict  = {}
    gmm_cache = {}
    log_fn(f"Balanced GMM K-selection ({VOXELS_PER_MOUSE} vox/mouse, "
           f"K={list(k_range)}) …")
    for k in k_range:
        gmm          = _fit_balanced_gmm(features, mouse_ids, k, n_init)
        bic_dict[k]  = gmm.bic(features)
        gmm_cache[k] = gmm
        log_fn(f"  K={k}  BIC={bic_dict[k]:.1f}")
    best_k = min(bic_dict, key=bic_dict.get)
    log_fn(f"  → Balanced-GMM best K = {best_k}")
    return best_k, gmm_cache


# ------------------------------------------------------------------
# Main joint pipeline
# ------------------------------------------------------------------

def run_joint_pipeline(mice_list, method, k_range, n_init_gmm, n_refs,
                        output_dir, log_fn=print, k_override=None,
                        normalization='cl',
                        gmm_silhouette_guard=GMM_SILHOUETTE_GUARD,
                        kmeans_gap_guard=KMEANS_GAP_GUARD,
                        min_cluster_ratio=JOINT_MIN_CLUSTER_SIZE_RATIO,
                        dtopo_weight=DTOPO_JOINT_WEIGHT,
                        dtopo_min_frac=DTOPO_MIN_NECROSIS_FRACTION,
                        use_mouse_weighting=True):
    """
    Joint clustering pipeline across N mice.

    Steps
    -----
    1. Load + normalize all mice (CL or per-mouse robust)
    2. Pool all voxels into one feature matrix
    3. Select optimal K  (balanced-GMM BIC when use_mouse_weighting=True)
    4. Run clustering → universal H0…Hk
       GMM: trained on balanced subsample, predicts on all voxels.
       K-means: uses sample_weight=1/n_i per voxel.
    5. Compute global centroids
    6. Two-pass: add d_topo_norm if necrosis detected

    Returns dict with: parameters, df_pooled, labels, features,
                       centroids, best_k, normalization, output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load + normalize
    common_params, df_pooled = load_joint_dataset(
        mice_list, normalization=normalization, log_fn=log_fn)
    features  = np.nan_to_num(df_pooled[common_params].values, nan=0.0)
    n_total   = len(features)
    mouse_ids = df_pooled['mouse_id'].values.astype(int)
    log_fn(f"Feature matrix: {n_total} voxels × {len(common_params)} params")

    # Per-mouse weighting: 1/n_i per voxel so each mouse contributes equally
    if use_mouse_weighting:
        n_mice_       = len(mice_list)
        mouse_counts  = np.bincount(mouse_ids, minlength=n_mice_).astype(np.float64)
        sample_weight = 1.0 / mouse_counts[mouse_ids]
        sample_weight /= sample_weight.sum()
        log_fn(f"Per-mouse weighting enabled (1/n_i, {VOXELS_PER_MOUSE} vox/mouse for GMM fit)")
    else:
        sample_weight = None

    # 2. K selection
    if k_override and k_override >= 2:
        best_k, gmm_cache = k_override, {}
        log_fn(f"K override: {best_k} (size guard skipped)")
    else:
        log_fn(f"Selecting optimal K (method={method})…")
        if use_mouse_weighting and method == 'gmm':
            best_k, gmm_cache = _select_k_gmm_balanced(
                features, mouse_ids, k_range, n_init_gmm, log_fn)
        else:
            best_k, gmm_cache = select_optimal_k(
                features, method=method, df_all=None,
                k_range=k_range, n_refs=n_refs, n_jobs=N_JOBS_GAP,
                n_init_gmm=n_init_gmm,
                gmm_silhouette_guard=gmm_silhouette_guard,
                kmeans_gap_guard=kmeans_gap_guard,
                output_dir=output_dir)
        log_fn(f"Optimal K = {best_k}")

    # 3. Clustering — size guard loop (only when K is auto-selected)
    current_k = best_k
    while True:
        log_fn(f"Running joint {method.upper()} clustering (k={current_k})…")

        # For weighted GMM: ensure the balanced model is in cache for current_k
        active_cache = gmm_cache
        if use_mouse_weighting and method == 'gmm' and current_k not in gmm_cache:
            active_cache = {current_k: _fit_balanced_gmm(
                features, mouse_ids, current_k, n_init_gmm)}

        labels, extra_info = run_clustering(
            features, df_pooled, current_k,
            method=method, n_init_gmm=n_init_gmm, gmm_cache=active_cache,
            sample_weight=sample_weight if (use_mouse_weighting and method == 'kmeans') else None)

        if k_override and k_override >= 2:
            best_k = current_k
            break

        small = []
        for c in range(current_k):
            n_c   = int((labels == c).sum())
            ratio = n_c / n_total
            if ratio < min_cluster_ratio:
                small.append((c, n_c, ratio))

        if not small:
            best_k = current_k
            log_fn(f"Size guard passed at k={current_k}")
            break

        for c, n_c, ratio in small:
            log_fn(f"  [size guard] H{c}: {n_c} voxels "
                   f"({100*ratio:.1f}% < {100*min_cluster_ratio:.0f}%) — too small")

        if current_k <= 2:
            log_fn(f"  [size guard] k_min=2 reached — keeping k=2")
            best_k = current_k
            break

        current_k -= 1
        log_fn(f"  [size guard] Reducing k → {current_k}")

    df_pooled = df_pooled.copy()
    df_pooled['Habitat'] = labels

    proba = extra_info.get('proba')
    if proba is not None:
        df_pooled['confidence'] = proba.max(axis=1).astype(np.float32)
    else:
        df_pooled['confidence'] = np.ones(len(labels), dtype=np.float32)

    # 4. Global centroids (Pass 1)
    centroids = {
        c: {p: float(features[labels == c, i].mean())
            for i, p in enumerate(common_params)}
        for c in range(best_k)
    }

    # 5. Two-pass: add d_topo_norm if necrosis detected in global centroids
    dtopo_arr, dtopo_report = _compute_joint_dtopo(
        df_pooled, labels, centroids, common_params,
        min_necrosis_fraction=dtopo_min_frac, log_fn=log_fn)
    if dtopo_arr is not None:
        log_fn(f"Pass 2: re-running clustering with d_topo_norm "
               f"(weight={dtopo_weight})…")
        df_pooled['d_topo_norm'] = dtopo_arr * dtopo_weight
        common_params = common_params + ['d_topo_norm']
        features      = np.nan_to_num(df_pooled[common_params].values, nan=0.0)

        pass2_cache = {}
        if use_mouse_weighting and method == 'gmm':
            pass2_cache = {best_k: _fit_balanced_gmm(
                features, mouse_ids, best_k, n_init_gmm)}

        labels, extra_info = run_clustering(
            features, df_pooled, best_k,
            method=method, n_init_gmm=n_init_gmm, gmm_cache=pass2_cache,
            sample_weight=sample_weight if (use_mouse_weighting and method == 'kmeans') else None)

        df_pooled['Habitat'] = labels
        proba = extra_info.get('proba')
        if proba is not None:
            df_pooled['confidence'] = proba.max(axis=1).astype(np.float32)
        else:
            df_pooled['confidence'] = np.ones(len(labels), dtype=np.float32)

        centroids = {
            c: {p: float(features[labels == c, i].mean())
                for i, p in enumerate(common_params)}
            for c in range(best_k)
        }
        log_fn("Pass 2 complete — d_topo_norm included in final clustering")

    log_fn("Universal habitat sizes:")
    for c in range(best_k):
        n_c = int((labels == c).sum())
        log_fn(f"  H{c}: {n_c} voxels ({100 * n_c / n_total:.1f}%)")

    save_centroids_json(centroids, output_dir)

    return {
        'parameters'   : common_params,
        'df_pooled'    : df_pooled,
        'labels'       : labels,
        'features'     : features,
        'centroids'    : centroids,
        'best_k'       : best_k,
        'normalization': normalization,
        'extra_info'   : extra_info,
        'output_dir'   : output_dir,
        'dtopo_report'  : dtopo_report,
        'dtopo_weight'  : dtopo_weight,
        'dtopo_min_frac': dtopo_min_frac,
    }


# ------------------------------------------------------------------
# Per-mouse export
# ------------------------------------------------------------------

def export_per_mouse(df_pooled, common_params, mice_list,
                     output_dir, best_k, normalization='cl', log_fn=print):
    habitat_labels = {c: f"H{c}" for c in range(best_k)}
    features       = np.nan_to_num(df_pooled[common_params].values, nan=0.0)
    labels         = df_pooled['Habitat'].values.astype(int)

    centroids = {
        c: {p: float(features[labels == c, i].mean())
            for i, p in enumerate(common_params)}
        for c in range(best_k)
    }
    _plot_global_profiles(centroids, common_params, best_k,
                          output_dir, normalization)

    for i, mouse in enumerate(mice_list):
        name      = mouse['name']
        mouse_dir = os.path.join(output_dir, name)
        os.makedirs(mouse_dir, exist_ok=True)

        mask        = df_pooled['mouse_id'].values == i
        extra_cols  = ['confidence'] if 'confidence' in df_pooled.columns else []
        df_mouse    = (df_pooled[mask][['X', 'Y', 'Slice'] + common_params
                                       + ['Habitat'] + extra_cols]
                       .copy()
                       .reset_index(drop=True))
        lbs_m   = labels[mask]
        feats_m = features[mask]
        conf_m  = (df_pooled['confidence'].values[mask]
                   if 'confidence' in df_pooled.columns else None)

        export_results(df_mouse, common_params, habitat_labels, mouse_dir)
        show_habitat_map(lbs_m, df_mouse, feats_m, common_params,
                         best_k, output_dir=mouse_dir)
        _plot_all_slices(df_mouse, lbs_m, best_k, name, mouse_dir, conf_m)

        n_vox = int(mask.sum())
        log_fn(f"[{name}] {n_vox} voxels → {mouse_dir}")

    generate_joint_report_pdf(
        df_pooled, common_params, mice_list,
        centroids, best_k, normalization,
        method='joint', output_dir=output_dir, log_fn=log_fn,
    )
