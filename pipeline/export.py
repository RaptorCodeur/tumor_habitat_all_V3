# =================================================================
# pipeline/export.py — CSV export, PDF report, NIfTI mask
# =================================================================

import os
import datetime
import json
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


def save_run_meta(scaling_info, cl_stats, output_dir):
    """
    Persist scaling_info (tumor median/IQR) and cl_stats per run so that
    compare_v4 can re-express centroids in CL-relative space for cross-animal
    archetype matching.
    """
    meta = {
        "scaling_info": {p: {"median": float(v["median"]), "iqr": float(v["iqr"])}
                         for p, v in (scaling_info or {}).items()},
        "cl_stats"    : {p: {"median": float(v["median"]), "std": float(v["std"]),
                              "mean": float(v["mean"]), "n": int(v["n"])}
                         for p, v in (cl_stats or {}).items()},
    }
    path = os.path.join(output_dir, "run_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def save_centroids_json(centroids, output_dir):
    """
    Persist enriched centroids (IQR-space + d_topo_norm when available).
    estimate_lf_weights.py loads this file so spatial LFs can fire.
    """
    path = os.path.join(output_dir, "centroids.json")
    with open(path, "w") as f:
        json.dump({str(k): {p: float(v) for p, v in feat.items()}
                   for k, feat in centroids.items()}, f, indent=2)


def export_results(df_scaled, parameters, habitat_labels, output_dir):
    """Export voxel-level results with habitat labels to CSV."""
    df_export = df_scaled.copy()
    df_export['Habitat_label'] = df_export['Habitat'].map(
        lambda h: habitat_labels.get(int(h), f'Habitat {h}')
    )
    # Save only the resolved parameters (DTI-MD may still be in df_scaled even
    # when AD+RD are present and resolve_dti_parameters removed it from
    # `parameters`).  Saving it would cause load_mouse_data in
    # multi_mouse_discovery to compute centroids in a 7-feature space while
    # the comparative pipeline uses 6, producing different K selection and
    # therefore different meta-habitats between the standalone app and compare.
    save_cols = (['X', 'Y', 'Slice']
                 + [p for p in parameters if p in df_export.columns]
                 + ['Habitat', 'Habitat_label'])
    csv_path = os.path.join(output_dir, 'habitats_result.csv')
    df_export[save_cols].to_csv(csv_path, index=False)
    print(f"Results saved : {csv_path}")


def generate_report_pdf(output_dir, df_scaled, parameters,
                        habitat_labels, best_k, clustering_method):
    """
    Compile all saved PNG figures and a habitat summary into a PDF report.
    Title page includes method, K, voxel count, parameters, and date.
    """
    pdf_path = os.path.join(output_dir, 'habitat_report_V2.pdf')

    figures = [
        ('1_correlation.png',    'Inter-modality Spearman correlations'),
        ('3_habitat_map.png',    'Habitat map + multiparametric profiles'),
        ('3_barchart.png',       'Habitat profiles -- bar chart'),
        ('4_heatmap.png',        'Habitat profiles -- heatmap'),
        ('5_radar_profiles.png', 'Habitat profiles -- radar chart'),
    ]
    if clustering_method == 'gmm':
        figures.insert(1, ('2_bic_silhouette.png',
                           'Optimal k -- BIC + Silhouette'))
        figures.append(('6_gmm_probabilities.png',
                        'GMM -- cluster assignment confidence'))
    elif clustering_method == 'kmeans':
        figures.insert(1, ('2_kmeans_silhouette.png',
                           'Optimal k -- Silhouette + Gap Statistic (feature space)'))
    else:
        figures.insert(1, ('2_silhouette.png',
                           'Optimal k -- Silhouette + Gap Statistic (spectral space)'))

    with PdfPages(pdf_path) as pdf:

        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.65, 'Tumor Habitat Segmentation',
                ha='center', fontsize=28, fontweight='bold', color='#1F3864')
        ax.text(0.5, 0.52, 'Analysis Report -- V2',
                ha='center', fontsize=20, color='#2E75B6')
        ax.text(0.5, 0.42,
                f'Method : {clustering_method.upper()}  |  Habitats : {best_k}',
                ha='center', fontsize=14, color='#444444')
        ax.text(0.5, 0.35, f'Tumor voxels : {len(df_scaled)}',
                ha='center', fontsize=14, color='#444444')
        ax.text(0.5, 0.28, f'Parameters : {", ".join(parameters)}',
                ha='center', fontsize=11, color='#666666')
        ax.text(0.5, 0.18, datetime.datetime.now().strftime('%B %d, %Y'),
                ha='center', fontsize=12, color='#888888')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Habitat summary page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.05, 0.95, 'Habitat Summary', fontsize=18,
                fontweight='bold', color='#1F3864', va='top')
        ax.text(0.05, 0.88,
                'Values are robust-scaled (0 = tumor median, unit = IQR)',
                fontsize=10, color='#888888', va='top', style='italic')

        profiles = df_scaled.groupby('Habitat')[parameters].mean().round(2)
        y = 0.80
        for hab_id, label in sorted(habitat_labels.items()):
            n_pix = (df_scaled['Habitat'] == hab_id).sum()
            pct   = n_pix / len(df_scaled) * 100
            ax.text(0.05, y, f'Habitat {hab_id} -- {label}',
                    fontsize=12, fontweight='bold', color='#2E75B6', va='top')
            ax.text(0.05, y - 0.04, f'{n_pix} voxels ({pct:.1f}%)',
                    fontsize=10, color='#444444', va='top')
            if hab_id in profiles.index:
                profile_str = '   '.join([
                    f"{p}: {profiles.loc[hab_id, p]:+.2f}"
                    for p in parameters
                ])
                ax.text(0.05, y - 0.08, profile_str,
                        fontsize=8, color='#666666', va='top',
                        family='monospace')
            y -= 0.18
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # One page per figure
        for fname, title in figures:
            fpath = os.path.join(output_dir, fname)
            if not os.path.exists(fpath):
                print(f"  [!] Figure not found, skipped : {fname}")
                continue
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            img = mpimg.imread(fpath)
            ax.imshow(img, aspect='auto')
            ax.set_title(title, fontsize=13, fontweight='bold',
                         color='#1F3864', pad=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"Report saved : {pdf_path}")
    return pdf_path


def save_habitat_mask(path_ref, df_scaled, df_all, output_dir):
    """
    Load NIfTI reference image and save the habitat mask as NIfTI.
    Returns the path to the saved mask.
    """
    nii_obj     = nib.load(path_ref)
    affine      = nii_obj.affine
    img_ref     = np.nan_to_num(nii_obj.get_fdata(), nan=0.0)
    img_ref     = np.squeeze(img_ref)
    img_display = img_ref[..., 0] if img_ref.ndim == 4 else img_ref

    habitats_mask = np.zeros(img_display.shape, dtype=int)
    xs = df_scaled['X'].values.astype(int) - 1
    ys = df_scaled['Y'].values.astype(int) - 1
    zs = df_scaled['Slice'].values.astype(int) - 1

    nx, ny, nz = img_display.shape
    oob = ((xs < 0) | (xs >= nx) |
           (ys < 0) | (ys >= ny) |
           (zs < 0) | (zs >= nz))
    if oob.any():
        print(f"  [!] {oob.sum()} voxel(s) out of NIfTI bounds "
              f"({nx}×{ny}×{nz}) — skipped")
        xs, ys, zs = xs[~oob], ys[~oob], zs[~oob]
        hab_vals = df_scaled['Habitat'].values.astype(int)[~oob] + 1
    else:
        hab_vals = df_scaled['Habitat'].values.astype(int) + 1

    habitats_mask[xs, ys, zs] = hab_vals

    mask_path = os.path.join(output_dir, 'habitats_mask_V2.nii.gz')
    nib.save(nib.Nifti1Image(habitats_mask.astype(np.int16), affine),
             mask_path)
    print(f"Mask saved : {mask_path}")
    return mask_path
