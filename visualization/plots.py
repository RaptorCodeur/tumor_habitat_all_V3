# =================================================================
# visualization/plots.py — All matplotlib profile plots
# =================================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from config import HABITAT_COLORS_HEX, assign_habitat_colors


def plot_correlation(df_scaled, parameters, output_dir=None):
    """
    Spearman correlation heatmap between MRI parameters.
    Informative only — no dimensionality reduction is performed.
    """
    print("\nPlotting Spearman correlation heatmap...")
    corr = df_scaled[parameters].corr(method='spearman')

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Inter-modality Spearman correlations (robust-scaled tumor pixels)')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '1_correlation.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()


def plot_habitat_barchart(df_scaled, parameters, habitat_labels,
                          output_dir=None):
    """
    Grouped bar chart of robust-scaled parameter values per habitat.
    Reference dashed line at 0 = tumor median.
    """
    profiles   = df_scaled.groupby('Habitat')[parameters].mean()
    n_habitats = len(profiles)
    n_params   = len(parameters)
    color_map  = assign_habitat_colors(list(habitat_labels.values()))
    x          = np.arange(n_params)
    width      = 0.8 / n_habitats

    fig, ax = plt.subplots(figsize=(max(12, n_params * 1.8), 6))
    for i, (hab_id, row) in enumerate(profiles.iterrows()):
        label  = habitat_labels.get(int(hab_id), f'Habitat {hab_id}')
        color  = color_map.get(label, '#95A5A6')
        offset = (i - n_habitats / 2 + 0.5) * width
        bars   = ax.bar(x + offset, row.values, width=width * 0.9,
                        label=f'H{hab_id} -- {label}',
                        color=color, edgecolor='black',
                        linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, row.values):
            va   = 'bottom' if val >= 0 else 'top'
            ypos = bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.1
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{val:.1f}', ha='center', va=va,
                    fontsize=7, rotation=90)

    ymax = profiles.values.max()
    ymin = profiles.values.min()
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--',
               label='Tumor median (robust scaling reference)')
    if ymax > 0.5:
        ax.axhspan(0.5, ymax * 1.05, alpha=0.04, color='red')
    if ymin < -0.5:
        ax.axhspan(ymin * 1.05, -0.5, alpha=0.04, color='blue')

    ax.set_xticks(x)
    ax.set_xticklabels(parameters, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Robust-scaled value (0 = tumor median)', fontsize=11)
    ax.set_title('Tumor habitat profiles -- robust-scaled MRI parameters',
                 fontsize=13, pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if ymax > 0.5:
        ax.text(n_params - 0.5, ymax * 0.95, 'above tumor median',
                color='red', fontsize=8, ha='right', alpha=0.6)
    if ymin < -0.5:
        ax.text(n_params - 0.5, ymin * 0.95, 'below tumor median',
                color='blue', fontsize=8, ha='right', alpha=0.6)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3_barchart.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap_profiles(df_scaled, parameters, habitat_labels,
                          best_k, output_dir=None):
    """
    Heatmap of mean robust-scaled values per habitat and parameter.
    Diverging colormap centered on 0 = tumor median.
    """
    profiles = df_scaled.groupby('Habitat')[parameters].mean().round(3)
    profiles.index = [
        f"H{i} -- {habitat_labels.get(int(i), '?')}"
        for i in profiles.index
    ]
    plt.figure(figsize=(10, max(3, best_k)))
    sns.heatmap(profiles, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, linewidths=0.5)
    plt.title('Tumor habitat profiles (robust-scaled per modality)')
    plt.xlabel('MRI modality')
    plt.ylabel('Habitat')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '4_heatmap.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()


def plot_radar_profiles(df_scaled, parameters, habitat_labels,
                        output_dir=None):
    """
    Radar (spider) chart: each habitat is a closed polygon.
    Dashed circle at 0 = tumor median for reference.
    """
    profiles  = df_scaled.groupby('Habitat')[parameters].mean()
    n_params  = len(parameters)
    color_map = assign_habitat_colors(list(habitat_labels.values()))
    angles    = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles   += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for hab_id, row in profiles.iterrows():
        values  = row[parameters].tolist()
        values += values[:1]
        label   = habitat_labels.get(int(hab_id), f'Habitat {hab_id}')
        color   = color_map.get(label, '#95A5A6')
        ax.plot(angles, values, color=color, linewidth=2,
                label=f'H{hab_id} -- {label}')
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.plot(angles, [0] * len(angles), color='black',
            linewidth=1, linestyle='--', alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(parameters, fontsize=10)
    ax.set_rlabel_position(30)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.set_title('Tumor habitat profiles -- radar chart\n'
                 '(0 = tumor median, unit = IQR)',
                 fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15),
              fontsize=9, framealpha=0.9)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '5_radar_profiles.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()


def plot_gmm_probabilities(labels, df_scaled, proba, colors, best_k,
                            output_dir=None):
    """
    Two-panel figure for GMM assignment confidence.
    Left : habitat map (cluster color per voxel).
    Right: probability map — P(voxel → assigned cluster).
           Low values flag ambiguous boundary voxels.
    Safe to call from a secondary thread (Agg backend).
    """
    slice_id    = df_scaled['Slice'].mode()[0]
    slice_mask  = (df_scaled['Slice'] == slice_id).values
    df_slice    = df_scaled[df_scaled['Slice'] == slice_id].copy()
    labels_sl   = labels[slice_mask]
    proba_sl    = proba[slice_mask]

    max_proba = proba_sl[np.arange(len(labels_sl)), labels_sl]

    x_vals = sorted(df_slice['X'].unique())
    y_vals = sorted(df_slice['Y'].unique())
    x_idx  = {x: i for i, x in enumerate(x_vals)}
    y_idx  = {y: i for i, y in enumerate(y_vals)}

    rows = df_slice['Y'].map(y_idx).values
    cols = df_slice['X'].map(x_idx).values

    grid_label = np.full((len(y_vals), len(x_vals)), -1, dtype=int)
    grid_proba = np.full((len(y_vals), len(x_vals)), np.nan)
    grid_label[rows, cols] = labels_sl
    grid_proba[rows, cols] = max_proba

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cmap = ListedColormap(colors)
    axes[0].imshow(np.ma.masked_where(grid_label == -1, grid_label),
                   cmap=cmap, vmin=0, vmax=best_k - 1,
                   interpolation='nearest', origin='upper')
    axes[0].set_title(f'Habitat map — slice {slice_id}', fontsize=11)
    axes[0].axis('off')

    im = axes[1].imshow(np.ma.masked_invalid(grid_proba),
                        cmap='RdYlGn', vmin=0, vmax=1,
                        interpolation='nearest', origin='upper')
    axes[1].set_title('Assignment probability (GMM)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label='P(voxel → assigned cluster)')

    plt.suptitle('GMM — cluster assignment confidence', fontsize=13)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '6_gmm_probabilities.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()
