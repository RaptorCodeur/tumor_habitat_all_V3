# =============================================================================
# synthetic_mice.py — Synthetic tumor MRI data generator
#
# Generates N mice with KNOWN ground-truth habitat labels for pipeline testing.
# Output CSVs are drop-in compatible with the joint clustering pipeline.
#
# Usage (command line):
#   python synthetic_mice.py --preset necrotic_dominant --n-mice 3 --out ./syn_data
#   python synthetic_mice.py --list-presets
#
# Usage (as a module):
#   from synthetic_mice import generate_cohort, PRESETS, MouseConfig, SpatialSeed
#   generate_cohort([PRESETS['three_habitats']('M01'), PRESETS['uniform']('M02')],
#                   output_root='./syn_data')
# =============================================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ─── Biological ground truth: raw MRI values per habitat type ────────────────
# Calibrated from A030122_R71F_10s_HFD and visual inspection of the cohort.
# Each entry: (mean, std).  std encodes intra-habitat biological diversity.
#
# DTI-AD / MD / RD units : µm²/ms × 10⁻⁶  (real values ~500–2000)
# DTI-FA                 : dimensionless (0–1)
# T2star / T2            : ms
# MTR                    : %

HABITAT_BIOLOGY: Dict[str, Dict[str, Tuple[float, float]]] = {
    'necrotic': {
        # Cell membranes broken → free water → high AD/MD; very low FA;
        # blood products → low T2star; late-stage → high T2
        'DTI-AD' : (1650.0, 280.0),
        'DTI-FA' : (0.033,  0.014),
        'DTI-MD' : (1420.0, 210.0),
        'DTI-RD' : (1560.0, 230.0),
        'T2star' : (7.5,    2.8),
        'MTR'    : (19.0,   4.5),
        'T2'     : (68.0,   12.0),
    },
    'hypoxic': {
        # Reduced cellularity, hypoxia, early necrosis precursor;
        # intermediate AD, low FA, moderate T2star
        'DTI-AD' : (920.0,  110.0),
        'DTI-FA' : (0.072,  0.022),
        'DTI-MD' : (830.0,  95.0),
        'DTI-RD' : (870.0,  100.0),
        'T2star' : (11.5,   3.2),
        'MTR'    : (27.0,   5.0),
        'T2'     : (42.0,   8.0),
    },
    'proliferative': {
        # Dense, rapidly dividing cells; restricted diffusion → low AD;
        # low FA (isotropic cells); moderate T2star; low T2
        'DTI-AD' : (640.0,  75.0),
        'DTI-FA' : (0.092,  0.024),
        'DTI-MD' : (575.0,  68.0),
        'DTI-RD' : (605.0,  72.0),
        'T2star' : (14.0,   3.0),
        'MTR'    : (36.0,   5.0),
        'T2'     : (29.0,   5.0),
    },
    'viable': {
        # Well-vascularised viable tumor rim; slightly higher AD than core;
        # moderate FA (some organisation); intermediate T2star
        'DTI-AD' : (730.0,  88.0),
        'DTI-FA' : (0.135,  0.030),
        'DTI-MD' : (665.0,  78.0),
        'DTI-RD' : (695.0,  82.0),
        'T2star' : (18.5,   4.0),
        'MTR'    : (30.0,   5.0),
        'T2'     : (45.0,   8.0),
    },
    'edematous': {
        # Oedema / tumour infiltration into surrounding tissue;
        # higher FA (aligned structures); high T2 and T2star
        'DTI-AD' : (785.0,  92.0),
        'DTI-FA' : (0.225,  0.042),
        'DTI-MD' : (725.0,  82.0),
        'DTI-RD' : (748.0,  86.0),
        'T2star' : (24.5,   5.0),
        'MTR'    : (21.0,   4.0),
        'T2'     : (62.0,   12.0),
    },
    'contralateral': {
        # Healthy contralateral hemisphere; lower AD, higher FA, homogeneous
        'DTI-AD' : (558.0,  28.0),
        'DTI-FA' : (0.225,  0.028),
        'DTI-MD' : (512.0,  26.0),
        'DTI-RD' : (533.0,  27.0),
        'T2star' : (17.2,   1.8),
        'MTR'    : (39.0,   3.0),
        'T2'     : (56.0,   5.5),
    },
}

ALL_PARAMS = ['DTI-AD', 'DTI-FA', 'DTI-MD', 'DTI-RD', 'T2star', 'MTR', 'T2']


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SpatialSeed:
    """One spatial attractor for a habitat type.

    x, y : position in tumor-local coordinates, range [-1, 1].
            (0, 0) = tumor center.  (1, 0) = right edge of bounding radius.
    sigma : Gaussian bandwidth as a fraction of the tumor radius.
            Small sigma = sharp habitat boundary; large sigma = diffuse.
    weight: relative amplitude vs other seeds before softmax normalisation.
            Use > 1 to make this seed dominant, < 1 to make it peripheral.
    """
    habitat_type : str
    x            : float
    y            : float
    sigma        : float
    weight       : float = 1.0


@dataclass
class MouseConfig:
    """Full specification of one synthetic mouse.

    tumor_radius   : radius of the tumor in pixels (~15–32 for real data).
    n_slices       : number of MRI slices (3–6 typical).
    seeds          : spatial layout of habitats (list of SpatialSeed).
    noise_scale    : global multiplicative noise fraction added on top of the
                     per-habitat std.  0.08 ≈ 8 % of the mean per parameter.
    cl_radius      : radius of the contralateral ROI.
    parameters     : MRI parameters to include (subset of ALL_PARAMS).
    """
    name         : str
    seeds        : List[SpatialSeed]
    tumor_radius : float            = 24.0
    n_slices     : int              = 5
    noise_scale  : float            = 0.08
    cl_radius    : float            = 9.0
    parameters   : List[str]        = field(default_factory=lambda: list(ALL_PARAMS))
    grid_size    : int              = 128
    tumor_center : Optional[Tuple[float, float]] = None
    cl_center    : Optional[Tuple[float, float]] = None


# ─── Preset phenotypes ────────────────────────────────────────────────────────

def preset_uniform(name: str = 'SynMouse_uniform', params=None) -> MouseConfig:
    """Single dominant habitat — nearly homogeneous tumour."""
    return MouseConfig(
        name=name, parameters=params or ALL_PARAMS,
        tumor_radius=20, n_slices=3, noise_scale=0.05,
        seeds=[SpatialSeed('proliferative', 0.0, 0.0, 1.2)],
    )


def preset_two_habitats(name: str = 'SynMouse_2H', params=None) -> MouseConfig:
    """Two clearly separated habitats side-by-side."""
    return MouseConfig(
        name=name, parameters=params or ALL_PARAMS,
        tumor_radius=22, n_slices=4,
        seeds=[
            SpatialSeed('necrotic',      -0.42, 0.0, 0.38),
            SpatialSeed('proliferative',  0.42, 0.0, 0.38),
        ],
    )


def preset_three_habitats(name: str = 'SynMouse_3H', params=None) -> MouseConfig:
    """Necrotic core + hypoxic ring + proliferative periphery.

    Necrotic center gets weight=4.0 to compensate for competing against 4
    hypoxic seeds (whose per-type weight is summed), keeping ~15-25 % of
    tumor voxels genuinely necrotic.
    """
    return MouseConfig(
        name=name, parameters=params or ALL_PARAMS,
        tumor_radius=26, n_slices=5,
        seeds=[
            SpatialSeed('necrotic',       0.0,  0.0,  0.28, weight=4.0),
            SpatialSeed('hypoxic',        0.50, 0.0,  0.32),
            SpatialSeed('hypoxic',       -0.50, 0.0,  0.32),
            SpatialSeed('hypoxic',        0.0,  0.50, 0.32),
            SpatialSeed('hypoxic',        0.0, -0.50, 0.32),
            SpatialSeed('proliferative',  0.84, 0.0,  0.20),
            SpatialSeed('proliferative', -0.84, 0.0,  0.20),
            SpatialSeed('proliferative',  0.0,  0.84, 0.20),
            SpatialSeed('proliferative',  0.0, -0.84, 0.20),
        ],
    )


def preset_necrotic_dominant(name: str = 'SynMouse_necro', params=None) -> MouseConfig:
    """Large necrotic core dominating the tumour volume."""
    return MouseConfig(
        name=name, parameters=params or ALL_PARAMS,
        tumor_radius=28, n_slices=5,
        seeds=[
            SpatialSeed('necrotic',       0.0,  0.0, 0.55, weight=2.0),
            SpatialSeed('viable',         0.78, 0.0, 0.24),
            SpatialSeed('viable',        -0.78, 0.0, 0.24),
            SpatialSeed('viable',         0.0,  0.78, 0.24),
            SpatialSeed('viable',         0.0, -0.78, 0.24),
        ],
    )


def preset_split_same_type(name: str = 'SynMouse_split', params=None) -> MouseConfig:
    """Two spatially SEPARATE zones of the SAME habitat type (necrotic).
    Tests whether the pipeline groups them correctly into one habitat class."""
    return MouseConfig(
        name=name, parameters=params or ALL_PARAMS,
        tumor_radius=30, n_slices=5,
        seeds=[
            SpatialSeed('necrotic',       -0.52, 0.0, 0.22),
            SpatialSeed('necrotic',        0.52, 0.0, 0.22),
            SpatialSeed('proliferative',   0.0,  0.0, 0.28),
            SpatialSeed('edematous',       0.0,  0.72, 0.24),
        ],
    )


def preset_four_habitats(name: str = 'SynMouse_4H', params=None) -> MouseConfig:
    """Four biologically distinct habitats in a complex spatial layout."""
    return MouseConfig(
        name=name, parameters=params or ALL_PARAMS,
        tumor_radius=30, n_slices=6,
        seeds=[
            SpatialSeed('necrotic',       0.0,  0.0,  0.24),
            SpatialSeed('hypoxic',        0.50, 0.0,  0.30),
            SpatialSeed('hypoxic',       -0.50, 0.0,  0.30),
            SpatialSeed('viable',         0.0,  0.62, 0.28),
            SpatialSeed('edematous',      0.0, -0.62, 0.28),
        ],
    )


def preset_edematous_heavy(name: str = 'SynMouse_edema', params=None) -> MouseConfig:
    """Oedematous core (~60 %) + proliferative rim (~40 %).
    Calibration: at the core the single edematous seed wins (weight=2, sigma=0.50);
    at the periphery (r > 0.55) each proliferative seed (sigma=0.28, R=0.78) wins.
    """
    return MouseConfig(
        name=name, parameters=params or ALL_PARAMS,
        tumor_radius=22, n_slices=4, noise_scale=0.07,
        seeds=[
            SpatialSeed('edematous',      0.0,  0.0,  0.50, weight=2.0),
            SpatialSeed('proliferative',  0.78, 0.0,  0.28),
            SpatialSeed('proliferative', -0.78, 0.0,  0.28),
            SpatialSeed('proliferative',  0.0,  0.78, 0.28),
            SpatialSeed('proliferative',  0.0, -0.78, 0.28),
        ],
    )


PRESETS: Dict[str, callable] = {
    'uniform'           : preset_uniform,
    'two_habitats'      : preset_two_habitats,
    'three_habitats'    : preset_three_habitats,
    'necrotic_dominant' : preset_necrotic_dominant,
    'split_same_type'   : preset_split_same_type,
    'four_habitats'     : preset_four_habitats,
    'edematous_heavy'   : preset_edematous_heavy,
}


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _ellipse_mask(cx: float, cy: float, rx: float, ry: float,
                  angle: float, grid_size: int) -> np.ndarray:
    """Boolean mask on a (grid_size, grid_size) grid (1-indexed coords)."""
    xs = np.arange(1, grid_size + 1, dtype=np.float64)
    ys = np.arange(1, grid_size + 1, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)  # shape (grid, grid), YY varies along axis-0
    dx = XX - cx
    dy = YY - cy
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx_r =  dx * cos_a + dy * sin_a
    dy_r = -dx * sin_a + dy * cos_a
    return (dx_r / rx) ** 2 + (dy_r / ry) ** 2 <= 1.0


def _mixing_weights(vox_x: np.ndarray, vox_y: np.ndarray,
                    seeds: List[SpatialSeed],
                    cx: float, cy: float, radius: float) -> np.ndarray:
    """Softmax over Gaussian kernels centred at each seed.

    Returns array of shape (n_voxels, n_seeds) summing to 1 along axis-1.
    """
    n_vox  = len(vox_x)
    n_seed = len(seeds)
    log_w  = np.empty((n_vox, n_seed), dtype=np.float64)

    for j, seed in enumerate(seeds):
        sx      = cx + seed.x * radius
        sy      = cy + seed.y * radius
        sigma_px = max(seed.sigma * radius, 1e-3)
        dist_sq  = (vox_x - sx) ** 2 + (vox_y - sy) ** 2
        log_w[:, j] = -0.5 * dist_sq / sigma_px ** 2 + np.log(max(seed.weight, 1e-12))

    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    return w


# ─── Feature sampling ─────────────────────────────────────────────────────────

def _sample_features(habitat_type: str, params: List[str],
                     n: int, noise_scale: float,
                     rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Draw n samples from the multivariate Gaussian of a given habitat.

    noise_scale adds an extra multiplicative jitter on top of the
    per-habitat std to model global image noise / acquisition variability.
    """
    bio = HABITAT_BIOLOGY[habitat_type]
    out = {}
    for p in params:
        mean, std = bio[p]
        total_std = std + noise_scale * abs(mean)
        vals = rng.normal(mean, total_std, n)
        # Enforce non-negative physical constraints
        if p == 'DTI-FA':
            vals = np.clip(vals, 0.0, 1.0)
        else:
            vals = np.clip(vals, 0.001, None)
        out[p] = vals
    return out


# ─── Core generator ───────────────────────────────────────────────────────────

def generate_mouse(config: MouseConfig,
                   rng: np.random.Generator,
                   output_dir: str) -> Dict:
    """Generate synthetic MRI data for one mouse and write CSVs.

    Output files
    ------------
    sub-{name}_{param}_roi_data.csv   — one per MRI parameter (ALL + CL rows)
    ground_truth.csv                  — true habitat label + mixing weights
    config.json                       — generation parameters

    Returns a summary dict with paths and voxel counts.
    """
    os.makedirs(output_dir, exist_ok=True)
    g = config.grid_size

    # ── Place tumor and CL centres ──────────────────────────────────────────
    if config.tumor_center is None:
        cx = g * 0.545 + rng.uniform(-6.0, 6.0)
        cy = g * 0.545 + rng.uniform(-6.0, 6.0)
    else:
        cx, cy = config.tumor_center

    if config.cl_center is None:
        cl_cx = g * 0.33 + rng.uniform(-4.0, 4.0)
        cl_cy = g * 0.53 + rng.uniform(-4.0, 4.0)
    else:
        cl_cx, cl_cy = config.cl_center

    # Slight random ellipticity (eccentricity ≤ 0.18 axis ratio difference)
    ecc   = rng.uniform(0.0, 0.18)
    angle = rng.uniform(0.0, np.pi)
    rx    = config.tumor_radius * (1.0 + ecc)
    ry    = config.tumor_radius * (1.0 - ecc)

    all_tumor_rows : List[Dict] = []   # one entry per voxel
    all_cl_rows    : List[Dict] = []

    # ── Per-slice generation ────────────────────────────────────────────────
    for sl in range(1, config.n_slices + 1):
        # Ellipsoidal cross-section: radius peaks at the middle slice
        t      = (sl - 1) / max(config.n_slices - 1, 1)  # 0 → 1
        scale  = max(np.sin(np.pi * (t * 0.8 + 0.1)), 0.38)
        rx_sl  = rx * scale
        ry_sl  = ry * scale

        # Tumor mask
        mask   = _ellipse_mask(cx, cy, rx_sl, ry_sl, angle, g)
        iy, ix = np.where(mask)
        vox_x  = (ix + 1).astype(np.float64)   # 1-indexed
        vox_y  = (iy + 1).astype(np.float64)
        n_vox  = len(vox_x)
        if n_vox == 0:
            continue

        # Mixing weights for each voxel over each spatial seed
        weights = _mixing_weights(vox_x, vox_y, config.seeds, cx, cy, config.tumor_radius)

        # Aggregate weights per habitat TYPE (sum over seeds of same type)
        habitat_types_list = [s.habitat_type for s in config.seeds]
        unique_htypes = list(dict.fromkeys(habitat_types_list))
        # type_weights[i, j] = total weight for voxel i and unique_htypes[j]
        type_weights = np.zeros((n_vox, len(unique_htypes)), dtype=np.float64)
        for si, ht in enumerate(habitat_types_list):
            ti = unique_htypes.index(ht)
            type_weights[:, ti] += weights[:, si]

        # Dominant habitat type and per-type confidence
        dom_type_idx  = np.argmax(type_weights, axis=1)
        dom_conf      = type_weights[np.arange(n_vox), dom_type_idx]

        # Dominant seed index (used only for feature sampling)
        dom_seed_idx  = np.argmax(weights, axis=1)

        # Pre-sample features grouped by dominant seed (avoids per-voxel loops)
        features = {p: np.empty(n_vox) for p in config.parameters}
        for seed_idx, seed in enumerate(config.seeds):
            mask_s = dom_seed_idx == seed_idx
            n_s    = int(mask_s.sum())
            if n_s == 0:
                continue
            vals = _sample_features(seed.habitat_type, config.parameters,
                                    n_s, config.noise_scale, rng)
            for p in config.parameters:
                features[p][mask_s] = vals[p]

        for vi in range(n_vox):
            dom_ht = unique_htypes[dom_type_idx[vi]]
            row = {
                'Slice'         : sl,
                'X'             : int(vox_x[vi]),
                'Y'             : int(vox_y[vi]),
                'dominant_seed' : int(dom_seed_idx[vi]),
                'habitat_type'  : dom_ht,
                # confidence = total weight of the dominant habitat type
                # (sum over all seeds of that type), not just the single dominant seed
                'confidence'    : float(dom_conf[vi]),
            }
            for si in range(len(config.seeds)):
                row[f'w{si}'] = float(weights[vi, si])
            for p in config.parameters:
                row[p] = float(features[p][vi])
            all_tumor_rows.append(row)

        # CL mask (circle, no eccentricity, CL tissue is more regular)
        cl_r    = config.cl_radius
        cl_mask = _ellipse_mask(cl_cx, cl_cy, cl_r, cl_r, 0.0, g)
        cl_iy, cl_ix = np.where(cl_mask)
        n_cl = len(cl_ix)
        if n_cl > 0:
            cl_vals = _sample_features('contralateral', config.parameters,
                                       n_cl, config.noise_scale * 0.45, rng)
            for vi in range(n_cl):
                row = {'Slice': sl, 'X': int(cl_ix[vi] + 1), 'Y': int(cl_iy[vi] + 1)}
                for p in config.parameters:
                    row[p] = float(cl_vals[p][vi])
                all_cl_rows.append(row)

    if not all_tumor_rows:
        raise RuntimeError(f"generate_mouse: no voxels generated for {config.name!r}. "
                           f"Check tumor_radius vs grid_size.")

    # ── Write one CSV per parameter ─────────────────────────────────────────
    csv_paths: Dict[str, str] = {}
    for p in config.parameters:
        rows = []
        for r in all_tumor_rows:
            rows.append({'Roi_ID': 4, 'Roi_Name': 'ALL',
                         'Slice': r['Slice'], 'X': r['X'], 'Y': r['Y'],
                         'Value': r[p]})
        for r in all_cl_rows:
            rows.append({'Roi_ID': 3, 'Roi_Name': 'CL',
                         'Slice': r['Slice'], 'X': r['X'], 'Y': r['Y'],
                         'Value': r[p]})
        df = pd.DataFrame(rows)[['Roi_ID', 'Roi_Name', 'Slice', 'X', 'Y', 'Value']]
        fname = f"sub-{config.name}_{p}_roi_data.csv"
        fpath = os.path.join(output_dir, fname)
        df.to_csv(fpath, index=False)
        csv_paths[p] = fpath

    # ── Ground truth CSV ────────────────────────────────────────────────────
    gt_cols = ['Slice', 'X', 'Y', 'dominant_seed', 'habitat_type', 'confidence']
    gt_cols += [f'w{si}' for si in range(len(config.seeds))]
    gt_df = pd.DataFrame(all_tumor_rows)[gt_cols]
    gt_path = os.path.join(output_dir, 'ground_truth.csv')
    gt_df.to_csv(gt_path, index=False)

    # ── Config JSON ─────────────────────────────────────────────────────────
    cfg_dict = {
        'name'         : config.name,
        'parameters'   : config.parameters,
        'tumor_radius' : config.tumor_radius,
        'n_slices'     : config.n_slices,
        'noise_scale'  : config.noise_scale,
        'cl_radius'    : config.cl_radius,
        'seeds'        : [
            {'habitat_type': s.habitat_type, 'x': s.x, 'y': s.y,
             'sigma': s.sigma, 'weight': s.weight}
            for s in config.seeds
        ],
        'n_tumor_voxels': len(all_tumor_rows),
        'n_cl_voxels'   : len(all_cl_rows),
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as fh:
        json.dump(cfg_dict, fh, indent=2)

    n_tot = len(all_tumor_rows)
    habitats_present = sorted(set(r['habitat_type'] for r in all_tumor_rows))
    print(f"[{config.name}]  {n_tot:,} tumor voxels | "
          f"{len(all_cl_rows):,} CL voxels | "
          f"{len(config.parameters)} params | "
          f"habitats: {habitats_present}")
    return {'csv_paths': csv_paths, 'gt_path': gt_path, 'n_voxels': n_tot}


def generate_cohort(configs: List[MouseConfig],
                    output_root: str,
                    seed: int = 42) -> List[Dict]:
    """Generate a full cohort of synthetic mice.

    Each mouse gets its own sub-directory under output_root named after
    config.name.  Returns a list of summary dicts (one per mouse).
    """
    rng = np.random.default_rng(seed)
    results = []
    print(f"Generating {len(configs)} synthetic mice → {output_root}")
    for cfg in configs:
        out_dir = os.path.join(output_root, cfg.name)
        results.append(generate_mouse(cfg, rng, out_dir))
    print(f"Done. {sum(r['n_voxels'] for r in results):,} total tumor voxels.")
    return results


# ─── Evaluation against ground truth ─────────────────────────────────────────

def evaluate_clustering(pred_csv: str, gt_csv: str,
                        confidence_threshold: float = 0.0) -> Dict:
    """Compare clustering predictions to synthetic ground truth.

    Parameters
    ----------
    pred_csv : path to habitats_result.csv (output of the pipeline)
               Expected columns: X, Y, Slice, Habitat  (or label column)
    gt_csv   : path to ground_truth.csv generated by this script
    confidence_threshold : exclude voxels whose ground-truth mixing confidence
                           is below this value (they live in transition zones).
                           Set to 0 to include all voxels.

    Returns dict with ARI, AMI, accuracy, n_voxels, n_excluded.
    """
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    pred = pd.read_csv(pred_csv)
    gt   = pd.read_csv(gt_csv)

    # Merge on spatial coordinates
    pred_col = next((c for c in pred.columns
                     if c.lower() in ('habitat', 'label', 'cluster')), None)
    if pred_col is None:
        raise ValueError(f"No habitat/label column found in {pred_csv}. "
                         f"Available: {list(pred.columns)}")

    merged = gt.merge(
        pred[['Slice', 'X', 'Y', pred_col]],
        on=['Slice', 'X', 'Y'], how='inner')

    n_total    = len(merged)
    n_excluded = 0

    if confidence_threshold > 0:
        keep       = merged['confidence'] >= confidence_threshold
        n_excluded = int((~keep).sum())
        merged     = merged[keep]

    if len(merged) == 0:
        return {'error': 'no voxels after confidence filter'}

    y_true = merged['habitat_type'].values
    y_pred = merged[pred_col].values

    ari = float(adjusted_rand_score(y_true, y_pred))
    ami = float(adjusted_mutual_info_score(y_true, y_pred))

    # Simple accuracy: majority vote mapping from pred cluster → true habitat
    from collections import Counter
    pred_labels = np.unique(y_pred)
    mapping = {}
    for pl in pred_labels:
        mask = y_pred == pl
        mapping[pl] = Counter(y_true[mask]).most_common(1)[0][0]
    y_mapped  = np.array([mapping[p] for p in y_pred])
    accuracy  = float((y_mapped == y_true).mean())

    return {
        'ARI'         : ari,
        'AMI'         : ami,
        'accuracy'    : accuracy,
        'n_voxels'    : len(merged),
        'n_excluded'  : n_excluded,
        'mapping'     : mapping,
    }


# ─── Quick visualisation (optional, requires matplotlib) ─────────────────────

def visualize_ground_truth(gt_csv: str, output_png: str = None,
                           slice_idx: int = None) -> None:
    """Plot the spatial layout of true habitats across slices."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not available — skipping visualisation.")
        return

    COLORS = {
        'necrotic'     : '#8B0000',
        'hypoxic'      : '#FF8C00',
        'proliferative': '#1E90FF',
        'viable'       : '#228B22',
        'edematous'    : '#9370DB',
        'contralateral': '#A9A9A9',
    }

    gt = pd.read_csv(gt_csv)
    slices = sorted(gt['Slice'].unique())
    if slice_idx is not None:
        slices = [s for s in slices if s == slice_idx] or slices[:1]

    n_cols = min(len(slices), 4)
    n_rows = (len(slices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = np.array(axes).flatten()

    habitat_types = sorted(gt['habitat_type'].unique())

    for ax_idx, sl in enumerate(slices):
        ax  = axes[ax_idx]
        df_s = gt[gt['Slice'] == sl]

        for ht in habitat_types:
            sub = df_s[df_s['habitat_type'] == ht]
            ax.scatter(sub['X'], sub['Y'], s=6,
                       c=COLORS.get(ht, '#333333'), label=ht, alpha=0.85)

        # Highlight transition voxels (confidence < 0.7)
        trans = df_s[df_s['confidence'] < 0.70]
        if len(trans):
            ax.scatter(trans['X'], trans['Y'], s=10,
                       edgecolors='white', facecolors='none',
                       linewidths=0.5, alpha=0.5)

        ax.set_title(f'Slice {sl}', fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide unused axes
    for ax in axes[len(slices):]:
        ax.axis('off')

    patches = [mpatches.Patch(color=COLORS.get(ht, '#333'), label=ht)
               for ht in habitat_types]
    fig.legend(handles=patches, loc='lower center', ncol=len(habitat_types),
               fontsize=9, frameon=True)
    fig.suptitle(f'Ground-truth habitats — {os.path.dirname(gt_csv)}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if output_png:
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        print(f"Visualisation saved: {output_png}")
    else:
        plt.show()
    plt.close(fig)


# ─── Command-line interface ───────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description='Generate synthetic tumor MRI data for pipeline testing.')
    parser.add_argument('--preset', choices=list(PRESETS.keys()),
                        help='Which preset phenotype to generate.')
    parser.add_argument('--n-mice', type=int, default=1,
                        help='Number of mice to generate with this preset.')
    parser.add_argument('--out', default='./synthetic_data',
                        help='Root output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--list-presets', action='store_true',
                        help='Print available presets and exit.')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate ground_truth.png for each mouse.')
    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable presets:")
        descs = {
            'uniform'           : '1 habitat — nearly homogeneous tumour',
            'two_habitats'      : '2 habitats side-by-side with transitions',
            'three_habitats'    : 'Necrotic core + hypoxic ring + proliferative rim',
            'necrotic_dominant' : 'Large necrotic core + thin viable rim',
            'split_same_type'   : '2 separate zones of same habitat type + 2 others',
            'four_habitats'     : '4 biologically distinct habitats',
            'edematous_heavy'   : 'Oedematous-dominant tumour, minimal necrosis',
        }
        for k, v in descs.items():
            print(f"  {k:<22} {v}")
        return

    if args.preset is None:
        parser.error('--preset is required (or use --list-presets).')

    rng  = np.random.default_rng(args.seed)
    cfgs = []
    for i in range(1, args.n_mice + 1):
        name = f'SynMouse_{i:02d}_{args.preset}'
        cfgs.append(PRESETS[args.preset](name))

    results = []
    for cfg in cfgs:
        out_dir = os.path.join(args.out, cfg.name)
        results.append(generate_mouse(cfg, rng, out_dir))
        if args.visualize:
            visualize_ground_truth(
                results[-1]['gt_path'],
                output_png=os.path.join(out_dir, 'ground_truth.png'))

    print(f"\nDone. Files written to: {args.out}")


if __name__ == '__main__':
    _cli()
