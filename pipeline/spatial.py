# =================================================================
# pipeline/spatial.py — Spatial feature enrichment and analysis (V5)
#
# Step 3 : prepare_features()
#   Quick k=2 GMM → necrosis guard → d_topo distance feature
#
# Step 4 : analyze_spatial_coherence()
#   Connected-component analysis per cluster, ambiguity map from GMM proba
#
# Step 5 : mrf_boundary_smooth()   [option — MRF_ENABLED in config]
#   Light ICM correction for uncertain / boundary voxels only
#
# Step 6 : hierarchical_segment()  [option — HIERARCHICAL_ENABLED in config]
#   k=2 core/periphery split, then sub-segment each zone independently
# =================================================================

import os
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as ndi_label

from config import (
    NECROSIS_T2_MIN, NECROSIS_MD_MIN, DTOPO_CORE_SEP_MIN, DTOPO_FEATURE_WEIGHT,
    MIN_COMPONENT_SIZE, AMBIGUITY_PROBA_THR,
    MRF_ENABLED, MRF_LAMBDA, MRF_CONFIDENCE_THR, MRF_N_ITER, TRANSITION_PENALTY,
    HIERARCHICAL_ENABLED, HIER_FEATURE_VARIANCE_THR,
    HIER_CORE_K_MAX, HIER_PERIPH_K_MAX,
    N_REFS_GAP, N_JOBS_GAP, GMM_SILHOUETTE_GUARD, KMEANS_GAP_GUARD, RANDOM_SEED,
)


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _virtual_md(centroid):
    """Return the best available diffusivity value for a centroid dict."""
    if 'DTI-MD' in centroid:
        return centroid['DTI-MD']
    if 'DTI-AD' in centroid and 'DTI-RD' in centroid:
        return (centroid['DTI-AD'] + 2.0 * centroid['DTI-RD']) / 3.0
    return None


def _habitat_family(name):
    """Map any habitat label (including sub-types) to its biological family."""
    families = [
        'Necrosis', 'Hypoxic/vascular', 'Cellular/proliferative',
        'Infiltrative', 'Bulk tumor',
    ]
    low = name.strip().lower()
    for fam in families:
        if low.startswith(fam.lower()):
            return fam
    return 'Bulk tumor'


def _hier_l1_guard(centroids2, parameters):
    """
    Identify core vs periphery from a 2-cluster k=2 GMM.

    Biological model (based on tumour progression hypothesis):
      Core zone  = necrosis + hypoxia signature:
        - T2  HIGH  : free water from cell lysis / liquefactive necrosis
        - MD  HIGH  : unrestricted diffusion after structural breakdown
        - T2* LOW   : deoxyHb accumulation in hypoxic/haemorrhagic zones
                      (negative in IQR space → subtracting flips sign)
        - MTR LOW   : macromolecular content lost in necrosis
                      (negative in IQR space → subtracting flips sign)
      Periphery = viable tumour: moderate T2/MD, preserved T2*, higher MTR.

    FA is deliberately NOT used as a primary criterion: it can be low in
    both necrotic cores AND dense cellular zones, making it ambiguous for
    this specific split.

    Composite core score = T2 + MD − T2* − MTR
    Uses only the parameters that are actually present in the clustering.

    Validity: max inter-cluster Δ across shared parameters > 0.20 IQR.

    Returns (valid: bool, core_id: int, periph_id: int, criterion: str).
    """
    ids = list(centroids2.keys())
    c0, c1 = ids[0], ids[1]

    # ── Separation validity ──────────────────────────────────────────
    shared = [p for p in parameters if p in centroids2[c0] and p in centroids2[c1]]
    if not shared:
        return False, None, None, "no shared parameters"
    max_delta = max(abs(centroids2[c0][p] - centroids2[c1][p]) for p in shared)
    if max_delta < 0.20:
        return False, None, None, f"clusters too similar (max d={max_delta:.2f} < 0.20)"

    # ── Core composite score ─────────────────────────────────────────
    # T2 and MD are the PRIMARY necrosis markers (free water, cell lysis).
    # T2* and MTR are secondary; weighted ×2 for primary, ×1 for secondary
    # so that a cluster with only primary markers is still correctly identified.
    def _core_score(cen):
        t2  = cen.get('T2',     0.0)
        md  = _virtual_md(cen) or 0.0
        t2s = cen.get('T2star', 0.0)   # negative in IQR space when hypoxic
        mtr = cen.get('MTR',    0.0)   # negative in IQR space in necrosis
        return 2.0*t2 + 2.0*md - t2s - mtr

    s0, s1    = _core_score(centroids2[c0]), _core_score(centroids2[c1])
    core_id   = c0 if s0 > s1 else c1
    periph_id = c1 if s0 > s1 else c0

    # Build a human-readable description of the strongest contributors
    cen_core  = centroids2[core_id]
    cen_peri  = centroids2[periph_id]
    contribs  = []
    for p, sign in [('T2', +1), ('DTI-MD', +1), ('DTI-AD', +1), ('DTI-RD', +1),
                    ('T2star', -1), ('MTR', -1)]:
        if p in shared:
            delta = sign * (cen_core.get(p, 0.0) - cen_peri.get(p, 0.0))
            if abs(delta) > 0.15:
                direction = ('^' if cen_core.get(p, 0.0) > cen_peri.get(p, 0.0)
                             else 'v')
                contribs.append(f"{p}{direction}(d{delta:+.2f})")
    crit = ("core=" + ", ".join(contribs[:4]) if contribs else "T2+MD-T2*-MTR")
    crit += f"  [score: core={max(s0,s1):.2f} vs periph={min(s0,s1):.2f}]"

    print(f"  [L1 guard] Core={core_id}, Periph={periph_id}")
    print(f"  [L1 guard] {crit}")
    return True, core_id, periph_id, crit


def compute_zone_dtopo(df_all, labels, core_cluster_ids):
    """
    Compute per-voxel 2-D distance-transform from the core zone.

    Used after hierarchical segmentation to populate d_topo_norm in centroids
    so the spatial guard for Infiltrative labelling remains active even when
    no necrosis was detected.

    Parameters
    ----------
    df_all           : DataFrame with X, Y, Slice columns
    labels           : int ndarray (n_voxels,) — final hierarchical cluster IDs
    core_cluster_ids : list of int — cluster IDs belonging to the core zone

    Returns
    -------
    d_topo : float32 ndarray (n_voxels,) normalised to [0, 1].
             0 = inside / adjacent to core zone, 1 = farthest periphery.
    """
    zone_labels = np.where(np.isin(labels, core_cluster_ids), 0, 1).astype(int)
    return _compute_dtopo(df_all, zone_labels, 0)


def compute_necrotic_dtopo(df_all, labels, hier_meta):
    """
    Compute per-voxel d_topo seeded from the necrotic sub-cluster identified
    during Level 2a of hierarchical_segment.

    When the necrosis guard succeeded inside hierarchical_segment, the seed is
    the single necrotic sub-cluster → Hypoxic clusters (adjacent ring) receive a
    small but non-zero d_topo, distinguishing them from true necrosis (d_topo=0).

    Falls back to compute_zone_dtopo (all core clusters as seed) when the guard
    failed, preserving the previous behaviour.

    Returns
    -------
    d_topo : float32 ndarray (n_voxels,)
    necrotic_cluster_id : int or None
    """
    nec_id = hier_meta.get('necrotic_cluster_id')
    if nec_id is not None:
        return _compute_dtopo(df_all, labels, nec_id), nec_id
    return compute_zone_dtopo(df_all, labels, hier_meta['core_cluster_ids']), None


def _necrosis_guard(centroids, parameters):
    """
    Check whether a clear necrotic cluster exists.

    Strategy: score each cluster by T2 + MD; verify the top cluster meets
    absolute thresholds AND is well-separated from the next cluster.

    Returns (passes: bool, necrotic_cluster_id: int or None).
    """
    scores = {}
    for c, cen in centroids.items():
        t2 = cen.get('T2', 0.0)
        md = _virtual_md(cen) or 0.0
        scores[c] = t2 + md

    best_c = max(scores, key=scores.__getitem__)
    cen_best = centroids[best_c]

    t2_val = cen_best.get('T2', 0.0)
    md_val = _virtual_md(cen_best) or 0.0

    if t2_val < NECROSIS_T2_MIN or md_val < NECROSIS_MD_MIN:
        print(f"  [d_topo guard] No necrosis (T2={t2_val:.2f}, MD={md_val:.2f}); "
              "d_topo skipped")
        return False, None

    if len(centroids) > 1:
        second_best = max(scores[c] for c in centroids if c != best_c)
        sep = scores[best_c] - second_best
        if sep < DTOPO_CORE_SEP_MIN:
            print(f"  [d_topo guard] Separation too low ({sep:.2f} < "
                  f"{DTOPO_CORE_SEP_MIN}); diffuse tumour — d_topo skipped")
            return False, None

    print(f"  [d_topo guard] Necrosis detected: cluster {best_c} "
          f"(T2={t2_val:.2f}, MD={md_val:.2f})")
    return True, best_c


def _compute_dtopo(df_all, labels, necrotic_cluster_id):
    """
    Per-slice 2-D distance transform from the necrotic seed mask.

    Returns a float32 array of shape (n_voxels,) normalised to [0, 1].
    Voxels in slices with no necrotic seed receive NaN (unknown distance).

    NaN is used instead of a placeholder value (previously 1.0) to prevent
    seedless apical/caudal slices from falsely appearing as "maximum periphery"
    and incorrectly allowing Infiltrative labelling in central tumour voxels.
    """
    n = len(df_all)
    dtopo = np.full(n, np.nan, dtype=np.float32)

    slices = df_all['Slice'].values
    x_vals = df_all['X'].values.astype(int)
    y_vals = df_all['Y'].values.astype(int)

    for sl in np.unique(slices):
        sl_mask = slices == sl
        idx     = np.where(sl_mask)[0]
        xs = x_vals[idx]
        ys = y_vals[idx]
        lbls_sl = labels[idx]

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cols = xs - x_min
        rows = ys - y_min
        H = y_max - y_min + 1
        W = x_max - x_min + 1

        seed_grid = np.zeros((H, W), dtype=bool)
        seed_grid[rows[lbls_sl == necrotic_cluster_id],
                  cols[lbls_sl == necrotic_cluster_id]] = True

        if not seed_grid.any():
            continue   # leave NaN for this slice

        dist = distance_transform_edt(~seed_grid).astype(np.float32)
        dtopo[idx] = dist[rows, cols]

    max_d = float(np.nanmax(dtopo)) if not np.all(np.isnan(dtopo)) else 0.0
    if max_d > 0:
        dtopo /= max_d
    return dtopo


def _build_voxel_neighbors(df_all):
    """
    Build 4-connectivity (in-slice) neighbor index for each voxel.
    Returns a list of lists: neighbors[i] = [j, ...] for each voxel i.
    """
    slices = df_all['Slice'].values
    xs     = df_all['X'].values.astype(int)
    ys     = df_all['Y'].values.astype(int)
    n      = len(df_all)

    coord_to_idx = {(slices[i], xs[i], ys[i]): i for i in range(n)}

    neighbors = [[] for _ in range(n)]
    for i in range(n):
        s, x, y = slices[i], xs[i], ys[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = coord_to_idx.get((s, x + dx, y + dy))
            if j is not None:
                neighbors[i].append(j)
    return neighbors


# ──────────────────────────────────────────────────────────────────
# Step 3 — d_topo feature preparation
# ──────────────────────────────────────────────────────────────────

def prepare_features(features_scaled, df_all, parameters, n_init_gmm=20):
    """
    Run a quick k=2 GMM for the necrosis guard.
    If necrosis is clearly detected, compute d_topo and append it as an
    additional feature column (weighted by DTOPO_FEATURE_WEIGHT).

    Parameters
    ----------
    features_scaled : ndarray (n_voxels, n_params)
    df_all          : DataFrame with X, Y, Slice columns
    parameters      : list of parameter names (same order as features_scaled columns)
    n_init_gmm      : GMM restarts for the guard run

    Returns
    -------
    features_out : ndarray — augmented (or original if guard fails)
    meta         : dict with keys has_necrosis, necrotic_cluster, d_topo (or None)
    """
    from sklearn.mixture import GaussianMixture

    print("\n-- Spatial feature prep (d_topo guard) --")

    gmm2 = GaussianMixture(
        n_components=2, init_params='kmeans',
        random_state=RANDOM_SEED, n_init=n_init_gmm,
        covariance_type='full', max_iter=200,
    )
    labels2 = gmm2.fit_predict(features_scaled)

    centroids2 = {
        c: {p: float(features_scaled[labels2 == c, i].mean())
            for i, p in enumerate(parameters)}
        for c in range(2)
    }

    passes, nec_id = _necrosis_guard(centroids2, parameters)

    if not passes:
        # Fallback: no confirmed necrosis, but we still compute a proxy d_topo
        # from the most T2+MD-elevated cluster so the spatial guard for Infiltrative
        # remains active.  Without this, every cluster with FA>0.1 and T2>0 would
        # qualify as Infiltrative regardless of position inside the tumour.
        proxy_scores = {c: (centroids2[c].get('T2', 0.0) +
                            (_virtual_md(centroids2[c]) or 0.0))
                        for c in centroids2}
        proxy_c = max(proxy_scores, key=proxy_scores.__getitem__)
        d_topo_proxy = _compute_dtopo(df_all, labels2, proxy_c)
        proxy_t2 = centroids2[proxy_c].get('T2', 0.0)
        proxy_md = _virtual_md(centroids2[proxy_c]) or 0.0
        print(f"  [d_topo proxy] cluster {proxy_c} as core proxy "
              f"(T2={proxy_t2:.2f}, MD={proxy_md:.2f}) — spatial guard active")
        return features_scaled, {
            'has_necrosis': False,
            'necrotic_cluster': None,
            'd_topo': d_topo_proxy,
        }

    d_topo = _compute_dtopo(df_all, labels2, nec_id)
    # Replace NaN (seedless slices) with the global mean d_topo so that voxels
    # in slices without a necrotic seed are placed at the average distance from
    # the core — not at 0.0 (= "adjacent to core") which was the previous wrong default.
    _fill = float(np.nanmean(d_topo)) if not np.all(np.isnan(d_topo)) else 0.5
    d_topo_feat = np.where(np.isnan(d_topo), _fill, d_topo)
    features_out = np.column_stack([
        features_scaled,
        d_topo_feat.reshape(-1, 1) * DTOPO_FEATURE_WEIGHT,
    ])
    print(f"  d_topo appended (weight={DTOPO_FEATURE_WEIGHT}); "
          f"feature matrix: {features_out.shape}")

    return features_out, {
        'has_necrosis'    : True,
        'necrotic_cluster': nec_id,
        'd_topo'          : d_topo,
    }


def inject_dtopo_centroids(centroids, labels, dtopo_meta):
    """
    Add d_topo_norm (mean normalised d_topo per cluster) to each centroid dict.

    d_topo from _compute_dtopo is already normalised to [0, 1] with NaN for
    slices that have no seed voxel.  We average only the non-NaN values per
    cluster so that seedless apical/caudal voxels do not inflate the mean.

    No-op when dtopo_meta is None or d_topo is absent.
    """
    if not dtopo_meta:
        return centroids
    d_topo = dtopo_meta.get('d_topo')
    if d_topo is None:
        return centroids
    for c in centroids:
        mask  = (labels == c)
        valid = d_topo[mask]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            centroids[c]['d_topo_norm'] = float(valid.mean())
    return centroids


# ──────────────────────────────────────────────────────────────────
# Step 4 — Spatial coherence analysis
# ──────────────────────────────────────────────────────────────────

def analyze_spatial_coherence(labels, df_all, proba, best_k, out_dir=None):
    """
    Per-cluster, per-slice connected-component analysis and ambiguity mapping.

    Parameters
    ----------
    labels   : int ndarray (n_voxels,)
    df_all   : DataFrame with X, Y, Slice
    proba    : float ndarray (n_voxels, k) or None
    best_k   : number of clusters
    out_dir  : directory for CSV report (None = skip file output)

    Returns
    -------
    confidence_arr : float32 (n_voxels,) — P(assigned label), 0 if no proba
    ambiguous_arr  : bool   (n_voxels,)  — True when 2nd-best P > threshold
    report         : dict   {cluster_id: {'n_components', 'micro_components',
                                          'micro_voxels', 'max_size'}}
    """
    print("\n-- Spatial coherence analysis --")

    slices  = df_all['Slice'].values
    x_vals  = df_all['X'].values.astype(int)
    y_vals  = df_all['Y'].values.astype(int)
    n       = len(labels)

    # ── Confidence and ambiguity from GMM proba ──────────────────
    if proba is not None:
        sorted_p        = np.sort(proba, axis=1)[:, ::-1]
        confidence_arr  = sorted_p[:, 0].astype(np.float32)
        ambiguous_arr   = sorted_p[:, 1] > AMBIGUITY_PROBA_THR
    else:
        confidence_arr = np.zeros(n, dtype=np.float32)
        ambiguous_arr  = np.zeros(n, dtype=bool)

    # ── Connected-component analysis ─────────────────────────────
    report      = {}
    report_rows = []

    for sl in np.unique(slices):
        sl_mask = slices == sl
        idx     = np.where(sl_mask)[0]
        xs = x_vals[idx]
        ys = y_vals[idx]

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cols = xs - x_min
        rows = ys - y_min
        H    = y_max - y_min + 1
        W    = x_max - x_min + 1
        lbls_sl = labels[idx]

        for c in range(best_k):
            grid = np.zeros((H, W), dtype=bool)
            c_mask = lbls_sl == c
            if not c_mask.any():
                continue
            grid[rows[c_mask], cols[c_mask]] = True

            labeled_grid, n_comp = ndi_label(grid)

            sizes        = []
            micro_count  = 0
            micro_voxels = 0
            for comp in range(1, n_comp + 1):
                sz = int((labeled_grid == comp).sum())
                sizes.append(sz)
                if sz < MIN_COMPONENT_SIZE:
                    micro_count  += 1
                    micro_voxels += sz

            max_sz = max(sizes) if sizes else 0

            report_rows.append({
                'cluster'          : c,
                'slice'            : sl,
                'n_components'     : n_comp,
                'max_component_size': max_sz,
                'micro_components' : micro_count,
                'micro_voxels'     : micro_voxels,
            })

            if c not in report:
                report[c] = {'n_components': 0, 'micro_components': 0,
                             'micro_voxels': 0, 'max_size': 0}
            report[c]['n_components']    += n_comp
            report[c]['micro_components'] += micro_count
            report[c]['micro_voxels']    += micro_voxels
            report[c]['max_size']         = max(report[c]['max_size'], max_sz)

    # ── Print summary ─────────────────────────────────────────────
    print(f"  {'Cluster':<10} {'Components':>12} {'Micro-comp':>12} "
          f"{'Micro-vox':>12} {'Max size':>10}")
    print("  " + "-" * 58)
    for c in range(best_k):
        r = report.get(c, {})
        micro_warn = "  [!]" if r.get('micro_components', 0) > 0 else ""
        print(f"  {c:<10} {r.get('n_components',0):>12} "
              f"{r.get('micro_components',0):>12} "
              f"{r.get('micro_voxels',0):>12} "
              f"{r.get('max_size',0):>10}{micro_warn}")

    if proba is not None:
        n_amb = int(ambiguous_arr.sum())
        pct   = 100 * n_amb / n
        print(f"  Ambiguous voxels (2nd-best P > {AMBIGUITY_PROBA_THR}): "
              f"{n_amb} ({pct:.1f}%)")

    # ── Save CSV ──────────────────────────────────────────────────
    if out_dir and report_rows:
        path = os.path.join(out_dir, 'spatial_coherence.csv')
        pd.DataFrame(report_rows).to_csv(path, index=False)
        print(f"  Spatial coherence report saved: {path}")

    return confidence_arr, ambiguous_arr, report


# ──────────────────────────────────────────────────────────────────
# Step 5 — MRF light boundary correction (option)
# ──────────────────────────────────────────────────────────────────

def mrf_boundary_smooth(labels, proba, df_all, ws_results):
    """
    Iterated Conditional Modes (ICM) correction for uncertain / boundary voxels.

    Only voxels satisfying at least one condition are updated:
      - P(assigned_label) < MRF_CONFIDENCE_THR, OR
      - at least one spatial neighbor has a different label.

    High-confidence interior voxels are never modified.

    Parameters
    ----------
    labels     : int ndarray (n_voxels,)
    proba      : float ndarray (n_voxels, k)
    df_all     : DataFrame with X, Y, Slice
    ws_results : {cluster_id: {'label': str, ...}} from label_clusters_ws

    Returns
    -------
    labels_out : int ndarray (n_voxels,) — corrected labels
    """
    print(f"\n-- MRF light correction (lambda={MRF_LAMBDA}, "
          f"conf_thr={MRF_CONFIDENCE_THR}, iter={MRF_N_ITER}) --")

    k          = proba.shape[1]
    labels_out = labels.copy()
    neighbors  = _build_voxel_neighbors(df_all)

    id_to_family = {c: _habitat_family(ws_results[c]['label'])
                    for c in ws_results}

    for it in range(MRF_N_ITER):
        n_changed = 0

        for i in range(len(labels_out)):
            assigned = labels_out[i]
            nbrs     = neighbors[i]
            if not nbrs:
                continue

            p_assigned     = proba[i, assigned]
            is_uncertain   = p_assigned < MRF_CONFIDENCE_THR
            is_boundary    = any(labels_out[j] != assigned for j in nbrs)

            if not is_uncertain and not is_boundary:
                continue

            nbr_labels = [labels_out[j] for j in nbrs]

            best_label  = assigned
            best_energy = float('inf')

            for k_cand in range(k):
                e_unary = -np.log(max(proba[i, k_cand], 1e-10))
                fam_cand = id_to_family.get(k_cand, 'Bulk tumor')
                e_pair = sum(
                    TRANSITION_PENALTY.get(fam_cand, {}).get(
                        id_to_family.get(nl, 'Bulk tumor'), 0.0)
                    for nl in nbr_labels
                )
                energy = e_unary + MRF_LAMBDA * e_pair
                if energy < best_energy:
                    best_energy = energy
                    best_label  = k_cand

            if best_label != assigned:
                labels_out[i] = best_label
                n_changed += 1

        print(f"  Iteration {it + 1}/{MRF_N_ITER}: {n_changed} voxels updated")
        if n_changed == 0:
            break

    return labels_out


# ──────────────────────────────────────────────────────────────────
# Step 6 — Hierarchical k=2 segmentation (option)
# ──────────────────────────────────────────────────────────────────

def _select_features_by_variance(features, feature_names, thr=HIER_FEATURE_VARIANCE_THR):
    """
    Select parameters whose intra-zone variance >= thr × max_variance.
    Near-constant parameters within a zone carry no discriminating information
    and add noise to sub-clustering.
    Always returns at least 2 features (the two most variable ones).
    """
    variances = np.var(features, axis=0)
    max_var   = variances.max() if variances.max() > 1e-10 else 1.0
    sel = [i for i, v in enumerate(variances) if v / max_var >= thr]
    if len(sel) < 2:
        sel = sorted(range(len(variances)), key=lambda i: -variances[i])[:2]
    return sel, [feature_names[i] for i in sel]


def _select_k_core(features, k_max, method, n_init, df_sub=None,
                   n_refs=N_REFS_GAP, n_jobs=N_JOBS_GAP,
                   gmm_silhouette_guard=GMM_SILHOUETTE_GUARD,
                   kmeans_gap_guard=KMEANS_GAP_GUARD,
                   spatial_weight=1.0):
    """
    Select optimal k for core sub-segmentation (k = 1 … k_max).

    - gmm    : GMM BIC comparison (k=1 included; BIC naturally penalises
               over-fitting so k=1 wins when the core is homogeneous).
    - kmeans / srsc : Gap + Silhouette via select_optimal_k, matching the
               criterion used for the periphery and the main pipeline.
               k_range starts at 2 because Gap statistic is undefined at k=1.
    """
    if method == 'gmm':
        from sklearn.mixture import GaussianMixture
        k_max_safe = min(k_max, max(1, len(features) // 5))
        bics = {}
        for k in range(1, k_max_safe + 1):
            try:
                gmm = GaussianMixture(n_components=k, random_state=RANDOM_SEED,
                                      n_init=n_init, covariance_type='full',
                                      max_iter=200, init_params='kmeans')
                gmm.fit(features)
                bics[k] = gmm.bic(features)
                print(f"    Core BIC k={k}: {bics[k]:.1f}")
            except Exception as exc:
                print(f"    Core BIC k={k}: failed ({exc})")
        return min(bics, key=bics.get) if bics else 1

    else:  # kmeans or srsc — Gap + Silhouette
        if len(features) < 10:
            print(f"    Core: trop peu de voxels ({len(features)}) → k=1")
            return 1
        k_max_safe = min(k_max, max(2, len(features) // 5))
        k_range = range(2, k_max_safe + 1)
        print(f"    Core — Gap+Silhouette k in {list(k_range)} "
              f"(method={method.upper()})...")
        best_k, _ = select_optimal_k(
            features,
            method=method,
            df_all=df_sub,
            k_range=k_range,
            n_refs=n_refs,
            n_jobs=n_jobs,
            spatial_weight=spatial_weight,
            n_init_gmm=n_init,
            gmm_silhouette_guard=gmm_silhouette_guard,
            kmeans_gap_guard=kmeans_gap_guard,
            output_dir=None,
        )
        return best_k


def _sub_segment(feats, df_sub, k, method, n_init_gmm, spatial_weight, tag):
    """
    Run sub-segmentation with the user-selected method.

    Level 1 (guard) always uses GMM; this helper is for levels 2a and 2b only.
    Returns (labels_int_array, proba_or_None).
    """
    if len(feats) < k * 5:
        print(f"  [!] Sub-seg {tag}: too few voxels ({len(feats)}) for k={k} "
              "→ single cluster")
        return np.zeros(len(feats), dtype=int), None
    try:
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            km  = KMeans(n_clusters=k, n_init=n_init_gmm,
                         random_state=RANDOM_SEED, max_iter=300)
            lbl = km.fit_predict(feats)
            return lbl, None

        elif method == 'srsc':
            from pipeline.clustering import run_clustering
            lbl, ex = run_clustering(
                feats, df_sub.reset_index(drop=True), k,
                method='srsc', spatial_weight=spatial_weight,
                n_init_gmm=n_init_gmm)
            return lbl, ex.get('proba')

        else:   # gmm (default and fallback)
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=k, init_params='kmeans',
                                   random_state=RANDOM_SEED, n_init=n_init_gmm,
                                   covariance_type='full', max_iter=200)
            lbl  = gmm.fit_predict(feats)
            prob = gmm.predict_proba(feats)
            return lbl, prob

    except Exception as exc:
        print(f"  [!] Sub-seg {tag} k={k} failed ({exc}) → single cluster")
        return np.zeros(len(feats), dtype=int), None


def hierarchical_segment(features_scaled, df_all, parameters, n_init_gmm=20,
                          method='gmm', spatial_weight=1.0,
                          n_refs=N_REFS_GAP, n_jobs=N_JOBS_GAP,
                          gmm_silhouette_guard=GMM_SILHOUETTE_GUARD,
                          kmeans_gap_guard=KMEANS_GAP_GUARD):
    """
    Two-level segmentation:
      Level 1  — k=2 GMM (always GMM: guard requires soft probabilities for
                 _necrosis_guard's centroid scoring)
      Level 2a — sub-segment core   (k=2 fixed, user method, discriminating features)
      Level 2b — sub-segment periph (k chosen by same criterion as main pipeline —
                 BIC for GMM, Gap+Silhouette for kmeans/SRSC — on selected features)

    Feature selection per sub-zone:
      Parameters whose intra-zone variance < HIER_FEATURE_VARIANCE_THR × max_variance
      are dropped.  They are near-constant within the zone and only add noise.
      At least 2 features are always kept (the two most variable ones).

    Guard: if k=2 does not yield a clear necrotic core, returns
    (None, {hierarchical_valid: False}) so the caller uses the standard pipeline.

    Returns
    -------
    labels_hier : int ndarray  — global cluster labels (0 … best_k-1)
    meta        : dict with keys:
        hierarchical_valid, best_k,
        core_cluster_ids, periphery_cluster_ids, core_id_k2,
        core_features_used, periphery_features_used
    """
    from sklearn.mixture import GaussianMixture
    from pipeline.selection import select_optimal_k

    print(f"\n-- Hierarchical segmentation  "
          f"(L1: GMM k=2 guard | L2: {method.upper()}) --")

    # ── Level 1: k=2 GMM — necrosis guard ───────────────────────
    gmm1 = GaussianMixture(n_components=1, init_params='kmeans',
                            random_state=RANDOM_SEED, n_init=1,
                            covariance_type='full', max_iter=200)
    gmm1.fit(features_scaled)
    bic1 = gmm1.bic(features_scaled)

    gmm2    = GaussianMixture(n_components=2, init_params='kmeans',
                               random_state=RANDOM_SEED, n_init=n_init_gmm,
                               covariance_type='full', max_iter=200)
    labels2 = gmm2.fit_predict(features_scaled)
    bic2    = gmm2.bic(features_scaled)

    print(f"  [L1 BIC] k=1: {bic1:.1f}  k=2: {bic2:.1f}")
    if bic1 <= bic2:
        print(f"  [L1 BIC] k=1 preferred — tumour appears homogeneous; fallback")
        return None, {'hierarchical_valid': False}

    centroids2 = {c: {p: float(features_scaled[labels2 == c, i].mean())
                      for i, p in enumerate(parameters)}
                  for c in range(2)}

    valid, core_id, periph_id, l1_crit = _hier_l1_guard(centroids2, parameters)
    if not valid:
        print(f"  [L1 guard] Failed ({l1_crit}) — fallback to normal pipeline")
        return None, {'hierarchical_valid': False}
    core_mask   = labels2 == core_id
    periph_mask = labels2 == periph_id

    # ── Level 2a: core sub-segmentation (k selected by GMM BIC) ────
    core_feats            = features_scaled[core_mask]
    df_core               = df_all[core_mask].reset_index(drop=True)
    core_sel, core_params = _select_features_by_variance(core_feats, parameters)
    print(f"  Core  — features selected "
          f"({len(core_sel)}/{len(parameters)}): {core_params}")

    print(f"  Core  — selecting k in 1..{HIER_CORE_K_MAX} "
          f"({core_mask.sum()} voxels, method={method.upper()})...")
    k_core = _select_k_core(
        core_feats[:, core_sel], HIER_CORE_K_MAX, method=method,
        n_init=n_init_gmm, df_sub=df_core,
        n_refs=n_refs, n_jobs=n_jobs,
        gmm_silhouette_guard=gmm_silhouette_guard,
        kmeans_gap_guard=kmeans_gap_guard,
        spatial_weight=spatial_weight)
    print(f"  Core  — optimal k = {k_core}")

    if k_core == 1:
        labels_core = np.zeros(len(df_core), dtype=int)
    else:
        labels_core, _ = _sub_segment(
            core_feats[:, core_sel], df_core, k_core,
            method, n_init_gmm, spatial_weight, 'core')
    n_core_lbl = len(np.unique(labels_core))

    # ── Identify necrotic sub-cluster within core ─────────────────────
    # Build IQR-space centroids for each core sub-cluster (all parameters,
    # not just core_sel, so that _necrosis_guard can access T2 and MD).
    _core_centroids = {
        sc: {param: float(core_feats[labels_core == sc, pi].mean())
             for pi, param in enumerate(parameters)}
        for sc in range(n_core_lbl)
        if (labels_core == sc).sum() > 0
    }
    _nec_ok, necrotic_sc_id = _necrosis_guard(_core_centroids, parameters)
    if not _nec_ok:
        necrotic_sc_id = None
        print("  [d_topo] Core: no clear necrotic sub-cluster — "
              "zone d_topo fallback will be used")
    else:
        print(f"  [d_topo] Core: necrotic sub-cluster = {necrotic_sc_id}")

    # ── Level 2b: periphery sub-segmentation (Bulk/Prolif/Infiltr) ──
    periph_feats              = features_scaled[periph_mask]
    df_periph                 = df_all[periph_mask].reset_index(drop=True)
    periph_sel, periph_params = _select_features_by_variance(periph_feats, parameters)
    print(f"  Periph — features selected "
          f"({len(periph_sel)}/{len(parameters)}): {periph_params}")

    # k selection: same criterion as the main pipeline (BIC for GMM,
    # Gap+Silhouette for kmeans/SRSC), capped at HIER_PERIPH_K_MAX.
    k_range_periph = range(2, HIER_PERIPH_K_MAX + 1)
    print(f"  Periph — selecting k in {list(k_range_periph)} "
          f"({periph_feats.shape[0]} voxels, method={method.upper()})...")
    n_periph_k, _ = select_optimal_k(
        periph_feats[:, periph_sel],
        method=method,
        df_all=df_periph,
        k_range=k_range_periph,
        n_refs=n_refs,
        n_jobs=n_jobs,
        spatial_weight=spatial_weight,
        n_init_gmm=n_init_gmm,
        gmm_silhouette_guard=gmm_silhouette_guard,
        kmeans_gap_guard=kmeans_gap_guard,
        output_dir=None,   # no sub-zone selection plots
    )
    print(f"  Periph — optimal k = {n_periph_k}")

    labels_peri, _ = _sub_segment(
        periph_feats[:, periph_sel], df_periph, n_periph_k,
        method, n_init_gmm, spatial_weight, 'periphery')
    n_peri_lbl = len(np.unique(labels_peri))

    # ── Assemble global labels ───────────────────────────────────
    final_labels              = np.zeros(len(features_scaled), dtype=int)
    final_labels[core_mask]   = labels_core
    final_labels[periph_mask] = labels_peri + n_core_lbl

    best_k_hier = n_core_lbl + n_peri_lbl
    core_ids    = list(range(n_core_lbl))
    peri_ids    = list(range(n_core_lbl, best_k_hier))

    print(f"  Core  clusters : {core_ids}  ({core_mask.sum()} voxels)")
    print(f"  Periph clusters: {peri_ids} ({periph_mask.sum()} voxels)")
    print(f"  Total k (hierarchical) : {best_k_hier}")

    return final_labels, {
        'hierarchical_valid'      : True,
        'best_k'                  : best_k_hier,
        'core_cluster_ids'        : core_ids,
        'periphery_cluster_ids'   : peri_ids,
        'core_id_k2'              : core_id,
        'necrotic_cluster_id'     : necrotic_sc_id,   # None if guard failed
        'core_features_used'      : core_params,
        'periphery_features_used' : periph_params,
        'l1_criterion'            : l1_crit,
    }
