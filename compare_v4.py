# =============================================================================
# compare_v4.py — Comparison metrics V4
# =============================================================================
# Two questions:
#   Inter-animal  : how do habitats diverge between animals / regimes?
#   Inter-method  : how much does the result depend on the algorithm?
#
# Pipeline:
#   1. Load CSVs, check normalisation consistency, compute centroids
#   2. Re-run _suggest_habitat per centroid (label used = suggest_habitat)
#      Flag mismatch with user label (annotated, not excluded)
#   3. QC gate — size floor (>30 vox) + η² floor (>0.10)
#              — low-confidence instances get weight 0.5
#   4. Bootstrap CI of centroids (voxel resampling, no re-clustering)
#   5. Stratify by suggest_label:
#      - direction  : cosine similarity between centroids
#      - magnitude  : Euclidean distance between centroids
#      - CI overlap : non-overlap → divergent parameter flagged
#   6. Relative contrast: intra-label / inter-label dispersion ratio
#      + centroid-level silhouette (cosine) with suggest_label partition
#   7. Output variable (2 components per label):
#      - C1 : inter-source coherence  = 1 − intra/inter ratio
#      - C2 : intra-source quality    = mean η² across sources
#   8. Figures F1–F6 + PDF report
#
# Values in the CSV are robust-scaled (0 = tumor median, unit = IQR).
# =============================================================================

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import silhouette_score

from matplotlib.colors import to_rgb

from config import (COMPARISON_PARAMETERS as _CFG_PARAMS, HABITAT_COLORS_HEX,
                    HABITAT_RULES, habitat_family, assign_habitat_colors)
from pipeline.labeling import _suggest_habitat
from pipeline.loading  import resolve_dti_parameters
from i18n              import t, display_label

# ---------------------------------------------------------------------------
# Label canonicalisation
# Strips Pass-2a disambiguation suffixes  (DTI-FA↓ / T2 low / DTI-RD 2/3)
# and Pass-2b aggressive marker (+) while preserving Pass-3 biological
# sub-types (hemorrhagic, cystic, edematous …).
# Order matters: more specific prefixes must come before their base label.
# ---------------------------------------------------------------------------
_CANONICAL_LABELS = [
    # Necrosis sub-types — before base "Necrosis"
    "Necrosis (hemorrhagic)",
    "Necrosis (proteic)",
    "Necrosis (cystic)",
    # Infiltrative sub-types — before base "Infiltrative"
    "Infiltrative (edematous)",
    "Infiltrative (infiltrated)",
    # Base HABITAT_RULES labels
] + [r["name"] for r in HABITAT_RULES] + [
    # System labels — longer variant before shorter
    "Bulk tumor (atypical)",
    "Bulk tumor",
    "Unknown",
]


def _canonicalize_label(label: str) -> str:
    """
    Return the canonical habitat name by stripping disambiguation suffixes.
      'Necrosis (kystique) (DTI-RD↓) +' → 'Necrosis (kystique)'
      'Bulk tumor (atypique) (T2↑)'      → 'Bulk tumor (atypique)'
      'Infiltrative (œdémateux)'          → 'Infiltrative (œdémateux)'
      'Necrosis +'                        → 'Necrosis'
    Labels with no matching prefix are returned unchanged.
    """
    low = label.lower().strip()
    for canonical in _CANONICAL_LABELS:
        if low.startswith(canonical.lower()):
            return canonical
    return label

# --- Module-level constants (move to config.py when needed) ---
SIZE_FLOOR  = 30    # min voxels per cluster to be considered valid
ETA2_FLOOR  = 0.10  # min mean η² to consider the partition informative
N_BOOT      = 200   # bootstrap resamples for centroid CI
COSINE_SIM_THRESHOLD    = 0.65  # raw cosine threshold (kept as reference)
COSINE_NORM_SKIP        = 0.35  # skip cosine for pairs where min centroid norm < this
COSINE_WEIGHTED_THRESHOLD = 0.45  # threshold for norm-weighted cosine (cos × min(norm, 1))
SHAPE_THRESHOLD         = 0.25  # max |â[k] − b̂[k]| on unit-norm centroids to be concordant
MAX_LABELS_F1           = 5     # max label columns per page in F1 heatmap
MAX_LABELS_F4           = 5     # max label rows per page in F4 bootstrap CI
ARCHETYPE_COS_THR       = 0.60  # min cos(cl_centroid, archetype) for "aligned" instance
ARCHETYPE_MAG_CV_THR    = 0.40  # max magnitude-ratio CV to be "severity-consistent"


# =============================================================================
# 1. HELPERS
# =============================================================================

# ---------------------------------------------------------------------------
# CL-relative centroid helpers
# ---------------------------------------------------------------------------

def _load_run_meta(out_dir):
    """
    Load scaling_info and cl_stats from run_meta.json saved alongside each run.
    Returns (scaling_info, cl_stats) or (None, None) when the file is absent
    (old runs without run_meta.json fall back to tumor-relative comparison).
    """
    if not out_dir:
        return None, None
    path = os.path.join(out_dir, "run_meta.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            meta = json.load(f)
        si = meta.get("scaling_info") or None
        cs = meta.get("cl_stats") or None
        return si, cs
    except Exception:
        return None, None


def _cl_normalize(tumor_centroid_arr, parameters, scaling_info, cl_stats):
    """
    Convert a tumor-relative centroid (IQR-scaled, 0 = tumor median) to a
    CL-relative centroid (z-score against contralateral tissue).

    Formula per parameter p:
        raw_p      = centroid_p  × tumor_IQR_p  + tumor_median_p
        cl_rel_p   = (raw_p − CL_median_p) / CL_std_p

    Returns ndarray of shape (len(parameters),) or None when any parameter
    is missing from either scaling_info or cl_stats.
    """
    cl_arr = []
    for v, p in zip(tumor_centroid_arr, parameters):
        si = scaling_info.get(p)
        cs = cl_stats.get(p)
        if not si or not cs:
            return None
        raw = v * si["iqr"] + si["median"]
        cl_std = max(float(cs["std"]), 1e-8)
        cl_arr.append((raw - float(cs["median"])) / cl_std)
    return np.array(cl_arr, dtype=float)


def _compute_archetypes(instances_by_label, parameters):
    """
    Compute per-label archetype from CL-relative centroids.

    An archetype is the mean CL-relative centroid across all instances that
    carry a cl_centroid.  It characterises the canonical MRI signature of each
    habitat in units that are common across animals (CL z-scores).

    Returns {label: {'centroid': arr, 'direction': arr, 'magnitude': float}}
    or {} when no CL data is available for any instance.
    """
    archetypes = {}
    for label, group in instances_by_label.items():
        cl_cents = [i["cl_centroid"] for i in group
                    if i.get("cl_centroid") is not None]
        if not cl_cents:
            continue
        archetype = np.mean(cl_cents, axis=0)
        mag = float(np.linalg.norm(archetype))
        if mag < 1e-8:
            continue
        archetypes[label] = {
            "centroid" : archetype,
            "direction": archetype / mag,
            "magnitude": mag,
        }
    return archetypes


def _balanced_chunks(lst, max_per_page):
    """Split lst into pages of at most max_per_page, distributed evenly across pages."""
    n = len(lst)
    if n <= max_per_page:
        return [lst]
    import math
    n_pages   = math.ceil(n / max_per_page)
    base, rem = divmod(n, n_pages)
    chunks, start = [], 0
    for i in range(n_pages):
        size = base + (1 if i < rem else 0)
        chunks.append(lst[start:start + size])
        start += size
    return chunks


def _cosine_dist(a, b):
    """Cosine distance in [0, 2]. Guards against zero-norm vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _euclidean_dist(a, b):
    return float(np.linalg.norm(a - b))


def _eta_squared(df, parameters):
    """Variance explained per parameter by the Habitat partition."""
    labels = df['Habitat'].values
    out    = {}
    for p in parameters:
        if p not in df.columns:
            continue
        vals   = np.nan_to_num(df[p].values.astype(float), nan=0.0)
        grand  = vals.mean()
        ss_tot = ((vals - grand) ** 2).sum()
        if ss_tot < 1e-12:
            out[p] = 0.0
            continue
        ss_bet = sum(
            (labels == c).sum() * (vals[labels == c].mean() - grand) ** 2
            for c in np.unique(labels)
        )
        out[p] = float(ss_bet / ss_tot)
    return out


def _bootstrap_ci(values, n_boot=N_BOOT, ci=95, seed=42):
    """
    Resample voxels with replacement, recompute centroid each time.
    Returns (lo, hi) per parameter.
    """
    rng = np.random.default_rng(seed)
    n   = len(values)
    if n < 3:
        m = values.mean(axis=0)
        return m.copy(), m.copy()
    boots = np.array([
        values[rng.integers(0, n, n)].mean(axis=0)
        for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (100 - ci) / 2,       axis=0)
    hi = np.percentile(boots, 100 - (100 - ci) / 2, axis=0)
    return lo, hi


# =============================================================================
# 2. INSTANCE EXTRACTION
# =============================================================================

def _load_instances(result, common_params):
    """
    Load habitats_result.csv for one run.
    One instance = one integer Habitat cluster.
    Returns list of instance dicts.
    """
    df     = pd.read_csv(result["csv"])
    source = result["name"]

    # dti_mode for _suggest_habitat rule selection
    _, dti_mode = resolve_dti_parameters(list(common_params))

    # CL-relative normalisation: load run_meta.json (scaling_info + cl_stats)
    # saved alongside each run.  Falls back gracefully when absent.
    scaling_info, cl_stats = _load_run_meta(result.get("out_dir", ""))

    # Normalisation check: robust-scaled values should be centred near 0
    tumor_medians = {p: float(df[p].median()) for p in common_params if p in df.columns}
    norm_ok = all(abs(v) < 0.15 for v in tumor_medians.values())

    # η² for the whole source (needs all clusters)
    eta2_source = _eta_squared(df, common_params)
    mean_eta2   = float(np.mean(list(eta2_source.values()))) if eta2_source else 0.0

    total_vox = max(len(df), 1)   # total tumour voxels for this source

    instances = []
    for cid, sub in df.groupby('Habitat'):
        n_vox = len(sub)
        vals  = np.nan_to_num(
            sub[common_params].to_numpy(float), nan=0.0)
        centroid = vals.mean(axis=0)
        norm_val = float(np.linalg.norm(centroid))

        # User label (modal value of Habitat_label in this cluster)
        if 'Habitat_label' in sub.columns:
            modes      = sub['Habitat_label'].dropna().mode()
            user_label = str(modes.iat[0]) if len(modes) else f"H{cid}"
        else:
            user_label = f"H{cid}"

        # Auto-suggest from centroid (robust-scaled, 0 = tumor median)
        c_dict = dict(zip(common_params, centroid.tolist()))
        suggest_label, suggest_desc, confidence, _, _ = _suggest_habitat(
            c_dict, common_params, dti_mode=dti_mode)

        # Canonical label: strip Pass-2a/2b disambiguation suffixes,
        # keep Pass-3 biological sub-types
        canonical_label = _canonicalize_label(user_label)

        # Mismatch: canonical differs from auto-suggest (ignores suffixes)
        mismatch = (
            suggest_label.lower() not in canonical_label.lower() and
            canonical_label.lower() not in suggest_label.lower()
        )

        # Bootstrap CI
        ci_lo, ci_hi = _bootstrap_ci(vals)

        # CL-relative centroid (None when run_meta.json unavailable)
        cl_centroid = None
        if scaling_info and cl_stats:
            cl_centroid = _cl_normalize(centroid, common_params, scaling_info, cl_stats)

        # QC flags
        size_ok  = n_vox >= SIZE_FLOOR
        eta2_ok  = mean_eta2 >= ETA2_FLOOR
        low_conf = (confidence == "low" or not size_ok or not eta2_ok)

        instances.append({
            "source"       : source,
            "cluster_id"   : int(cid),
            "user_label"   : user_label,
            "label"        : canonical_label, # grouping key — sub-type preserved, suffixes stripped
            "suggest_desc" : suggest_desc,
            "confidence"   : confidence,
            "mismatch"     : mismatch,
            "n_voxels"     : n_vox,
            "proportion"   : n_vox / total_vox,   # % of tumour volume
            "total_vox"    : total_vox,
            "centroid"     : centroid,
            "cl_centroid"  : cl_centroid,  # CL-relative z-score, None if no CL data
            "values"       : vals,
            "ci_lo"        : ci_lo,
            "ci_hi"        : ci_hi,
            "norm"         : norm_val,
            "eta2"         : eta2_source,
            "mean_eta2"    : mean_eta2,
            "size_ok"      : size_ok,
            "eta2_ok"      : eta2_ok,
            "low_conf"     : low_conf,
            "weight"       : 0.5 if low_conf else 1.0,
            "norm_ok"      : norm_ok,
        })
    return instances


# =============================================================================
# 3. COHERENCE METRICS
# =============================================================================

def _label_coherence(instances_by_label, parameters):
    """
    For each label group:
      intra_cosine : mean pairwise cosine distance within label
      inter_cosine : mean pairwise cosine distance between this label and all others
      ratio        : intra_cosine / inter_cosine  (direction only)
      intra_euc    : mean pairwise Euclidean distance within label
      inter_euc    : mean pairwise Euclidean distance between this label and others
      ratio_euc    : intra_euc / inter_euc  (direction + magnitude)
      cos_sims     : list of pairwise cosine similarities within label
      euc_dists    : list of pairwise Euclidean distances within label
      overlap_map  : {(srcA, srcB): {concordant, divergent_params}}
      coherent     : ratio < 0.5 AND ratio_euc < 0.5 AND mean cos_sim > threshold
    """
    all_inst        = [i for grp in instances_by_label.values() for i in grp]
    all_sources     = set(i["source"] for i in all_inst)
    n_total_sources = len(all_sources)
    results         = {}

    def _shape_entry(inst_a, inst_b):
        """Shape concordance between two instances (unit-norm centroid diff)."""
        ca, cb = inst_a["centroid"], inst_b["centroid"]
        na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
        scale_ratio = float(na / nb) if nb > 1e-8 else 1.0
        if min(na, nb) < COSINE_NORM_SKIP:
            return {"concordant": True, "divergent_params": [],
                    "scale_ratio": scale_ratio, "near_origin": True}
        a_hat = ca / na
        b_hat = cb / nb
        div = [p for k, p in enumerate(parameters)
               if abs(float(a_hat[k]) - float(b_hat[k])) > SHAPE_THRESHOLD]
        return {"concordant": len(div) == 0, "divergent_params": div,
                "scale_ratio": scale_ratio, "near_origin": False}

    for label, group in instances_by_label.items():
        c_in  = [i["centroid"] for i in group]
        c_out = [i["centroid"] for i in all_inst if i["label"] != label]

        # Separate pairs by source relationship
        all_idx     = list(combinations(range(len(group)), 2))
        cross_idx   = [(ia, ib) for ia, ib in all_idx
                       if group[ia]["source"] != group[ib]["source"]]
        within_idx  = [(ia, ib) for ia, ib in all_idx
                       if group[ia]["source"] == group[ib]["source"]]

        # ── Ratios — cross-source pairs only (within-source excluded) ─
        intra = (float(np.mean([_cosine_dist(c_in[ia], c_in[ib])
                                for ia, ib in cross_idx]))
                 if cross_idx else 0.0)
        inter = (float(np.mean([_cosine_dist(a, b)
                                 for a in c_in for b in c_out]))
                 if c_out else 1.0)
        ratio = float(intra / inter) if inter > 1e-8 else 0.0

        intra_euc = (float(np.mean([_euclidean_dist(c_in[ia], c_in[ib])
                                    for ia, ib in cross_idx]))
                     if cross_idx else 0.0)
        inter_euc = (float(np.mean([_euclidean_dist(a, b)
                                     for a in c_in for b in c_out]))
                     if c_out else 1.0)
        ratio_euc = float(intra_euc / inter_euc) if inter_euc > 1e-8 else 0.0

        # ── wcos / cos_sims — cross-source only ───────────────────────
        cos_sims          = []
        euc_dists         = []
        weighted_cos_list = []
        n_cos_skipped     = 0
        for ia, ib in cross_idx:
            a, b = c_in[ia], c_in[ib]
            cs = 1.0 - _cosine_dist(a, b)
            cos_sims.append(cs)
            euc_dists.append(_euclidean_dist(a, b))
            na, nb   = np.linalg.norm(a), np.linalg.norm(b)
            min_norm = min(na, nb)
            if min_norm < COSINE_NORM_SKIP:
                n_cos_skipped += 1
            else:
                weighted_cos_list.append(cs * min(min_norm, 1.0))

        # ── Cross-source shape concordance ────────────────────────────
        overlap_map = {}
        for ia, ib in cross_idx:
            key = (group[ia]["source"], group[ib]["source"])
            overlap_map[key] = _shape_entry(group[ia], group[ib])

        # ── Within-source divergence (diagnostic only) ────────────────
        # Only stored when two clusters of the same animal diverge.
        # When divergent → also compute per-cluster cross-source breakdown.
        within_divergent = {}   # {source: [{"cid_a","cid_b","entry"}]}
        for ia, ib in within_idx:
            entry = _shape_entry(group[ia], group[ib])
            if not entry["concordant"]:
                src = group[ia]["source"]
                within_divergent.setdefault(src, []).append({
                    "cid_a" : group[ia]["cluster_id"],
                    "cid_b" : group[ib]["cluster_id"],
                    "entry" : entry,
                })

        # Per-cluster cross-source breakdown — only for internally divergent sources
        expanded_cross = {}  # {(src_div, cid, other_src): entry}
        for src in within_divergent:
            div_insts   = [i for i in group if i["source"] == src]
            other_insts = [i for i in group if i["source"] != src]
            for inst_a in div_insts:
                for inst_b in other_insts:
                    key = (inst_a["source"], inst_a["cluster_id"], inst_b["source"])
                    expanded_cross[key] = _shape_entry(inst_a, inst_b)

        mean_cos          = float(np.mean(cos_sims)) if cos_sims else 1.0
        mean_weighted_cos = (float(np.mean(weighted_cos_list))
                             if weighted_cos_list else 1.0)
        cos_ok   = mean_weighted_cos >= COSINE_WEIGHTED_THRESHOLD
        coherent = (ratio < 0.5 and ratio_euc < 0.5 and cos_ok)

        # ── Archetype alignment (CL-relative space) ───────────────────
        # Archetype = mean CL-relative centroid for this label.
        # Per-instance metrics:
        #   arch_cos  : cosine(cl_centroid, archetype_direction) — direction match
        #   mag_ratio : ‖cl_centroid‖ / archetype_magnitude       — severity match
        cl_cents = [i["cl_centroid"] for i in group
                    if i.get("cl_centroid") is not None]
        arch_cos_list  = []
        mag_ratio_list = []
        if cl_cents:
            archetype     = np.mean(cl_cents, axis=0)
            arch_mag      = float(np.linalg.norm(archetype))
            if arch_mag > 1e-8:
                arch_dir  = archetype / arch_mag
                for i in group:
                    cc = i.get("cl_centroid")
                    if cc is None:
                        continue
                    arch_cos_list.append(1.0 - _cosine_dist(cc, arch_dir))
                    inst_mag = float(np.linalg.norm(cc))
                    mag_ratio_list.append(inst_mag / arch_mag)

        mean_arch_align = (float(np.mean(arch_cos_list))
                           if arch_cos_list else None)
        mag_cv = (float(np.std(mag_ratio_list) / np.mean(mag_ratio_list))
                  if len(mag_ratio_list) > 1 and np.mean(mag_ratio_list) > 1e-8
                  else None)

        # ── Point 6 — Reproducibility ─────────────────────────────────
        sources_present  = set(i["source"] for i in group)
        n_present        = len(sources_present)
        presence_rate    = n_present / n_total_sources if n_total_sources > 0 else 0.0

        # ── Point 3 — Volume / proportion consistency ─────────────────
        # One proportion per instance (cluster % of tumour in its source).
        # If a source has several clusters with the same label, each appears.
        proportions = [i["proportion"] for i in group]
        mean_prop   = float(np.mean(proportions))
        std_prop    = float(np.std(proportions, ddof=1)) if len(proportions) > 1 else 0.0
        prop_cv     = std_prop / mean_prop if mean_prop > 1e-6 else 0.0

        results[label] = {
            "n_instances"      : len(group),
            "sources"          : [i["source"] for i in group],
            "mean_centroid"    : np.mean(c_in, axis=0),
            "intra_cosine"     : float(intra),
            "inter_cosine"     : float(inter),
            "ratio"            : ratio,
            "intra_euc"        : intra_euc,
            "inter_euc"        : inter_euc,
            "ratio_euc"        : ratio_euc,
            "cos_sims"         : cos_sims,
            "euc_dists"        : euc_dists,
            "mean_weighted_cos": mean_weighted_cos,
            "n_cos_skipped"    : n_cos_skipped,
            "overlap_map"      : overlap_map,
            "within_divergent" : within_divergent,
            "expanded_cross"   : expanded_cross,
            "low_conf_count"   : sum(i["low_conf"] for i in group),
            "coherent"         : coherent,
            # reproducibility
            "n_sources_present": n_present,
            "n_sources_total"  : n_total_sources,
            "presence_rate"    : presence_rate,
            # volume consistency
            "mean_proportion"  : mean_prop,
            "prop_cv"          : prop_cv,
            "proportions"      : proportions,
            # CL-relative archetype alignment
            "arch_align"       : mean_arch_align,  # mean cos(cl_i, archetype_dir)
            "mag_cv"           : mag_cv,            # CV of severity ratios
            "cl_available"     : bool(cl_cents),
        }
    return results


def _centroid_silhouette(instances_by_label):
    """
    Silhouette score using suggest_label as partition, cosine distance,
    one data point per instance (centroid). Returns float or None.
    """
    labels    = []
    centroids = []
    for label, group in instances_by_label.items():
        for inst in group:
            labels.append(label)
            centroids.append(inst["centroid"])
    if len(set(labels)) < 2 or len(labels) < 2:
        return None
    n = len(centroids)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine_dist(centroids[i], centroids[j])
            D[i, j] = D[j, i] = d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(silhouette_score(D, labels, metric='precomputed'))


# =============================================================================
# 4. FIGURES
# =============================================================================

def _fig_centroid_heatmap(instances, instances_by_label, parameters, output_dir):
    """
    F1 — Rows = sources, columns = labels, one subplot per cell.
    Splits into multiple pages (MAX_LABELS_F1 columns each) when there are
    many labels so each page remains readable.
    Returns a list of paths (one per page).
    """
    labels  = list(instances_by_label.keys())
    sources = list(dict.fromkeys(i["source"] for i in instances))
    n_src, n_par = len(sources), len(parameters)

    # Pre-aggregate: {label: {source: mean_centroid}}
    agg = {}
    for label, group in instances_by_label.items():
        agg[label] = {}
        for src in sources:
            sub = [i for i in group if i["source"] == src]
            if sub:
                agg[label][src] = np.mean([i["centroid"] for i in sub], axis=0)

    vmax = max(abs(i["centroid"]).max() for i in instances)
    vmax = float(np.clip(vmax, 0.5, 3.0))

    chunks  = _balanced_chunks(labels, MAX_LABELS_F1)
    n_pages = len(chunks)
    paths   = []

    for pg, chunk in enumerate(chunks):
        n_lbl = len(chunk)
        fig, axes = plt.subplots(
            n_src, n_lbl,
            figsize=(max(5, 2.0 * n_lbl), max(3, 1.5 * n_src + 1.2)),
            squeeze=False)

        for col, label in enumerate(chunk):
            low_c = any(i["low_conf"] for i in instances_by_label[label])
            for row, src in enumerate(sources):
                ax = axes[row][col]
                if src in agg[label]:
                    c = agg[label][src]
                    ax.imshow(c.reshape(1, -1), aspect='auto',
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                    ax.set_xticks(range(n_par))
                    ax.set_xticklabels(parameters, rotation=45,
                                       ha='right', fontsize=6.5)
                    ax.set_yticks([])
                    for k, v in enumerate(c):
                        ax.text(k, 0, f"{v:+.2f}", ha='center', va='center',
                                fontsize=6, color='black')
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, "—", ha='center', va='center',
                            transform=ax.transAxes, fontsize=14, color='#BBBBBB')
                if col == 0:
                    ax.set_ylabel(src, fontsize=8, rotation=0,
                                  labelpad=55, va='center')
                if row == 0:
                    ttl_color = '#E05252' if low_c else '#1F3864'
                    ax.set_title(label, fontsize=8, fontweight='bold',
                                 color=ttl_color)

        pg_tag = f"  ({pg + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(
            f"F1 — Centroid heatmap by label{pg_tag}  (0 = tumor median, unit = IQR)\n"
            "Red title = at least one low-confidence instance",
            fontsize=10, fontweight='bold', y=1.01)
        plt.tight_layout()
        suffix = f"_p{pg + 1}" if n_pages > 1 else ""
        path   = os.path.join(output_dir, f"F1_centroid_heatmap{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)

    return paths


def _fig_cosine_matrix(instances, output_dir):
    """F2 — Pairwise cosine similarity matrix, sorted by label."""
    sorted_inst = sorted(instances, key=lambda i: i["label"])
    n    = len(sorted_inst)
    D    = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = 1.0 - _cosine_dist(
                sorted_inst[i]["centroid"], sorted_inst[j]["centroid"])

    tick_lbl = [f"{i['source']}\n{i['label'][:14]}" for i in sorted_inst]

    fig, ax = plt.subplots(
        figsize=(max(5, 0.55 * n + 2), max(4, 0.55 * n + 2)))
    im = ax.imshow(D, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_lbl, fontsize=7)
    plt.colorbar(im, ax=ax, label='Cosine similarity  (1 = identical direction)')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{D[i,j]:.2f}", ha='center', va='center',
                    fontsize=6, color='black')
    ax.set_title("F2 — Pairwise cosine similarity between centroids\n"
                 "(green = similar direction, red = divergent)",
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "F2_cosine_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _fig_dispersion_ratio(label_coh, output_dir):
    """
    F3 — Intra/inter dispersion ratio per label (relative contrast).
    Green < 0.5 (coherent), orange 0.5–0.8, red > 0.8.
    """
    labels = list(label_coh.keys())
    ratios = [label_coh[l]["ratio"] for l in labels]
    colors = ['#5CB85C' if r < 0.5 else ('#F0A030' if r < 0.8 else '#E05252')
              for r in ratios]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 4.5))
    bars = ax.bar(range(len(labels)), ratios, color=colors,
                  edgecolor='white', linewidth=1.2, zorder=2)
    ax.axhline(1.0, color='#555555', ls='--', lw=1.2, zorder=3,
               label='ratio = 1  (intra = inter, no separation)')
    ax.axhline(0.5, color='#5CB85C', ls=':', lw=1.2, zorder=3,
               label='ratio = 0.5  (coherence threshold)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Intra / inter dispersion ratio  (cosine)", fontsize=10)
    ax.set_title("F3 — Relative contrast per label\n"
                 "lower ratio = tighter intra-label cluster, better separation from others",
                 fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(fontsize=8)
    for bar, r, lbl in zip(bars, ratios, labels):
        n   = label_coh[lbl]["n_instances"]
        lc  = label_coh[lbl]["low_conf_count"]
        ann = f"{r:.2f}\n(n={n})"
        if lc:
            ann += f"\n⚠{lc}"
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                ann, ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, "F3_dispersion_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _fig_bootstrap_ci(instances, parameters, output_dir):
    """
    F4 — Centroid ± 95% bootstrap CI per source, faceted by label.
    Marker shape encodes within-source cluster rank:
      1st cluster from a source → circle  'o'
      2nd cluster from a source → square  's'
      3rd cluster from a source → triangle '^'
    Same color = same source. Different shape = different cluster of that source.
    Non-overlapping CI on a parameter → divergent.
    """
    # Marker shapes by within-source occurrence index (0-based)
    MARKERS = ['o', 's', '^', 'D', 'P', 'X']

    labels  = sorted(set(i["label"] for i in instances))
    sources = list(dict.fromkeys(i["source"] for i in instances))
    n_par   = len(parameters)
    x       = np.arange(n_par)

    cmap_src  = plt.cm.tab10(np.linspace(0, 0.9, max(len(sources), 1)))
    src_color = {s: c for s, c in zip(sources, cmap_src)}

    chunks  = _balanced_chunks(labels, MAX_LABELS_F4)
    n_pages = len(chunks)
    paths   = []

    for pg, chunk in enumerate(chunks):
        n_rows = len(chunk)
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(max(7, n_par * 0.9), max(3, 2.5 * n_rows)),
            sharex=True, squeeze=False)

        for row, label in enumerate(chunk):
            ax           = axes[row][0]
            group        = [i for i in instances if i["label"] == label]
            src_count    = {}
            legend_shown = set()
            for inst in group:
                c    = inst["centroid"]
                lo   = inst["ci_lo"]
                hi   = inst["ci_hi"]
                col  = src_color[inst["source"]]
                rank = src_count.get(inst["source"], 0)
                src_count[inst["source"]] = rank + 1
                marker = MARKERS[min(rank, len(MARKERS) - 1)]
                if inst["source"] not in legend_shown:
                    legend_label = inst["source"]
                    legend_shown.add(inst["source"])
                else:
                    legend_label = "_nolegend_"
                ax.errorbar(x, c, yerr=[c - lo, hi - c],
                            fmt=marker, color=col, capsize=4,
                            linewidth=1.5, markersize=6, label=legend_label)

            ax.axhline(0, color='#BBBBBB', ls='--', lw=0.8)
            ax.set_ylabel(label[:22], fontsize=8, rotation=0,
                          labelpad=68, va='center')
            ax.legend(fontsize=7, loc='upper right')
            if row == 0:
                pg_tag = f"  ({pg + 1}/{n_pages})" if n_pages > 1 else ""
                ax.set_title(
                    f"F4 — Bootstrap CI of centroids per label{pg_tag}"
                    "  (95%, voxel resampling)\n"
                    "Color = source  ·  Shape: ○ 1st cluster  □ 2nd cluster  △ 3rd cluster",
                    fontsize=10, fontweight='bold')

        axes[-1][0].set_xticks(x)
        axes[-1][0].set_xticklabels(parameters, rotation=35, ha='right', fontsize=8)
        plt.tight_layout()
        suffix = f"_p{pg + 1}" if n_pages > 1 else ""
        path   = os.path.join(output_dir, f"F4_bootstrap_ci{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)

    return paths


def _fig_qc_table(instances, output_dir, show_user_label=False):
    """
    F5 — QC summary table.
    N vox, η², Conf. are color-coded green/red instead of separate check columns.
    'User label' column only appears when show_user_label=True (user renamed a label).
    """
    rows = []
    for inst in instances:
        top_e = sorted(inst["eta2"].items(), key=lambda x: x[1], reverse=True)[:3]
        eta2_str = ", ".join(f"{p}={v:.2f}" for p, v in top_e)
        row = [inst["source"], inst["label"]]
        if show_user_label:
            row.append(inst["user_label"] if inst["mismatch"] else "—")
        row += [
            str(inst["n_voxels"]),
            f"{inst['proportion']*100:.1f}%",
            f"{inst['mean_eta2']:.2f}",
            inst["confidence"],
            eta2_str,
        ]
        rows.append(row)

    cols = ["Source", "Label (auto)"]
    if show_user_label:
        cols.append("User label")
    cols += ["N vox", "% vol", "η²", "Conf.", "Top η² params"]

    idx_nvox = cols.index("N vox")
    idx_pvol = cols.index("% vol")
    idx_eta2 = cols.index("η²")
    idx_conf = cols.index("Conf.")

    fig, ax = plt.subplots(
        figsize=(max(12, 1.9 * len(cols)), max(2.0, 0.42 * len(rows) + 1.6)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    GREEN_BG,  GREEN_FG   = '#E8F5E9', '#2E7D32'
    RED_BG,    RED_FG     = '#FFEBEE', '#C62828'
    ORANGE_BG, ORANGE_FG  = '#FFF3E0', '#E65100'

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1F3864')
            cell.set_text_props(color='white', fontweight='bold')
            continue
        inst = instances[r - 1]
        if c == idx_nvox:
            ok = inst["size_ok"]
            cell.set_facecolor(GREEN_BG if ok else RED_BG)
            cell.set_text_props(color=GREEN_FG if ok else RED_FG, fontweight='bold')
        elif c == idx_pvol:
            # colour-code by proportion: < 5 % = grey, 5-30 % = normal, > 30 % = bold
            pct = inst["proportion"] * 100
            if pct < 5:
                cell.set_facecolor('#F5F5F5')
                cell.set_text_props(color='#9E9E9E')
            elif pct > 30:
                cell.set_text_props(fontweight='bold')
        elif c == idx_eta2:
            ok = inst["eta2_ok"]
            cell.set_facecolor(GREEN_BG if ok else RED_BG)
            cell.set_text_props(color=GREEN_FG if ok else RED_FG, fontweight='bold')
        elif c == idx_conf:
            if inst["confidence"] == "high":
                cell.set_facecolor(GREEN_BG)
                cell.set_text_props(color=GREEN_FG, fontweight='bold')
            elif inst["confidence"] == "low":
                cell.set_facecolor(RED_BG)
                cell.set_text_props(color=RED_FG, fontweight='bold')
            else:
                cell.set_facecolor(ORANGE_BG)
                cell.set_text_props(color=ORANGE_FG)
        elif inst["low_conf"]:
            cell.set_facecolor('#FFFDE7')

    subtitle = ("  ·  'User label' shown: at least one label was manually renamed"
                if show_user_label else "")
    ax.set_title(
        "F5 — QC table  "
        "(N vox: green ≥ 30 / red < 30  ·  % vol: grey < 5 % / bold > 30 %  "
        "·  η²: green ≥ 0.10 / red < 0.10  ·  Conf.: green high / orange medium / red low)\n"
        "Light yellow row = low-confidence instance (weight 0.5 in summary metrics)"
        + subtitle,
        fontsize=8.5, fontweight='bold', pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "F5_qc_table.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _fig_radar_charts(instances, instances_by_label, parameters, output_dir):
    """
    F7 — Radar chart per label. One line per source (or method).
    Axes = MRI parameters. Values = mean centroid (robust-scaled, 0 = tumor median).
    Overlap of lines = concordant; separation = discordant.
    """
    import math

    labels  = list(instances_by_label.keys())
    sources = list(dict.fromkeys(i["source"] for i in instances))
    n_par   = len(parameters)

    angles = [n / float(n_par) * 2 * math.pi for n in range(n_par)]
    angles += angles[:1]  # close the polygon

    ncols = min(3, len(labels))
    nrows = math.ceil(len(labels) / ncols)

    cmap_src  = plt.cm.tab10(np.linspace(0, 0.9, max(len(sources), 1)))
    src_color = {s: c for s, c in zip(sources, cmap_src)}

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        subplot_kw=dict(polar=True),
        squeeze=False)

    dash_styles = ['--', ':', '-.', (0, (3, 1, 1, 1))]  # up to 4 sub-habitats

    for idx, label in enumerate(labels):
        row = idx // ncols
        col = idx % ncols
        ax  = axes[row][col]

        group  = instances_by_label[label]
        by_src = {}
        for inst in group:
            by_src.setdefault(inst["source"], []).append(inst["centroid"])

        all_c = np.array([c for cs in by_src.values() for c in cs])
        vmax  = max(float(np.abs(all_c).max()), 0.5)

        for src, cs in by_src.items():
            color  = src_color[src]
            mean_c = np.mean(cs, axis=0)
            vals_m = mean_c.tolist() + mean_c[:1].tolist()

            if len(cs) == 1:
                # Single habitat → solid line
                ax.plot(angles, vals_m, color=color, linewidth=2,
                        label=src, alpha=0.90)
                ax.fill(angles, vals_m, color=color, alpha=0.07)
            else:
                # Multiple habitats from same source → dashed individuals + solid mean
                for k, c in enumerate(cs):
                    vals_k = c.tolist() + c[:1].tolist()
                    ls     = dash_styles[k % len(dash_styles)]
                    ax.plot(angles, vals_k, color=color, linewidth=1.2,
                            ls=ls, alpha=0.55,
                            label=f"{src} — habit.{k+1}" if k == 0
                                  else f"— habit.{k+1}")
                # Mean: solid, slightly thicker
                ax.plot(angles, vals_m, color=color, linewidth=2.5,
                        ls='-', alpha=0.90, label=f"{src} (moy.)")
                ax.fill(angles, vals_m, color=color, alpha=0.07)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(parameters, size=7)
        ax.set_ylim(-vmax, vmax)
        ax.plot(angles, [0.0] * (n_par + 1), color='#AAAAAA',
                lw=0.8, ls='--', zorder=0)

        low_c = any(i["low_conf"] for i in group)
        ax.set_title(label, fontsize=9, fontweight='bold',
                     color='#E05252' if low_c else '#1F3864', pad=14)
        ax.legend(fontsize=7, loc='upper right',
                  bbox_to_anchor=(1.35, 1.15))

    # Hide unused axes
    for idx in range(len(labels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "F7 — Radar charts per label  (0 = tumor median, unit = IQR)\n"
        "Solid = source mean  |  Dashed = individual habitats (same label, same source)",
        fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "F7_radar_charts.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _fig_output_components(label_coh, instances_by_label, output_dir):
    """
    F6 — Scatter: Component 1 (inter-source coherence) vs Component 2
    (intra-source quality / mean η²). One point per label.
    Colour = green (coherent) / red (non-coherent).
    Size = number of instances.
    """
    labels = list(label_coh.keys())
    c1 = [1.0 - label_coh[l]["ratio"] for l in labels]   # higher = better
    c2 = [float(np.average(
              [i["mean_eta2"] for i in instances_by_label[l]],
              weights=[i["weight"] for i in instances_by_label[l]]))
          for l in labels]
    sizes  = [label_coh[l]["n_instances"] * 100 for l in labels]
    colors = ['#5CB85C' if label_coh[l]["coherent"] else '#E05252'
              for l in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(c1, c2, s=sizes, c=colors, edgecolors='white',
               linewidths=1.2, zorder=3, alpha=0.85)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl[:18], (c1[i], c2[i]),
                    textcoords='offset points', xytext=(6, 4), fontsize=7.5)

    ax.axvline(0.5, color='#AAAAAA', ls=':', lw=1.0,
               label='coherence threshold')
    ax.axhline(ETA2_FLOOR, color='#AAAAAA', ls='--', lw=1.0,
               label=f'η² floor ({ETA2_FLOOR})')
    ax.set_xlabel("C1 — inter-source coherence  (1 − intra/inter ratio)",
                  fontsize=9)
    ax.set_ylabel("C2 — intra-source quality  (weighted mean η²)", fontsize=9)
    ax.set_title("F6 — Output components per label\n"
                 "Top-right = coherent across sources AND high internal structure",
                 fontsize=10, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, max(c2 + [0.5]) + 0.05)
    ax.legend(handles=[
        Patch(color='#5CB85C', label='Coherent'),
        Patch(color='#E05252', label='Non-coherent'),
    ], fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(output_dir, "F6_output_components.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# =============================================================================
# 5. SEGMENTATION GRID
# =============================================================================

def _fig_segmentation_grid(results, instances_by_label, output_dir):
    """
    F8 — One subplot per animal: 2D habitat map on the modal slice,
    coloured by Habitat_label with a common legend across all animals.
    """
    # --- Collect every Habitat_label value present in any CSV ---
    all_full_labels = set()
    for r in results:
        try:
            df_tmp = pd.read_csv(r["csv"])
            if 'Habitat_label' in df_tmp.columns:
                all_full_labels.update(
                    df_tmp['Habitat_label'].dropna().astype(str).unique())
        except Exception:
            pass

    # Strip disambiguation suffixes (e.g. " (DTI-FA low)", " +") from every
    # full label, then assign colors and display names on canonical labels.
    # Multiple full labels that share the same canonical → same color/legend entry.
    from collections import defaultdict as _dd
    full_to_can = {full: _canonicalize_label(full) for full in all_full_labels}
    unique_cans = sorted(set(full_to_can.values()))

    # One color per canonical, tones within same family when 2+ canonicals share it
    canonical_color = assign_habitat_colors(unique_cans)

    # All full labels inherit their canonical's color
    full_label_color = {full: canonical_color[_canonicalize_label(full)]
                        for full in all_full_labels}

    # Display name: translated family, numbered only when 2+ canonicals in family
    family_to_cans = _dd(list)
    for can in unique_cans:
        fam = habitat_family(can)
        if can not in family_to_cans[fam]:
            family_to_cans[fam].append(can)

    canonical_display = {}
    for fam, cans in sorted(family_to_cans.items()):
        fam_display = display_label(fam)
        if len(cans) == 1:
            canonical_display[cans[0]] = fam_display
        else:
            for n, can in enumerate(cans, 1):
                canonical_display[can] = f"{fam_display} {n}"

    full_label_display = {full: canonical_display[_canonicalize_label(full)]
                          for full in all_full_labels}

    n     = len(results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.8 * ncols, 4.5 * nrows + 1.2),
                              squeeze=False)
    axes_flat = axes.flatten()

    for idx, r in enumerate(results):
        ax = axes_flat[idx]
        try:
            df = pd.read_csv(r["csv"])
        except Exception:
            ax.text(0.5, 0.5, "CSV not found", ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='red')
            ax.set_title(r["name"]); ax.axis('off')
            continue

        if 'Habitat_label' not in df.columns or df.empty:
            ax.text(0.5, 0.5, "No label data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(r["name"]); ax.axis('off')
            continue

        # Modal slice
        slice_id = int(df['Slice'].mode()[0])
        df_s = df[df['Slice'] == slice_id]

        xs = df_s['X'].values.astype(int)
        ys = df_s['Y'].values.astype(int)
        lbls = df_s['Habitat_label'].astype(str).values

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        img = np.ones((h, w, 3), dtype=float)   # white background
        for lbl in np.unique(lbls):
            mask = lbls == lbl
            rgb  = to_rgb(full_label_color.get(lbl, '#888888'))
            img[ys[mask] - y_min, xs[mask] - x_min] = rgb

        ax.imshow(img, interpolation='nearest', origin='upper',
                  aspect='equal')
        ax.set_title(f"{r['name']}  (K={r['best_k']})",
                     fontsize=10, fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    # Common legend — one entry per display name (sorted alphabetically)
    legend_map = {}
    for full, display in full_label_display.items():
        legend_map[display] = full_label_color[full]
    patches = [Patch(facecolor=legend_map[name], label=name)
               for name in sorted(legend_map)]
    fig.legend(handles=patches, loc='lower center',
               ncol=min(len(legend_map), 5), fontsize=9,
               bbox_to_anchor=(0.5, 0.01), frameon=True,
               edgecolor='#CCCCCC')

    fig.suptitle("F8 — Habitat segmentation maps  (common colour legend)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    path = os.path.join(output_dir, "F8_segmentation_grid.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# =============================================================================
# 5b. ARCHETYPE FIGURE
# =============================================================================

def _fig_archetype_radar(archetypes, instances_by_label, parameters, label_coh,
                          output_dir):
    """
    F9 — CL-relative archetype per label.

    Each subplot shows the mean CL-relative centroid (archetype) for one label
    as a filled radar.  Individual instances are overlaid as thin lines.
    Axis values are CL z-scores:  0 = CL median,  ±1 = 1 CL std.

    Only labels with at least one CL-relative centroid are shown.
    Returns the output path, or None when no CL data is available.
    """
    import math

    labels_with_cl = [l for l in archetypes]
    if not labels_with_cl:
        return None

    n_par    = len(parameters)
    angles   = [n / float(n_par) * 2 * math.pi for n in range(n_par)]
    angles  += angles[:1]

    ncols = min(3, len(labels_with_cl))
    nrows = math.ceil(len(labels_with_cl) / ncols)

    sources   = list(dict.fromkeys(
        i["source"] for grp in instances_by_label.values() for i in grp))
    cmap_src  = plt.cm.tab10(np.linspace(0, 0.9, max(len(sources), 1)))
    src_color = {s: c for s, c in zip(sources, cmap_src)}

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        subplot_kw=dict(polar=True),
        squeeze=False)

    for idx, label in enumerate(labels_with_cl):
        row = idx // ncols
        col = idx % ncols
        ax  = axes[row][col]
        arc = archetypes[label]

        # Archetype polygon
        arch_vals = arc["centroid"].tolist() + arc["centroid"][:1].tolist()
        vmax      = max(float(np.abs(arc["centroid"]).max()), 0.5)

        # Per-instance lines in CL space
        group = instances_by_label.get(label, [])
        for inst in group:
            cc = inst.get("cl_centroid")
            if cc is None:
                continue
            vmax = max(vmax, float(np.abs(cc).max()))
            inst_vals = cc.tolist() + cc[:1].tolist()
            ax.plot(angles, inst_vals,
                    color=src_color.get(inst["source"], "#AAAAAA"),
                    linewidth=1.0, alpha=0.40)

        # Archetype (mean)
        ax.plot(angles, arch_vals, color='#1F3864', linewidth=2.5, zorder=3)
        ax.fill(angles, arch_vals, color='#1F3864', alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(parameters, size=7)
        ax.set_ylim(-vmax, vmax)
        ax.plot(angles, [0.0] * (n_par + 1),
                color='#AAAAAA', lw=0.8, ls='--', zorder=0)

        coh    = label_coh.get(label, {})
        low_c  = coh.get("low_conf_count", 0) > 0
        aa     = coh.get("arch_align")
        aa_str = f"  align={aa:.2f}" if aa is not None else ""
        ax.set_title(f"{label}{aa_str}",
                     fontsize=8, fontweight='bold', pad=14,
                     color='#E05252' if low_c else '#1F3864')

    # Per-source legend (outside the subplots)
    handles = [plt.Line2D([0], [0], color=src_color[s], linewidth=2, label=s)
               for s in sources]
    handles.append(plt.Line2D([0], [0], color='#1F3864', linewidth=2.5,
                               label='Archetype (mean)'))
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 5), fontsize=8,
               bbox_to_anchor=(0.5, 0.01))

    for idx in range(len(labels_with_cl), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "F9 — CL-relative archetypes per label\n"
        "Axes = MRI parameters in CL z-scores  (0 = CL median, ±1 = 1 CL std)\n"
        "Bold dark = archetype (mean across animals)  |  Thin = individual instances",
        fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    path = os.path.join(output_dir, "F9_archetype_radar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# =============================================================================
# 6. PDF REPORT
# =============================================================================

def _write_pdf(results, all_instances, label_coh, sil,
               parameters, fig_paths, output_dir):
    pdf_path = os.path.join(output_dir, "comparison_report.pdf")

    fig_titles = [
        t("f1_title"), t("f2_title"), t("f3_title"), t("f4_title"),
        t("f5_title"), t("f6_title"), t("f7_title"), t("f8_title"),
        t("f9_title"),
    ]

    with PdfPages(pdf_path) as pdf:

        # --- Title page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.90, t("pdf_title"),
                ha='center', fontsize=24, fontweight='bold', color='#1F3864')

        # --- Study list with source files ---
        study_lines = []
        for r in results:
            files = r.get("files", [])
            if files:
                names  = [os.path.basename(f) for f in files]
                common = os.path.commonprefix(names).rstrip('-_.')
            else:
                common = ""
            id_str = f"  {common}" if common else ""
            study_lines.append(
                f"  {r['name']:<20}  {r['method'].upper():<8}  K = {r['best_k']}{id_str}")
        ax.text(0.5, 0.72, "\n".join(study_lines),
                ha='center', fontsize=9, color='#333333', fontfamily='monospace',
                va='top')

        n_lines  = len(study_lines)
        y_params = max(0.72 - n_lines * 0.028 - 0.04, 0.20)
        ax.text(0.5, y_params,
                t("pdf_params", p=', '.join(parameters)),
                ha='center', fontsize=10, color='#666666')
        ax.text(0.5, y_params - 0.06,
                f"{t('pdf_size_floor', n=SIZE_FLOOR)}   "
                f"{t('pdf_eta2_floor', n=ETA2_FLOOR)}   "
                f"{t('pdf_cosine_thr', n=COSINE_SIM_THRESHOLD)}",
                ha='center', fontsize=9, color='#888888')
        if sil is not None:
            ax.text(0.5, y_params - 0.12,
                    t("pdf_global_sil", v=sil),
                    ha='center', fontsize=10, color='#2E75B6', fontweight='bold')
        ax.text(0.5, 0.04,
                datetime.datetime.now().strftime("%B %d, %Y"),
                ha='center', fontsize=11, color='#AAAAAA')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Coherence summary page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        hdr  = (f"{'Label':<28}  {'r_cos':>6}  {'r_euc':>6}  {'wcos':>6}  {'euc':>6}"
                f"  {'n':>3}  {'pres.':>5}  {'%vol':>6}  {'CV%':>5}"
                f"  {'C1':>5}  {'arch↑':>6}  {'magCV':>6}  {'coherent':>8}")
        sep  = "─" * 100
        body = [hdr, sep]
        for label, coh in sorted(label_coh.items()):
            wcos_str = (f"{coh['mean_weighted_cos']:.3f}"
                        if coh["cos_sims"] else "  — ")
            skip_tag = (f"(sk{coh['n_cos_skipped']})" if coh["n_cos_skipped"] else "")
            euc_str  = (f"{np.mean(coh['euc_dists']):.3f}"
                        if coh["euc_dists"] else "  — ")
            c1_val   = 1.0 - coh["ratio"]
            flag     = (t("pdf_coherent_yes") if coh["coherent"] else t("pdf_coherent_no"))
            warn     = f"  ⚠×{coh['low_conf_count']}" if coh["low_conf_count"] else ""
            pres_str = f"{coh['n_sources_present']}/{coh['n_sources_total']}"
            vol_str  = f"{coh['mean_proportion']*100:5.1f}%"
            cv_str   = f"{coh['prop_cv']*100:4.0f}%"
            aa       = coh.get("arch_align")
            mc       = coh.get("mag_cv")
            aa_str   = f"{aa:>6.3f}" if aa is not None else "     —"
            mc_str   = f"{mc:>6.3f}" if mc is not None else "     —"
            body.append(
                f"{label:<28}  {coh['ratio']:>6.3f}  {coh['ratio_euc']:>6.3f}"
                f"  {wcos_str:>6}{skip_tag}  {euc_str:>6}"
                f"  {coh['n_instances']:>3}  {pres_str:>5}  {vol_str:>6}  {cv_str:>5}"
                f"  {c1_val:>5.3f}  {aa_str}  {mc_str}  {flag}{warn}")
            # Cross-source shape concordance
            for (sa, sb), ov in coh["overlap_map"].items():
                sc = f"scale={ov['scale_ratio']:.2f}×"
                if ov["near_origin"]:
                    tag = f"{t('pdf_near_origin')} ({sc})"
                elif ov["concordant"]:
                    tag = f"{t('pdf_concordant')} ({sc})"
                else:
                    tag = f"{t('pdf_divergent_shape')}{', '.join(ov['divergent_params'])}  ({sc})"
                body.append(f"    {sa} vs {sb} : {tag}")
            # Within-source divergence + per-cluster cross-source breakdown
            for src, divs in coh["within_divergent"].items():
                for d in divs:
                    params_str = ", ".join(d["entry"]["divergent_params"])
                    body.append(f"    ⚠ {src} H{d['cid_a']} vs H{d['cid_b']}"
                                f" : {t('pdf_internally_div')} ({params_str})")
                # Per-cluster comparisons to other sources
                for (src_div, cid, other_src), ov in coh["expanded_cross"].items():
                    if src_div != src:
                        continue
                    sc  = f"scale={ov['scale_ratio']:.2f}×"
                    if ov["near_origin"]:
                        tag = f"{t('pdf_near_origin')} ({sc})"
                    elif ov["concordant"]:
                        tag = f"{t('pdf_concordant')} ({sc})"
                    else:
                        tag = f"{t('pdf_divergent')}{', '.join(ov['divergent_params'])}  ({sc})"
                    body.append(f"      {src} H{cid} vs {other_src} : {tag}")
        ax.text(0.03, 0.94, "\n".join(body),
                va='top', fontsize=8.5, fontfamily='monospace',
                transform=ax.transAxes, color='#1C1C1E')
        ax.set_title(t("pdf_coherence_title"), fontsize=13,
                     fontweight='bold', color='#1F3864', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- One page per figure (F1/F4 may produce multiple pages) ---
        for entry_paths, base_title in zip(fig_paths, fig_titles):
            # Normalise: single path or list of paths
            if isinstance(entry_paths, list):
                page_list = entry_paths
            else:
                page_list = [entry_paths]
            n_pg = len(page_list)
            for pg_idx, path in enumerate(page_list):
                if not os.path.exists(path):
                    continue
                title = (f"{base_title}  ({pg_idx + 1}/{n_pg})"
                         if n_pg > 1 else base_title)
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                img = mpimg.imread(path)
                ax.imshow(img, aspect='auto')
                ax.set_title(title, fontsize=12, fontweight='bold',
                             color='#1F3864', pad=8)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"  PDF report : {pdf_path}")


# =============================================================================
# 6. ORCHESTRATOR
# =============================================================================

def run_comparison_v4(results, output_dir):
    """
    Full V4 comparison pipeline.

    results : list of {name, csv, method, best_k, out_dir}
    output_dir : where figures, JSON summary, and PDF are saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("  COMPARISON V4")
    print("=" * 70)

    # --- Common parameters ---
    param_sets = []
    for r in results:
        df    = pd.read_csv(r["csv"])
        found = [p for p in _CFG_PARAMS if p in df.columns]
        param_sets.append(set(found))

    base = [p for p in _CFG_PARAMS if all(p in ps for ps in param_sets)]
    common_params, _ = resolve_dti_parameters(base)
    if not common_params:
        print("  [!] No common parameters found — comparison aborted.")
        return
    print(f"\n  Common parameters : {common_params}")

    # --- Load instances ---
    all_instances = []
    for r in results:
        insts = _load_instances(r, common_params)
        for inst in insts:
            flags = []
            if inst["low_conf"]:  flags.append("low-conf")
            if inst["mismatch"]:  flags.append(f"mismatch(user={inst['user_label']!r})")
            if not inst["norm_ok"]: flags.append("norm-warn")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"  {r['name']:<18} H{inst['cluster_id']}  "
                  f"{inst['label']:<28}  n={inst['n_voxels']:>4}{flag_str}")
        all_instances.extend(insts)

    if len(all_instances) < 2:
        print("  [!] Fewer than 2 instances — comparison aborted.")
        return

    # --- Group by suggest_label ---
    instances_by_label = defaultdict(list)
    for inst in all_instances:
        instances_by_label[inst["label"]].append(inst)

    # --- CL-relative archetypes ---
    archetypes = _compute_archetypes(instances_by_label, common_params)
    if archetypes:
        print(f"\n  CL-relative archetypes computed for {len(archetypes)} label(s): "
              f"{list(archetypes.keys())}")
    else:
        print("\n  [!] No CL data available — archetype matching skipped "
              "(run_meta.json absent from result directories)")

    # --- Coherence metrics ---
    print("\n" + "=" * 70)
    print("  LABEL COHERENCE")
    print("=" * 70)
    label_coh = _label_coherence(instances_by_label, common_params)

    for label, coh in sorted(label_coh.items()):
        src_counts = {s: coh["sources"].count(s)
                      for s in dict.fromkeys(coh["sources"])}
        src_str    = ", ".join(f"{s}×{n}" for s, n in src_counts.items())
        cos_str    = (f"  cos_sim={np.mean(coh['cos_sims']):.3f}"
                      if coh["cos_sims"] else "  (singleton)")
        euc_str    = (f"  euc={np.mean(coh['euc_dists']):.3f}"
                      if coh["euc_dists"] else "")
        tag        = "COHERENT" if coh["coherent"] else "non-coherent"
        lc_warn    = f"  ⚠ {coh['low_conf_count']} low-conf" if coh["low_conf_count"] else ""
        aa         = coh.get("arch_align")
        aa_str     = f"  arch={aa:.3f}" if aa is not None else ""
        print(f"  {label:<30}  ratio={coh['ratio']:.3f}"
              f"{cos_str}{euc_str}{aa_str}  [{tag}]  [{src_str}]{lc_warn}")
        for (sa, sb), ov in coh["overlap_map"].items():
            tag_ov = ("concordant" if ov["concordant"]
                      else f"divergent CI on: {', '.join(ov['divergent_params'])}")
            print(f"    {sa} vs {sb} : {tag_ov}")

    sil = _centroid_silhouette(instances_by_label)
    if sil is not None:
        print(f"\n  Global silhouette (cosine, centroids) : {sil:.3f}")

    coherent = [l for l, c in label_coh.items() if c["coherent"]]
    orphans  = [l for l, c in label_coh.items() if c["n_instances"] == 1]
    print(f"\n  Coherent labels : {coherent}")
    print(f"  Orphan labels   : {orphans}")

    # --- Figures ---
    print("\n" + "=" * 70)
    print("  FIGURES")
    print("=" * 70)
    # 'User label' column only when at least one label was manually renamed
    show_user_label = any(i["mismatch"] for i in all_instances)

    f9_path = _fig_archetype_radar(archetypes, instances_by_label,
                                    common_params, label_coh, output_dir)
    fig_paths = [
        _fig_centroid_heatmap(all_instances, instances_by_label,
                               common_params, output_dir),
        _fig_cosine_matrix(all_instances, output_dir),
        _fig_dispersion_ratio(label_coh, output_dir),
        _fig_bootstrap_ci(all_instances, common_params, output_dir),
        _fig_qc_table(all_instances, output_dir, show_user_label=show_user_label),
        _fig_output_components(label_coh, instances_by_label, output_dir),
        _fig_radar_charts(all_instances, instances_by_label,
                           common_params, output_dir),
        _fig_segmentation_grid(results, instances_by_label, output_dir),
        f9_path,   # None when no CL data; _write_pdf skips missing paths
    ]
    for entry in fig_paths:
        for p in (entry if isinstance(entry, list) else [entry]):
            if p is not None:
                print(f"  {os.path.basename(p)}")

    # --- PDF ---
    _write_pdf(results, all_instances, label_coh, sil,
               common_params, fig_paths, output_dir)

    # --- JSON summary ---
    summary = {
        "generated_at"     : datetime.datetime.now().isoformat(),
        "runs"             : [{"name": r["name"], "method": r["method"],
                               "best_k": r["best_k"]} for r in results],
        "common_parameters": common_params,
        "thresholds"       : {"size_floor": SIZE_FLOOR,
                              "eta2_floor": ETA2_FLOOR,
                              "cosine_sim": COSINE_SIM_THRESHOLD},
        "silhouette"       : sil,
        "labels"           : {
            l: {"n_instances"  : c["n_instances"],
                "ratio"        : round(c["ratio"], 4),
                "intra_cosine" : round(c["intra_cosine"], 4),
                "inter_cosine" : round(c["inter_cosine"], 4),
                "coherent"     : c["coherent"],
                "low_conf"     : c["low_conf_count"],
                "C1"           : round(1.0 - c["ratio"], 4),
                "arch_align"   : round(c["arch_align"], 4) if c.get("arch_align") is not None else None,
                "mag_cv"       : round(c["mag_cv"], 4)     if c.get("mag_cv")     is not None else None,
                "cl_available" : c.get("cl_available", False)}
            for l, c in label_coh.items()
        },
    }
    json_path = os.path.join(output_dir, "comparison_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary : {json_path}")
