# =================================================================
# pipeline/labeling.py — Habitat suggestion and interactive labeling
# =================================================================

import json
import os
from collections import defaultdict
import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from config import (
    HABITAT_RULES,
    UNDETERMINED_THRESHOLD, HABITAT_COLORS_HEX,
    BIO_EXTENT_THRESHOLDS, BIO_EXTENT_SMALL_EXTENT_RATIO,
    RELATIVE_SIMILAR, RELATIVE_SMALLER, RELATIVE_MUCH_SMALLER,
    RELATIVE_BETWEEN_MARGIN, BULK_TUMOR_COVERAGE_MIN,
    MIN_RULE_CONFIDENCE, AMBIGUITY_MARGIN,
    HYPOX_DTOPO_PRIOR_THR,
)
from i18n import t

# Fallback palette used when no theme is passed (standalone / light mode)
_THEME_LIGHT = {
    "bg"         : "#F7F8FA",
    "bg_card"    : "#FFFFFF",
    "bg_input"   : "#FFFFFF",
    "bg_cluster" : "#F0F2F5",
    "accent"     : "#3A6EA5",
    "text"       : "#1C1C1E",
    "text_sub"   : "#6B7280",
    "text_mono"  : "#4B5563",
    "separator"  : "#DDE1E7",
    "font_family": "Helvetica",
}


def _inject_virtual_md(centroid_dict):
    """
    If DTI-AD and DTI-RD are present but DTI-MD is not, compute
    virtual MD = (AD + 2*RD) / 3 and inject it.
    Returns a (possibly augmented) copy of centroid_dict.
    """
    cd = centroid_dict
    if ('DTI-AD' in cd and 'DTI-RD' in cd and 'DTI-MD' not in cd):
        cd = dict(cd)
        cd['DTI-MD'] = (cd['DTI-AD'] + 2.0 * cd['DTI-RD']) / 3.0
    return cd


def _suggest_habitat(centroid_dict, parameters,
                     undetermined_thresh=UNDETERMINED_THRESHOLD,
                     dti_mode='MD'):
    """
    Pass-1 habitat suggestion from centroid robust-scaled values.
    Returns (name, description, confidence, is_ambiguous).

    V5 scoring (continuous, [0, 1]):
      - ALL required conditions pass → score = 0.60 + 0.40 * sup_ratio
      - Partial required pass       → score = 0.30 * req_ratio + 0.10 * sup_ratio
      - Excluded                    → score = 0.0
    A rule wins only when its score >= MIN_RULE_CONFIDENCE (0.55 by default),
    which effectively requires all required conditions to be met.
    When no rule wins, the cluster defaults to "Bulk tumor" by exclusion —
    more faithful than forcing it through Unknown → Pass-2 heuristics.
    is_ambiguous is True when the top-2 rule scores are within AMBIGUITY_MARGIN.

    When dti_mode='AD_RD', virtual DTI-MD = (AD + 2*RD) / 3 is injected
    so all rules operate on a consistent diffusivity metric.
    """
    cd    = _inject_virtual_md(centroid_dict)
    rules = HABITAT_RULES

    def _condition_met(val, lo, hi):
        return (lo is None or val >= lo) and (hi is None or val <= hi)

    rule_scores = []   # list of (name, score, desc)

    for rule in rules:
        # must_have: if any key parameter is absent, rule cannot apply
        if any(p not in cd for p in rule.get("must_have", [])):
            rule_scores.append((rule["name"], 0.0, rule["description"]))
            continue

        # Simple exclude: ANY single condition triggers exclusion (OR logic)
        excluded = any(
            p in cd and _condition_met(cd[p], lo, hi)
            for p, (lo, hi) in rule.get("exclude", {}).items()
        )
        # Compound exclude: each dict requires ALL its conditions to be met (AND within, OR between)
        compound_excluded = any(
            all(p in cd and _condition_met(cd[p], lo, hi)
                for p, (lo, hi) in conds.items())
            for conds in rule.get("compound_exclude", [])
        )
        if excluded or compound_excluded:
            rule_scores.append((rule["name"], 0.0, rule["description"]))
            continue

        req_total, req_met = 0, 0
        for p, (lo, hi) in rule.get("required", {}).items():
            if p not in cd:
                continue
            req_total += 1
            if _condition_met(cd[p], lo, hi):
                req_met += 1

        sup_total, sup_met = 0, 0
        for p, (lo, hi) in rule.get("support", {}).items():
            if p not in cd:
                continue
            sup_total += 1
            if _condition_met(cd[p], lo, hi):
                sup_met += 1

        req_ratio = req_met / req_total if req_total > 0 else 0.0
        sup_ratio = sup_met / sup_total if sup_total > 0 else 0.0

        if req_ratio >= 1.0:
            score = 0.60 + 0.40 * sup_ratio
        else:
            score = 0.30 * req_ratio + 0.10 * sup_ratio

        rule_scores.append((rule["name"], score, rule["description"]))

    rule_scores.sort(key=lambda x: x[1], reverse=True)

    top_name, top_score, top_desc = (
        rule_scores[0] if rule_scores else ("Bulk tumor", 0.0, ""))

    # Ambiguity: two rules both above threshold and close to each other
    is_ambiguous = False
    if len(rule_scores) >= 2 and top_score >= MIN_RULE_CONFIDENCE:
        second_score = rule_scores[1][1]
        if second_score >= MIN_RULE_CONFIDENCE and \
                abs(top_score - second_score) < AMBIGUITY_MARGIN:
            is_ambiguous = True

    if top_score >= MIN_RULE_CONFIDENCE:
        best_name = top_name
        best_desc = top_desc
        if top_score >= 0.90:
            best_conf = "high"
        elif top_score >= 0.60:   # 0.60 = all-required, no support → medium (was 0.65)
            best_conf = "medium"
        else:
            best_conf = "low"
    else:
        best_name = "Bulk tumor"
        if top_score > 0.0:
            best_desc = (
                f"no rule above threshold — bulk tumor by exclusion  "
                f"[nearest: {top_name}, score={top_score:.2f}]"
            )
        else:
            best_desc = "no distinctive signature — bulk tumor by exclusion"
        best_conf = "high" if top_score < 0.30 else "medium"

    # Infiltrative is mechanistically rare (requires preserved WM fibres at
    # tumour margin).  Cap to low-confidence regardless of rule score so it
    # is never promoted by default — the user must explicitly confirm it.
    if best_name == "Infiltrative":
        best_conf = "low"

    return best_name, best_desc, best_conf, is_ambiguous, rule_scores


# ---------------------------------------------------------------------------
# Relative comparison helpers (used by classify_three_passes)
# ---------------------------------------------------------------------------

def _rel_similar(val, ref):
    return abs(val - ref) < RELATIVE_SIMILAR

def _rel_smaller(val, ref):
    return val < ref - RELATIVE_SMALLER

def _rel_much_smaller(val, ref):
    return val < ref - RELATIVE_MUCH_SMALLER

def _rel_between(val, ref_a, ref_b):
    """True if val is strictly between ref_a and ref_b (with margin)."""
    lo, hi = min(ref_a, ref_b), max(ref_a, ref_b)
    m = RELATIVE_BETWEEN_MARGIN
    return (hi - lo) > 2 * m and lo + m < val < hi - m


# ---------------------------------------------------------------------------
# Three-pass classification
# ---------------------------------------------------------------------------

def classify_three_passes(centroids, parameters, features_by_cluster,
                           dti_mode='MD', zone_map=None):
    """
    Full 3-pass habitat classification.

    Parameters
    ----------
    centroids           : {cluster_id (int): {param: robust-scaled value}}
    parameters          : list of parameter names
    features_by_cluster : {cluster_id: np.ndarray shape (n_vox, n_params)}
    dti_mode            : 'MD' or 'AD_RD'
    zone_map            : {cluster_id: 'core' | 'periphery'} or None.
                          When provided (hierarchical mode), biologically
                          impossible labels are replaced with Bulk tumor:
                            • core cluster → Infiltrative is impossible
                              (infiltrating WM fibres are not at the centre)
                            • periphery cluster → Necrosis is impossible
                              (necrosis is a core-zone event)
                          The original Pass-1 suggestion is preserved in
                          'label_p1' for traceability.

    Returns
    -------
    dict {cluster_id: {
        'label'          : str   — final label (after all 3 passes)
        'label_p1'       : str   — pass-1 absolute label
        'label_p2'       : str   — pass-2 label (after reclassification)
        'confidence'     : str   — 'high' / 'medium' / 'low'
        'desc'           : str   — rule description
        'reclass_note'   : str | None
        'subcat_note'    : str | None
        'anchor_necrotic': int | None
        'anchor_median'  : int
    }}
    """
    # ── Inject virtual MD = (AD + 2×RD) / 3 when MD was excluded ────
    # Ensures all three passes and anchor scoring see a properly-scaled MD.
    if dti_mode == 'AD_RD':
        centroids = {c: _inject_virtual_md(v) for c, v in centroids.items()}

    # ── Pass 1 : absolute rule-based classification ───────────────────
    p1 = {}
    for c, centroid in centroids.items():
        lbl, desc, conf, is_amb, rscores = _suggest_habitat(
            centroid, parameters, dti_mode=dti_mode)
        p1[c] = {'label': lbl, 'label_raw': lbl, 'desc': desc, 'desc_raw': desc,
                 'confidence': conf, 'is_ambiguous': is_amb, 'zone_note': None,
                 'rule_scores': rscores}

    # ── Zone constraint (hierarchical mode) ──────────────────────────
    # Applied immediately after Pass 1 so that relative comparisons in
    # Pass 2 start from a biologically coherent state.
    # label_raw / desc_raw are preserved so the dialog can show the original
    # Pass-1 suggestion alongside the zone-constrained final label.
    # Instead of forcing "Bulk tumor", pick the next-best rule above threshold
    # that is not the biologically forbidden label.
    if zone_map:
        for c in list(p1.keys()):
            zone     = zone_map.get(c)
            lbl      = p1[c]['label']
            forbidden = None
            if zone == 'core' and lbl == 'Infiltrative':
                forbidden = 'Infiltrative'
                reason    = 'infiltrating WM fibres absent in tumour core'
            elif zone == 'periphery' and lbl == 'Necrosis':
                forbidden = 'Necrosis'
                reason    = 'necrosis is a core-zone event'
            if forbidden is not None:
                next_lbl = next(
                    (name for name, score, _d in p1[c]['rule_scores']
                     if score >= MIN_RULE_CONFIDENCE and name != forbidden),
                    'Bulk tumor'
                )
                p1[c]['label']     = next_lbl
                p1[c]['zone_note'] = (
                    f'zone-constrained [{zone}]: {lbl} -> {next_lbl} ({reason})'
                )

    # ── Anchor identification ─────────────────────────────────────────
    n_total = sum(len(f) for f in features_by_cluster.values()) or 1

    def _l2(c):
        return float(np.linalg.norm(
            [centroids[c].get(p, 0.0) for p in parameters]))

    # Median anchor: smallest L2 among clusters with >30% coverage;
    # fallback to global L2 minimum.
    anchor_median = None
    for c in sorted(centroids, key=_l2):
        if len(features_by_cluster[c]) / n_total > BULK_TUMOR_COVERAGE_MIN:
            anchor_median = c
            break
    if anchor_median is None:
        anchor_median = min(centroids, key=_l2)

    coverage_anchor = len(features_by_cluster[anchor_median]) / n_total
    bulk_label = ("Bulk tumor" if coverage_anchor > BULK_TUMOR_COVERAGE_MIN
                  else "Bulk tumor (atypical)")

    # Necrotic anchor: highest MD − T2* among zone-constrained Pass-1 Necrosis clusters.
    necrotic_ids = [c for c, d in p1.items() if d['label'] == "Necrosis"]

    def _nec_score(c):
        cd = centroids[c]
        if 'DTI-MD' in cd:
            md = cd['DTI-MD']
        elif 'DTI-AD' in cd and 'DTI-RD' in cd:
            md = (cd['DTI-AD'] + 2.0 * cd['DTI-RD']) / 3.0
        elif 'DTI-AD' in cd:
            md = cd['DTI-AD']
        else:
            md = 0.0
        return md - cd.get('T2star', 0.0)

    anchor_necrotic = (max(necrotic_ids, key=_nec_score)
                       if necrotic_ids else None)

    # Anchor guard: only activate Pass-2 necrosis reclassification when the
    # anchor cluster was labelled Necrosis with HIGH confidence (score ≥ 0.90,
    # i.e. all required + ≥75% support conditions met).  A medium-confidence
    # anchor is unreliable and propagates drift-towards-necrosis into
    # neighbouring clusters.
    if (anchor_necrotic is not None and
            p1[anchor_necrotic]['confidence'] != 'high'):
        anchor_necrotic = None

    med_c = centroids[anchor_median]
    nec_c = centroids[anchor_necrotic] if anchor_necrotic is not None else None

    # ── Per-parameter normalisation [0,1] for relative comparisons ────
    # Prevents MD (large raw IQR) from dominating distance / delta checks
    # in passes 2 and 3.  Pass-1 rules still use raw robust-scaled values.
    all_c = list(centroids)
    norm_c = {}                         # norm_c[cluster_id][param] ∈ [0, 1]
    for p in parameters:
        vals = [centroids[c].get(p, 0.0) for c in all_c]
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        for c in all_c:
            norm_c.setdefault(c, {})[p] = (
                (centroids[c].get(p, 0.0) - mn) / rng if rng > 1e-8 else 0.5)

    norm_med = norm_c[anchor_median]
    norm_nec = norm_c[anchor_necrotic] if anchor_necrotic is not None else None

    # ── Pass 2 : relative reclassification ───────────────────────────
    # All delta / similar checks use NORMALISED values [0,1].
    p2 = {c: p1[c]['label'] for c in centroids}
    reclass_notes = {c: None for c in centroids}

    # Anchor median is always relabelled Bulk tumor (it IS the tumour reference).
    # Record a note if Pass-1 had assigned something more specific.
    p1_med_lbl = p1[anchor_median]['label']
    if p1_med_lbl not in ('Bulk tumor', 'Bulk tumor (atypical)'):
        reclass_notes[anchor_median] = (
            f"pass-1 was '{p1_med_lbl}' but smallest-L2 cluster is median tumour "
            f"reference — relabelled {bulk_label}"
        )
    p2[anchor_median] = bulk_label

    # ── Pass 2a : hypoxic penumbra spatial prior ─────────────────────
    # Fires independently of the necrotic anchor.
    # A cluster labelled Hypoxic/vascular that sits far from the tumour
    # core (d_topo_norm > HYPOX_DTOPO_PRIOR_THR) is outside the expected
    # penumbra zone → demote to Bulk tumor.
    # Gracefully skipped when d_topo was not computed (d_topo_norm absent).
    for c in centroids:
        if c == anchor_median:
            continue
        if p2[c] != "Hypoxic/vascular":
            continue
        dtopo_raw = centroids[c].get("d_topo_norm")
        if dtopo_raw is not None and dtopo_raw > HYPOX_DTOPO_PRIOR_THR:
            p2[c] = bulk_label
            reclass_notes[c] = t("reclass_hypox_far_periphery",
                                  dtopo=dtopo_raw)

    if norm_nec is not None:
        for c in centroids:
            if c in (anchor_necrotic, anchor_median):
                continue
            lbl = p2[c]
            nc  = norm_c[c]
            fa  = "DTI-FA"
            md  = "DTI-MD"
            t2  = "T2"

            if lbl == "Infiltrative":
                # In normed space, infiltrative has HIGH FA (top of range).
                # Reclassify → Necrosis if FA collapses to bottom AND MD is high.
                if (fa in nc and md in nc):
                    fa_low  = _rel_much_smaller(nc[fa], norm_med.get(fa, 0.5))
                    md_high = nc.get(md, 0.0) > norm_med.get(md, 0.5) + 0.3
                    if fa_low and md_high:
                        p2[c] = "Necrosis"
                        reclass_notes[c] = t("reclass_fa_low_md_high",
                                             fa=nc[fa], md=nc[md])

            elif lbl == "Hypoxic/vascular":
                if (md in nc and t2 in nc):
                    if (_rel_similar(nc[md], norm_nec[md]) and
                            _rel_similar(nc[t2], norm_nec[t2])):
                        p2[c] = "Necrosis"
                        reclass_notes[c] = t("reclass_md_t2_near_nec",
                                             md=nc[md], t2=nc[t2])

            elif lbl == "Necrosis" and c != anchor_necrotic:
                # MD between anchors + FA not collapsed → infiltrative
                if (md in nc and fa in nc):
                    if (_rel_between(nc[md], norm_med.get(md, 0.0),
                                     norm_nec.get(md, 1.0)) and
                            nc[fa] > norm_nec.get(fa, 0.0) + 0.3):
                        p2[c] = "Infiltrative"
                        reclass_notes[c] = t("reclass_md_intermediate",
                                             md=nc[md])

    # ── Pass 3 : within-category sub-classification ───────────────────
    # Uses normalised values so that all parameters have equal weight.
    p3 = dict(p2)
    subcat_notes = {c: None for c in centroids}

    for c in centroids:
        lbl = p2[c]
        cd  = centroids[c]    # raw values (for display in notes only)
        nc  = norm_c[c]       # normalised [0,1] (for all comparisons)

        if "Necrosis" in lbl:
            # Pass 3 sub-classification uses raw robust-scaled values (IQR space)
            # because RELATIVE_* constants are calibrated in those units.
            # (Pass 2 uses normed [0,1] space; same constant values are forgiving
            # by design — they should be tighter if Pass 2 over-fires.)
            t2s   = cd.get("T2star")
            mtr   = cd.get("MTR")
            m_t2s = med_c.get("T2star", 0.0)
            m_mtr = med_c.get("MTR",    0.0)

            # Priority: hemorrhagic (T2* very low) > proteic (MTR high) > cystic (MTR low)
            if t2s is not None and _rel_much_smaller(t2s, m_t2s):
                p3[c] = "Necrosis (hemorrhagic)"
                subcat_notes[c] = t("subcat_t2star_low", val=t2s, ref=m_t2s)
            elif mtr is not None:
                if mtr > m_mtr + RELATIVE_SMALLER:
                    p3[c] = "Necrosis (proteic)"
                    subcat_notes[c] = t("subcat_mtr_high", val=mtr, ref=m_mtr)
                elif _rel_much_smaller(mtr, m_mtr):
                    p3[c] = "Necrosis (cystic)"
                    subcat_notes[c] = t("subcat_mtr_low", val=mtr, ref=m_mtr)

        elif lbl == "Infiltrative":
            # Jellison sub-classification within the FA+ zone.
            # FA is now POSITIVE for infiltrative (above tumour median).
            # Position: 0 = at tumour median FA, 1 = maximum FA in this tumour.
            fa = "DTI-FA"
            if fa in nc:
                fa_pos = nc[fa]   # normalised: 0 = lowest FA cluster, 1 = highest
                # Find normalised FA of median anchor to anchor "0"
                fa_med_norm = norm_med.get(fa, 0.5)
                # Position above median in the positive FA zone
                above_range = 1.0 - fa_med_norm
                if above_range > 0.05:
                    pos = (fa_pos - fa_med_norm) / above_range   # 0=median, 1=max FA
                    if pos < 0.5:
                        p3[c] = "Infiltrative (edematous)"
                        subcat_notes[c] = t("subcat_fa_moderate", pos=pos)
                    else:
                        p3[c] = "Infiltrative (infiltrated)"
                        subcat_notes[c] = t("subcat_fa_high", pos=pos)

    # ── Build result dict ─────────────────────────────────────────────
    return {
        c: {
            'label'          : p3[c],
            'label_p1'       : p1[c]['label_raw'],   # original before zone constraint
            'label_p2'       : p2[c],
            'confidence'     : p1[c]['confidence'],
            'is_ambiguous'   : p1[c]['is_ambiguous'],
            'desc'           : p1[c]['desc_raw'],    # original rule description
            'zone_note'      : p1[c]['zone_note'],   # non-None when constraint fired
            'reclass_note'   : reclass_notes[c],
            'subcat_note'    : subcat_notes[c],
            'anchor_necrotic': anchor_necrotic,
            'anchor_median'  : anchor_median,
        }
        for c in centroids
    }


def _compute_bio_extent(raw_c, cl_stats, scaling_info, parameters):
    """
    Bio extent score per parameter:
      score = (raw_value − CL_median) / (tumor_median − CL_median)
      0 = at CL level   1 = at tumor median   >1 = beyond tumor median

    When the CL-to-tumor extent is < BIO_EXTENT_SMALL_EXTENT_RATIO of the
    CL median, the parameter is not discriminative: score is forced to 1.0
    (at tumor median level) and the parameter name is added to small_ext
    for a GUI warning.

    Returns (scores_dict, small_ext_list).
    """
    scores    = {}
    small_ext = []
    for p in parameters:
        if p not in cl_stats or p not in raw_c or p not in scaling_info:
            continue
        cl_med = cl_stats[p]['median']
        tu_med = scaling_info[p]['median']
        extent = tu_med - cl_med
        ref    = max(abs(cl_med), 1e-8)
        if abs(extent) / ref < BIO_EXTENT_SMALL_EXTENT_RATIO:
            scores[p] = 1.0
            small_ext.append(p)
        else:
            scores[p] = float((raw_c[p] - cl_med) / extent)
    return scores, small_ext


def _interpret_bio_extent(bio_scores, robust_centroid, parameters):
    """
    Generate an interpretation sentence and a suggested label from bio extent scores.
    Returns (interp_text, suggested_label) or (None, None) if no scores.
    """
    if not bio_scores:
        return None, None

    mean_score = float(np.mean(list(bio_scores.values())))
    thresh     = BIO_EXTENT_THRESHOLDS

    if mean_score < thresh['healthy']:
        label = t("bio_near_normal")
        level = t("bio_level_healthy", score=mean_score)
    elif mean_score < thresh['infiltrative']:
        label = t("bio_infiltrative_lbl")
        level = t("bio_level_infiltrative", score=mean_score)
    elif mean_score < thresh['viable']:
        label = t("bio_active_tumor")
        level = t("bio_level_viable", score=mean_score)
    elif mean_score < thresh['aggressive']:
        label = t("bio_aggressive")
        level = t("bio_level_aggressive", score=mean_score)
    else:
        label = t("bio_necrosis_lbl")
        level = t("bio_level_necrosis", score=mean_score)

    # Direction from robust scaling: top-3 most deviant parameters
    deviant = sorted(
        [(p, robust_centroid.get(p, 0.0)) for p in parameters],
        key=lambda x: abs(x[1]), reverse=True
    )[:3]
    dirs = [f"{p}{'↑' if v > 0 else '↓'}" for p, v in deviant if abs(v) > 0.15]
    if dirs:
        level += "  —  " + "  ".join(dirs)

    return level, label




def _show_info_popup(title, body, theme):
    """Open a small modal info popup."""
    T  = theme
    ff = T["font_family"]
    popup = tk.Toplevel()
    popup.title(title)
    popup.configure(bg=T["bg"])
    popup.resizable(False, False)
    popup.grab_set()
    popup.lift()
    popup.attributes('-topmost', True)
    tk.Label(popup,
             text=body,
             font=("Courier", 9),
             bg=T["bg"], fg=T["text"],
             justify='left',
             padx=20, pady=14,
             ).pack()
    tk.Button(popup,
              text=t("close"),
              font=(ff, 9),
              bg=T["accent"], fg="#FFFFFF",
              activebackground=T.get("accent_dark", T["accent"]),
              activeforeground="#FFFFFF",
              relief="flat", bd=0,
              padx=16, pady=5,
              cursor="hand2",
              command=popup.destroy,
              ).pack(pady=(0, 12))


def _label_dialog(centroids, parameters, best_k, colors,
                  theme=None, cl_stats=None, raw_centroids=None,
                  scaling_info=None, dti_mode='MD',
                  three_pass_results=None, hier_meta=None):
    """
    Tkinter modal window for interactive habitat labeling.
    MUST be called from the main thread only.
    Scrollable with mouse-wheel support; Confirm button fixed at bottom.
    Press Enter or click Confirm to validate.

    theme        : THEME dict from app.py. None → light fallback.
    cl_stats     : {param: {'mean','std','median','n'}} from load_cl_stats().
    raw_centroids: {cluster_id: {param: raw_value}} in original unscaled space.
    scaling_info : {param: {'median', 'iqr'}} from robust_scale().
    dti_mode     : 'AD_RD' or 'MD' — selects the habitat rule set.
    hier_meta    : dict from hierarchical_segment() or None.
                   When provided, cluster cards are grouped as Core / Periphery
                   and spatial mismatches are flagged.
    When cl_stats + raw_centroids + scaling_info are all provided, three
    analysis levels are shown per cluster (vs tumor / bio extent / interpretation).
    """
    T        = theme if theme is not None else _THEME_LIGHT
    ff       = T["font_family"]
    show_bio = bool(cl_stats and raw_centroids is not None and scaling_info)

    # Hierarchical zone maps (empty when not in hier mode)
    _core_ids  = set(hier_meta['core_cluster_ids'])   if hier_meta else set()
    _periph_ids = set(hier_meta['periphery_cluster_ids']) if hier_meta else set()
    _hier_active = bool(hier_meta and hier_meta.get('hierarchical_valid'))

    # Labels that are spatially inconsistent with a zone
    _core_ok_labels    = {"Necrosis", "Hypoxic/vascular", "Cellular/proliferative",
                          "Bulk tumor"}
    _periph_ok_labels  = {"Cellular/proliferative", "Infiltrative", "Hypoxic/vascular",
                          "Bulk tumor"}

    result  = {}
    entries = {}

    win = tk.Toplevel()
    win.title(t("label_dlg_title"))
    win.geometry("720x540")
    win.configure(bg=T["bg"])
    win.grab_set()
    win.lift()
    win.attributes('-topmost', True)

    main_frame = tk.Frame(win, bg=T["bg"])
    main_frame.pack(fill='both', expand=True)

    canvas = tk.Canvas(main_frame, bg=T["bg"], highlightthickness=0, bd=0)
    scrollbar = ttk.Scrollbar(main_frame, orient='vertical',
                               command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg=T["bg"])

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    def _on_mousewheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    tk.Label(scroll_frame,
             text=t("label_dlg_header"),
             font=(ff, 13, "bold"),
             bg=T["bg"], fg=T["text"],
             ).pack(pady=(16, 4), padx=16)

    subtitle = (
        t("label_dlg_subtitle_bio")
        if show_bio else
        t("label_dlg_subtitle_simple")
    )
    tk.Label(scroll_frame,
             text=subtitle,
             font=(ff, 9),
             bg=T["bg"], fg=T["text_sub"],
             ).pack(padx=16, pady=(0, 4))

    if show_bio:
        info_row = tk.Frame(scroll_frame, bg=T["bg"])
        info_row.pack(pady=(0, 10))
        tk.Button(info_row,
                  text=t("label_vs_tumor"),
                  font=(ff, 8),
                  bg=T["accent"], fg="#FFFFFF",
                  activebackground=T.get("accent_dark", T["accent"]),
                  activeforeground="#FFFFFF",
                  relief="flat", bd=0,
                  padx=8, pady=2,
                  cursor="hand2",
                  command=lambda: _show_info_popup(
                      t("info_vs_tumor_title"), t("vs_tumor_info"), T),
                  ).pack(side='left', padx=(0, 8))
        tk.Button(info_row,
                  text=t("label_bio_extent"),
                  font=(ff, 8),
                  bg=T["accent"], fg="#FFFFFF",
                  activebackground=T.get("accent_dark", T["accent"]),
                  activeforeground="#FFFFFF",
                  relief="flat", bd=0,
                  padx=8, pady=2,
                  cursor="hand2",
                  command=lambda: _show_info_popup(
                      t("info_bio_title"), t("bio_extent_info"), T),
                  ).pack(side='left')
    else:
        tk.Label(scroll_frame, bg=T["bg"]).pack(pady=(0, 10))

    outer = tk.Frame(scroll_frame, bg=T["bg"])
    outer.pack(fill='both', expand=True, padx=16)

    # ── Pass 1 : compute all cluster data ────────────────────────────
    tpr = three_pass_results or {}   # shorthand
    cdata = []
    for c in range(best_k):
        if tpr and c in tpr:
            # 3-pass results provided: use pre-computed labels
            suggestion    = tpr[c]['label_p1']
            desc          = tpr[c]['desc']
            confidence    = tpr[c]['confidence']
            is_amb        = tpr[c].get('is_ambiguous', False)
            default_label = tpr[c]['label']   # final p3 label
            reclass_note  = tpr[c].get('reclass_note')
            subcat_note   = tpr[c].get('subcat_note')
            zone_note     = tpr[c].get('zone_note')
        else:
            suggestion, desc, confidence, is_amb, _ = _suggest_habitat(
                centroids[c], parameters, dti_mode=dti_mode)
            default_label = suggestion
            reclass_note  = None
            subcat_note   = None
            zone_note     = None

        bio_scores, small_ext = (
            _compute_bio_extent(
                raw_centroids.get(c, {}), cl_stats, scaling_info, parameters)
            if show_bio else ({}, [])
        )
        interp_text, bio_label = _interpret_bio_extent(bio_scores, centroids[c], parameters)
        mean_score = float(np.mean(list(bio_scores.values()))) if bio_scores else 0.0

        cdata.append({
            'suggestion'   : suggestion,
            'desc'         : desc,
            'confidence'   : confidence,
            'is_ambiguous' : is_amb,
            'bio_scores'   : bio_scores,
            'small_ext'    : small_ext,
            'interp_text'  : interp_text,
            'bio_label'    : bio_label,
            'mean_score'   : mean_score,
            'default_label': default_label,
            'reclass_note' : reclass_note,
            'subcat_note'  : subcat_note,
            'zone_note'    : zone_note,
        })

    # ── Pass 2a : disambiguate duplicate final labels ─────────────────
    # Per-parameter min-max norm across ALL clusters so MD's large raw range
    # does not dominate the "most discriminant parameter" selection.
    all_cids = list(range(best_k))
    _p_min = {p: min(centroids[c].get(p, 0.0) for c in all_cids) for p in parameters}
    _p_max = {p: max(centroids[c].get(p, 0.0) for c in all_cids) for p in parameters}
    _p_rng = {p: _p_max[p] - _p_min[p] for p in parameters}

    def _norm_val(cid, p):
        rng = _p_rng.get(p, 0.0)
        return (centroids[cid].get(p, 0.0) - _p_min[p]) / rng if rng > 1e-8 else 0.5

    label_groups = defaultdict(list)
    for c, d in enumerate(cdata):
        label_groups[d['default_label']].append(c)

    for base_label, group in label_groups.items():
        if len(group) < 2:
            continue
        best_p = max(
            parameters,
            key=lambda p: float(np.var([_norm_val(c, p) for c in group]))
        )
        sorted_group = sorted(group, key=lambda c: _norm_val(c, best_p))
        n = len(sorted_group)
        if n == 2:
            suffixes = [f" ({best_p}v)", f" ({best_p}^)"]
        elif n == 3:
            suffixes = [f" ({best_p} low)", f" ({best_p} mid)", f" ({best_p} high)"]
        else:
            suffixes = [f" ({best_p} {i+1}/{n})" for i in range(n)]
        for i, c in enumerate(sorted_group):
            cdata[c]['default_label'] += suffixes[i]

    # ── Pass 2b : mark most aggressive cluster ────────────────────────
    most_agg = None
    if tpr:
        most_agg = next(iter(tpr.values())).get('anchor_necrotic')
    elif show_bio:
        scored = [(c, cdata[c]['mean_score'])
                  for c in range(best_k) if cdata[c]['bio_scores']]
        if scored:
            most_agg = max(scored, key=lambda x: x[1])[0]
    if most_agg is not None:
        cdata[most_agg]['default_label'] += " +"

    # ── Pass 3 : render UI ────────────────────────────────────────────
    # When hier_meta is active, show zone separators (Core first, then Periphery).
    # Hierarchical labels guarantee core IDs come before periphery IDs in range().
    _zone_header_shown = {'core': False, 'periph': False}

    def _zone_separator(label_text, color):
        row = tk.Frame(outer, bg=T["bg"])
        row.pack(fill='x', pady=(10, 2))
        tk.Label(row, text=label_text,
                 font=(ff, 9, "bold"), bg=T["bg"], fg=color,
                 ).pack(side='left', padx=4)
        tk.Frame(row, height=1, bg=color).pack(side='left', fill='x', expand=True, padx=4)

    for c in range(best_k):
        d             = cdata[c]
        suggestion    = d['suggestion']
        desc          = d['desc']
        confidence    = d['confidence']
        is_amb        = d.get('is_ambiguous', False)
        bio_scores    = d['bio_scores']
        small_ext     = d['small_ext']
        interp_text   = d['interp_text']
        bio_label     = d['bio_label']
        mean_score    = d['mean_score']
        default_label = d['default_label']
        reclass_note  = d.get('reclass_note')
        subcat_note   = d.get('subcat_note')
        zone_note     = d.get('zone_note')

        # ── Hierarchical zone separators ─────────────────────────────
        if _hier_active:
            if c in _core_ids and not _zone_header_shown['core']:
                crit_txt = hier_meta.get('l1_criterion', '')
                _zone_separator(
                    f"── Core zone  ({crit_txt})", "#C0392B")
                _zone_header_shown['core'] = True
            elif c in _periph_ids and not _zone_header_shown['periph']:
                _zone_separator("── Periphery zone", "#27AE60")
                _zone_header_shown['periph'] = True

        bloc = tk.Frame(outer, bg=T["bg_cluster"],
                        highlightthickness=1,
                        highlightbackground=(
                            "#E07020" if zone_note else T["separator"]))
        bloc.pack(fill='x', pady=6)

        title_bar = tk.Frame(bloc, bg=T["bg_card"])
        title_bar.pack(fill='x')

        dot = tk.Canvas(title_bar, width=14, height=14,
                        bg=T["bg_card"], highlightthickness=0)
        dot.create_oval(1, 1, 13, 13, fill=colors[c], outline="")
        dot.pack(side='left', padx=(10, 6), pady=8)

        tk.Label(title_bar,
                 text=t("label_cluster", n=c),
                 font=(ff, 10, "bold"),
                 bg=T["bg_card"], fg=T["text"],
                 ).pack(side='left', pady=8)

        # Zone badge (Core / Periphery)
        if _hier_active:
            zone_lbl = "Core" if c in _core_ids else "Periph"
            zone_clr = "#C0392B" if c in _core_ids else "#27AE60"
            tk.Label(title_bar,
                     text=f"  [{zone_lbl}]",
                     font=(ff, 8, "bold"),
                     bg=T["bg_card"], fg=zone_clr,
                     ).pack(side='left', pady=8)

        if c == most_agg:
            tk.Label(title_bar,
                     text=t("label_most_agg"),
                     font=(ff, 8),
                     bg=T["bg_card"], fg="#E05252",
                     ).pack(side='right', padx=10, pady=8)

        # --- Line 1 : vs tumor (robust-scaled) ---
        robust_parts = []
        for p in parameters:
            val   = centroids[c][p]
            arrow = "▲" if val > 0.3 else ("▼" if val < -0.3 else "≈")
            robust_parts.append(f"{p}={val:+.2f}{arrow}")

        line1_pady = (8, 2) if show_bio else (8, 8)
        tk.Label(bloc,
                 text=t("label_vs_tumor_row") + "   ".join(robust_parts),
                 font=("Courier", 8),
                 bg=T["bg_cluster"], fg=T["text_mono"],
                 anchor='w',
                 ).pack(fill='x', padx=10, pady=line1_pady)

        if show_bio:
            # --- Line 2 : bio extent (0→1) ---
            bio_parts = []
            for p in parameters:
                if p in bio_scores:
                    bio_parts.append(f"{p}={bio_scores[p]:+.1f}")
                else:
                    bio_parts.append(f"{p}=—")

            mean_str    = f"   |  mean={mean_score:.2f}" if bio_scores else ""
            warn_suffix = ("   " + t("label_warn_small") + ", ".join(small_ext)) if small_ext else ""
            tk.Label(bloc,
                     text=t("label_bio_row") + "   ".join(bio_parts) + mean_str + warn_suffix,
                     font=("Courier", 8),
                     bg=T["bg_cluster"], fg=T["accent"],
                     anchor='w',
                     ).pack(fill='x', padx=10, pady=(0, 2))

            # --- Line 3 : interpretation ---
            if mean_score < BIO_EXTENT_THRESHOLDS['healthy']:
                interp_color = "#5CB85C"
            elif mean_score < BIO_EXTENT_THRESHOLDS['viable']:
                interp_color = "#F0A030"
            elif mean_score < BIO_EXTENT_THRESHOLDS['aggressive']:
                interp_color = "#E07020"
            else:
                interp_color = "#E05252"

            bio_severity = (f"[{bio_label}]  {interp_text}"
                            if bio_label and interp_text else interp_text or "—")
            tk.Label(bloc,
                     text=t("label_severity_row") + bio_severity,
                     font=("Courier", 8),
                     bg=T["bg_cluster"], fg=interp_color,
                     anchor='w',
                     ).pack(fill='x', padx=10, pady=(0, 6))

        # --- Tumor direction (pass-1 rule result) ---
        conf_color = (T["accent"] if confidence == "high"
                      else ("#F0A030" if confidence == "medium" else "#E0A050"))
        amb_suffix = "  ⚠ ambiguous" if is_amb else ""
        tk.Label(bloc,
                 text=(f"{t('label_pass1_row')} [{confidence}]{amb_suffix} : "
                       f"{suggestion}  —  {desc}"),
                 font=(ff, 9),
                 bg=T["bg_cluster"], fg=(conf_color if not is_amb else "#E07020"),
                 anchor='w',
                 ).pack(fill='x', padx=10, pady=(2, 2))

        # --- Zone constraint note (shown when Pass-1 label was overridden) ---
        if zone_note:
            tk.Label(bloc,
                     text=f"  !!  {zone_note}",
                     font=(ff, 8, "bold"),
                     bg=T["bg_cluster"], fg="#E07020",
                     anchor='w',
                     ).pack(fill='x', padx=10, pady=(0, 2))

        # --- Pass-2 reclassification note ---
        if reclass_note:
            tk.Label(bloc,
                     text=t("label_reclass") + reclass_note,
                     font=(ff, 8),
                     bg=T["bg_cluster"], fg="#E07020",
                     anchor='w', wraplength=580, justify='left',
                     ).pack(fill='x', padx=10, pady=(0, 2))

        # --- Pass-3 sub-classification note ---
        if subcat_note:
            tk.Label(bloc,
                     text=t("label_subcat") + subcat_note,
                     font=(ff, 8),
                     bg=T["bg_cluster"], fg="#2E75B6",
                     anchor='w', wraplength=580, justify='left',
                     ).pack(fill='x', padx=10, pady=(0, 8))
        else:
            tk.Label(bloc, bg=T["bg_cluster"], height=1).pack()

        f_entry = tk.Frame(bloc, bg=T["bg_cluster"])
        f_entry.pack(fill='x', padx=10, pady=(0, 10))

        tk.Label(f_entry,
                 text=t("label_field"),
                 font=(ff, 10),
                 bg=T["bg_cluster"], fg=T["text_sub"],
                 ).pack(side='left')

        var   = tk.StringVar(value=default_label)
        entry = tk.Entry(f_entry,
                         textvariable=var,
                         width=36,
                         font=(ff, 10),
                         bg=T["bg_input"],
                         fg=T["text"],
                         insertbackground=T["text"],
                         selectbackground=T["accent"],
                         relief="flat",
                         highlightthickness=1,
                         highlightbackground=T["separator"],
                         highlightcolor=T["accent"])
        entry.pack(side='left', padx=(8, 0), ipady=4)
        entries[c] = var

    def confirm():
        for c in range(best_k):
            val       = entries[c].get().strip()
            result[c] = val if val else f"Habitat_{c}"
        canvas.unbind_all("<MouseWheel>")
        win.grab_release()
        win.destroy()

    btn_frame = tk.Frame(win, bg=T["bg_card"])
    btn_frame.pack(fill='x', side='bottom')

    tk.Button(
        btn_frame,
        text=f"✓   {t('confirm')}",
        font=(ff, 11, "bold"),
        bg=T["accent"],
        fg="#FFFFFF",
        activebackground=T.get("accent_dark", T["accent"]),
        activeforeground="#FFFFFF",
        relief="flat", bd=0,
        cursor="hand2",
        padx=0, pady=12,
        command=confirm,
    ).pack(fill='x')

    win.bind("<Return>", lambda e: confirm())
    win.wait_window()
    return result


def show_habitat_map(labels, df_scaled, features_scaled,
                     parameters, best_k, output_dir=None):
    """
    Plot 2D habitat map and per-cluster bar chart.
    Safe to call from a secondary thread (Agg backend, no display).
    """
    colors = HABITAT_COLORS_HEX[:best_k]

    slice_id = df_scaled['Slice'].mode()[0]
    df_slice = df_scaled[df_scaled['Slice'] == slice_id].copy()
    df_slice['label'] = labels[df_scaled['Slice'] == slice_id]

    x_vals = sorted(df_slice['X'].unique())
    y_vals = sorted(df_slice['Y'].unique())
    x_idx  = {x: i for i, x in enumerate(x_vals)}
    y_idx  = {y: i for i, y in enumerate(y_vals)}

    grid = np.full((len(y_vals), len(x_vals)), -1, dtype=int)
    grid[df_slice['Y'].map(y_idx).values,
         df_slice['X'].map(x_idx).values] = df_slice['label'].values.astype(int)

    cmap      = ListedColormap(colors)
    centroids = {c: {p: features_scaled[labels == c, i].mean()
                     for i, p in enumerate(parameters)}
                 for c in range(best_k)}

    fig = plt.figure(figsize=(5 + 3 * best_k, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, best_k])

    ax_map = fig.add_subplot(gs[0])
    masked = np.ma.masked_where(grid == -1, grid)
    ax_map.imshow(masked, cmap=cmap, vmin=0, vmax=best_k - 1,
                  interpolation='nearest', origin='upper')
    ax_map.set_title(f'Habitat map -- slice {slice_id}', fontsize=11)
    ax_map.axis('off')

    ax_bar = fig.add_subplot(gs[1])
    x_pos  = np.arange(len(parameters))
    width  = 0.8 / best_k
    for c in range(best_k):
        vals = [centroids[c][p] for p in parameters]
        ax_bar.bar(x_pos + c * width, vals, width,
                   label=f'Cluster {c}', color=colors[c], alpha=0.8)
    ax_bar.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_bar.set_xticks(x_pos + width * (best_k - 1) / 2)
    ax_bar.set_xticklabels(parameters, rotation=35, ha='right', fontsize=9)
    ax_bar.set_ylabel('Robust-scaled value (0 = tumor median)')
    ax_bar.set_title('Multiparametric profile per cluster')
    ax_bar.legend(fontsize=8)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '3_habitat_map.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()
    return centroids


def label_habitats(labels, df_scaled, features_scaled, parameters,
                   best_k, output_dir=None):
    """
    Standalone wrapper for use outside the GUI.
    Displays habitat map then opens the labeling dialog (light theme).
    """
    centroids = show_habitat_map(labels, df_scaled, features_scaled,
                                  parameters, best_k, output_dir)

    colors = HABITAT_COLORS_HEX[:best_k]
    habitat_labels = _label_dialog(centroids, parameters, best_k, colors)

    if output_dir:
        label_path = os.path.join(output_dir, 'habitat_labels.json')
        with open(label_path, 'w') as f:
            json.dump(habitat_labels, f, indent=2)
        print(f"Labels saved : {label_path}")

    return habitat_labels
