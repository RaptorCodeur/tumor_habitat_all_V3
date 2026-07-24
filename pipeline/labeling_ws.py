# pipeline/labeling_ws.py
# Weakly-supervised cluster labelling — replaces classify_three_passes.
# 13 Labeling Functions (LFs) with abstention → weighted vote → P(label) per cluster.

import math

# ── Label constants ───────────────────────────────────────────────
ABSTAIN      = "__ABSTAIN__"
NECROSIS     = "Necrosis"
HYPOXIC      = "Hypoxic/vascular"
CELLULAR     = "Cellular/proliferative"
INFILTRATIVE = "Infiltrative"
BULK         = "Bulk tumor"
ASTROCYTE    = "Astrocyte barrier"
LABELS       = [NECROSIS, HYPOXIC, CELLULAR, INFILTRATIVE, BULK, ASTROCYTE]

# ── LF thresholds (biological, not data-derived) ──────────────────
_T2_NEC       = 0.5    # T2 min → necrosis (free water)
_MD_NEC       = 0.5    # MD min → necrosis (free diffusion)
_FA_NEC       = -0.3   # FA max → near-isotropic diffusion (necrosis)
_MD_NEC_EXT   = 1.0    # MD extreme → definitive free water
_T2S_HYP         = -0.15  # T2* max → deoxyHb / iron signal (hypoxic)
_T2S_HYP_STRONG  = -0.50  # T2* max (strong) → definitive deoxyHb, ignores T2 floor
_T2_HYP          = -0.50  # T2 min for lf_hyp_t2s_t2 (was 0.0; relaxed so dense+hypoxic clusters pass)
_DTOPO_HYP_LO    = 0.03   # d_topo min → just outside the necrotic sub-cluster
_DTOPO_HYP_HI    = 0.45   # d_topo max → peri-necrotic ring (recalibrated for sub-cluster seed)
_T2S_SPATIAL_MAX = 0.30   # T2* max for lf_hyp_spatial (positive T2* → not vascular)
_T2S_PERICORE_MAX= 0.50   # T2* max for lf_hyp_pericore (same rationale)
_T2_HYP_EXCL     = 1.5    # T2 above this = extreme necrosis → not hypoxic
_MD_HYP_EXCL     = 1.5    # MD above this = extreme necrosis → not hypoxic
_DTOPO_PERICORE_HI = 0.30  # d_topo max for peri-core hypoxic LF
_MD_CEL       = -0.1   # MD max → restricted diffusion (cellular)
_MTR_CEL      = 0.1    # MTR min → macromolecules (cellular)
_MD_CEL_STR   = -0.3   # MD max → strong restriction
_T2_CEL       = -0.2   # T2 max → dense cellularity
_FA_CEL_EXCL  = 0.3    # FA above this → WM fibres preserved → not cellular mass
_DTOPO_CP_LO  = 0.05   # d_topo min for pericore Cellular LF
_DTOPO_CP_HI  = 0.25   # d_topo max for pericore Cellular LF
_T2_CP        = -0.20  # T2 max → dense cellularity in pericore
_T2S_CP       = -0.10  # T2* min → not vasculaire (excludes true hypoxic)
_MD_CP        = -0.20  # MD max → restricted diffusion
_FA_CP        = 0.50   # FA min → organised fibres (alternative signal)
_FA_AST       = 1.3    # FA min → highly organised astrocytic fibres (> 1 IQR — very strict)
_MTR_AST      = 0.3    # MTR min → macromolecules (GFAP)
_RD_AST       = -0.5   # RD max → perpendicular diffusion very restricted
_FA_INF       = 0.1    # FA min → preserved WM fibres
_DTOPO_INF    = 0.5    # d_topo min → peripheral (not core)
_MD_INF_LO    = -0.2   # MD min → not fully cellular
_MD_INF_HI    = 0.3    # MD max → not necrotic
_FA_INF_WEAK  = 0.0    # FA min → weak infiltrative LF
_DTOPO_INF_W  = 0.4    # d_topo min → weak infiltrative LF
_T2S_BULK_ABS = 0.2    # |T2*| max → no vascular signal (bulk)
_MD_BULK_ABS  = 0.2    # |MD| max → close to tumour median
_T2_BULK_ABS  = 0.3    # |T2| max → close to tumour median
_DTOPO_BULK   = 0.75   # d_topo min → peripheral bulk
_T2S_BULK_NV  = -0.20  # T2* min → no vascular signal in periphery
_L2_BULK_REL  = 0.25   # l2_rel max → closest-to-median cluster → Bulk

# ε floor per label: probability mass when no LF votes.
# BULK > others → default prediction is Bulk tumor when all LFs abstain.
# INFILTRATIVE = 0 → only appears with an explicit LF vote.
_EPS = {NECROSIS: 0.01, HYPOXIC: 0.01, CELLULAR: 0.01,
        INFILTRATIVE: 0.0, BULK: 0.05, ASTROCYTE: 0.0}


# ── Helpers ───────────────────────────────────────────────────────
def _g(row, key):
    return row.get(key)

def _entropy(p):
    h = 0.0
    for v in p.values():
        if v > 1e-9:
            h -= v * math.log2(v)
    return h

def _confidence(max_p):
    if max_p >= 0.70: return 'high'
    if max_p >= 0.50: return 'medium'
    return 'low'

def _inject_virtual_md(cd):
    """If DTI-AD + DTI-RD present but DTI-MD absent, inject virtual MD = (AD+2*RD)/3."""
    if 'DTI-AD' in cd and 'DTI-RD' in cd and 'DTI-MD' not in cd:
        cd = dict(cd)
        cd['DTI-MD'] = (cd['DTI-AD'] + 2.0 * cd['DTI-RD']) / 3.0
    return cd


# ── Anchor identification ─────────────────────────────────────────
def _median_anchor(centroids, parameters):
    """Cluster closest to the origin in IQR space (= tumour median reference)."""
    return min(centroids,
               key=lambda c: sum(centroids[c].get(p, 0.0)**2 for p in parameters))

def _necrotic_candidate(centroids, parameters):
    """
    Cluster most consistent with necrosis: highest (MD + T2) with both > 0.
    Returns None if no cluster has both elevated.
    """
    if "DTI-MD" not in parameters or "T2" not in parameters:
        return None
    def _score(c):
        m = centroids[c].get("DTI-MD", 0.0)
        t = centroids[c].get("T2",     0.0)
        return m + t if (m > 0 and t > 0) else -1.0
    best = max(centroids, key=_score)
    return best if _score(best) > 0 else None


# ── Feature table ─────────────────────────────────────────────────
def build_feature_table(centroids, parameters):
    """
    Build an enriched feature dict for each cluster, adding:
      l2_from_median   : L2 distance from origin (tumour median) in IQR space
      l2_rel           : l2_from_median / max across clusters  ∈ [0, 1]
      delta_md_from_nec: MD_cluster − MD_necrotic_candidate    (None if no candidate)
      delta_t2_from_nec: T2_cluster − T2_necrotic_candidate    (None if no candidate)

    Returns (rows, anchor_med, anchor_nec).
    """
    # Inject virtual MD into each centroid if needed
    centroids = {c: _inject_virtual_md(cd) for c, cd in centroids.items()}

    anchor_med = _median_anchor(centroids, parameters)
    anchor_nec = _necrotic_candidate(centroids, parameters)

    l2 = {c: math.sqrt(sum(centroids[c].get(p, 0.0)**2 for p in parameters))
          for c in centroids}
    l2_max = max(l2.values(), default=1.0) or 1.0

    nec_md = centroids[anchor_nec].get("DTI-MD") if anchor_nec is not None else None
    nec_t2 = centroids[anchor_nec].get("T2")     if anchor_nec is not None else None

    rows = []
    for c, cd in centroids.items():
        row = dict(cd)
        row['cluster_id']        = c
        row['l2_from_median']    = l2[c]
        row['l2_rel']            = l2[c] / l2_max
        row['delta_md_from_nec'] = (
            cd.get("DTI-MD", 0.0) - nec_md
            if nec_md is not None and "DTI-MD" in cd else None
        )
        row['delta_t2_from_nec'] = (
            cd.get("T2", 0.0) - nec_t2
            if nec_t2 is not None and "T2" in cd else None
        )
        rows.append(row)

    return rows, anchor_med, anchor_nec


# ── Labeling Functions ────────────────────────────────────────────
# Convention: each LF receives one feature-enriched cluster dict (row)
# and returns a label string or ABSTAIN.

def lf_nec_t2_md(r):
    """T2 and MD both clearly elevated → tissue destruction → Necrosis."""
    t2, md = _g(r, "T2"), _g(r, "DTI-MD")
    if t2 is None or md is None: return ABSTAIN
    return NECROSIS if (t2 > _T2_NEC and md > _MD_NEC) else ABSTAIN

def lf_nec_fa_collapse(r):
    """FA near-isotropic with elevated MD → loss of tissue structure → Necrosis."""
    fa, md = _g(r, "DTI-FA"), _g(r, "DTI-MD")
    if fa is None or md is None: return ABSTAIN
    return NECROSIS if (fa < _FA_NEC and md > 0.2) else ABSTAIN

def lf_nec_md_extreme(r):
    """MD extreme (> 1 IQR above median) → definitive free diffusion → Necrosis."""
    md = _g(r, "DTI-MD")
    if md is None: return ABSTAIN
    return NECROSIS if md > _MD_NEC_EXT else ABSTAIN

def lf_hyp_t2s_t2(r):
    """T2* below median (deoxyHb) + T2 not negative → Hypoxic/vascular."""
    t2s, t2 = _g(r, "T2star"), _g(r, "T2")
    if t2s is None or t2 is None: return ABSTAIN
    return HYPOXIC if (t2s < _T2S_HYP and t2 >= _T2_HYP) else ABSTAIN

def lf_hyp_spatial(r):
    """Peri-necrotic zone + T2* not clearly positive → Hypoxic/vascular."""
    dt  = _g(r, "d_topo_norm")
    t2s = _g(r, "T2star")
    if dt is None: return ABSTAIN
    if t2s is not None and t2s > _T2S_SPATIAL_MAX: return ABSTAIN  # positive T2* → not vascular
    return HYPOXIC if _DTOPO_HYP_LO < dt < _DTOPO_HYP_HI else ABSTAIN

def lf_hyp_pericore(r):
    """Adjacent to core, no extreme necrosis, no positive T2* → Hypoxic/vascular."""
    dt = _g(r, "d_topo_norm")
    if dt is None: return ABSTAIN
    if not (_DTOPO_HYP_LO < dt < _DTOPO_PERICORE_HI):
        return ABSTAIN
    t2  = _g(r, "T2")
    md  = _g(r, "DTI-MD")
    t2s = _g(r, "T2star")
    if (t2 is not None and t2 > _T2_HYP_EXCL) and (md is not None and md > _MD_HYP_EXCL):
        return ABSTAIN  # too extreme → true necrosis
    if t2s is not None and t2s > _T2S_PERICORE_MAX: return ABSTAIN  # positive T2* → not vascular
    return HYPOXIC

def lf_hyp_t2s_strong(r):
    """T2* strongly negative (< -0.5 IQR) → definitive deoxyHb signal → Hypoxic/vascular.
    Ignores T2 floor — catches dense+hypoxic clusters with low T2 and very low T2*.
    """
    t2s = _g(r, "T2star")
    if t2s is None: return ABSTAIN
    md = _g(r, "DTI-MD")
    if md is not None and md > _MD_HYP_EXCL: return ABSTAIN  # free-water signal → necrosis
    return HYPOXIC if t2s < _T2S_HYP_STRONG else ABSTAIN

def lf_cel_md_mtr(r):
    """MD restricted + MTR elevated → high cell density + macromolecules → Cellular."""
    md, mtr = _g(r, "DTI-MD"), _g(r, "MTR")
    if md is None or mtr is None: return ABSTAIN
    fa = _g(r, "DTI-FA")
    if fa is not None and fa > _FA_CEL_EXCL: return ABSTAIN  # high FA → WM fibres, not cellular mass
    return CELLULAR if (md < _MD_CEL and mtr > _MTR_CEL) else ABSTAIN

def lf_cel_md_strong(r):
    """MD strongly restricted → high cell density → Cellular/proliferative."""
    md = _g(r, "DTI-MD")
    if md is None: return ABSTAIN
    fa = _g(r, "DTI-FA")
    if fa is not None and fa > _FA_CEL_EXCL: return ABSTAIN  # high FA → WM fibres, not cellular mass
    return CELLULAR if md < _MD_CEL_STR else ABSTAIN

def lf_cel_t2_md(r):
    """T2 and MD both below median → dense cellularity shortens both → Cellular."""
    t2, md = _g(r, "T2"), _g(r, "DTI-MD")
    if t2 is None or md is None: return ABSTAIN
    return CELLULAR if (t2 < _T2_CEL and md < _MD_CEL) else ABSTAIN

def lf_cel_pericore(r):
    """Pericore zone, dense T2, neutral T2*, restricted MD or organised FA → Cellular.

    Targets dense-cellularity clusters in the peri-necrotic ring (dt 0.05–0.25)
    where T2* is near-neutral (not vascular) and T2 is negative.
    T2* < -0.10 exclusion separates true hypoxic clusters (with deoxyHb signal)
    from Cellular zones that happen to sit near the core.
    Accepts either restricted MD (<-0.20) or organised FA (>0.50) as diffusion
    evidence, to cover both standard Cellular and fibre-adjacent Cellular profiles.
    """
    dt  = _g(r, "d_topo_norm");  t2s = _g(r, "T2star")
    t2  = _g(r, "T2");           md  = _g(r, "DTI-MD")
    fa  = _g(r, "DTI-FA")
    if dt is None: return ABSTAIN
    if not (_DTOPO_CP_LO < dt < _DTOPO_CP_HI): return ABSTAIN
    if t2s is None or t2s < _T2S_CP: return ABSTAIN   # vasculaire/hypoxic → not Cellular
    if t2 is None  or t2  > _T2_CP:  return ABSTAIN   # T2 must be negative (dense)
    restricted = (md is not None and md < _MD_CP)
    organised  = (fa is not None and fa > _FA_CP)
    if not (restricted or organised): return ABSTAIN
    return CELLULAR

def lf_inf_fa_periph(r):
    """FA preserved + cluster in peripheral zone → WM fibre infiltration → Infiltrative."""
    fa, dt = _g(r, "DTI-FA"), _g(r, "d_topo_norm")
    if fa is None or dt is None: return ABSTAIN
    return INFILTRATIVE if (fa > _FA_INF and dt > _DTOPO_INF) else ABSTAIN

def lf_inf_md_intermed(r):
    """MD intermediate + FA non-zero + peripheral → partial WM infiltration → Infiltrative."""
    fa, md, dt = _g(r, "DTI-FA"), _g(r, "DTI-MD"), _g(r, "d_topo_norm")
    if fa is None or md is None or dt is None: return ABSTAIN
    return INFILTRATIVE if (_MD_INF_LO < md < _MD_INF_HI
                            and fa > _FA_INF_WEAK
                            and dt > _DTOPO_INF_W) else ABSTAIN

def lf_bulk_near_median(r):
    """All key features near zero → no distinguishing signal → Bulk tumor."""
    t2s = _g(r, "T2star")
    md  = _g(r, "DTI-MD")
    t2  = _g(r, "T2")
    if t2s is None or md is None or t2 is None: return ABSTAIN
    return BULK if (abs(t2s) < _T2S_BULK_ABS
                    and abs(md)  < _MD_BULK_ABS
                    and abs(t2)  < _T2_BULK_ABS) else ABSTAIN

def lf_bulk_periph(r):
    """Peripheral cluster without vascular signal → tumour margin → Bulk tumor."""
    dt, t2s = _g(r, "d_topo_norm"), _g(r, "T2star")
    if dt is None or t2s is None: return ABSTAIN
    return BULK if (dt > _DTOPO_BULK and t2s > _T2S_BULK_NV) else ABSTAIN

def lf_bulk_rel_median(r):
    """Closest cluster to tumour median by relative L2 → reference cluster → Bulk tumor."""
    l2r = _g(r, "l2_rel")
    if l2r is None: return ABSTAIN
    if l2r >= _L2_BULK_REL: return ABSTAIN
    md = _g(r, "DTI-MD")
    if md is not None and md < -0.10: return ABSTAIN  # restricted diffusion → Cellular
    return BULK

def lf_ast_fa_mtr(r):
    """Very high FA + elevated MTR → organised astrocytic fibres (GFAP) → Astrocyte barrier."""
    fa  = _g(r, "DTI-FA")
    mtr = _g(r, "MTR")
    if fa is None or mtr is None: return ABSTAIN
    t2 = _g(r, "T2")
    md = _g(r, "DTI-MD")
    if (t2 is not None and t2 > _T2_NEC) and (md is not None and md > _MD_NEC):
        return ABSTAIN  # necrotic profile → not astrocytic
    return ASTROCYTE if (fa > _FA_AST and mtr > _MTR_AST) else ABSTAIN

def lf_ast_fa_rd(r):
    """Very high FA + very restricted RD → anisotropic astrocytic barrier → Astrocyte barrier."""
    fa = _g(r, "DTI-FA")
    rd = _g(r, "DTI-RD")
    if fa is None or rd is None: return ABSTAIN
    t2 = _g(r, "T2")
    md = _g(r, "DTI-MD")
    if (t2 is not None and t2 > _T2_NEC) and (md is not None and md > _MD_NEC):
        return ABSTAIN  # necrotic profile → not astrocytic
    return ASTROCYTE if (fa > _FA_AST and rd < _RD_AST) else ABSTAIN


# ── LF registry ───────────────────────────────────────────────────
LF_REGISTRY = [
    ("lf_nec_t2_md",        lf_nec_t2_md,        NECROSIS),
    ("lf_nec_fa_collapse",   lf_nec_fa_collapse,   NECROSIS),
    ("lf_nec_md_extreme",    lf_nec_md_extreme,    NECROSIS),
    ("lf_hyp_t2s_t2",        lf_hyp_t2s_t2,        HYPOXIC),
    ("lf_hyp_spatial",       lf_hyp_spatial,       HYPOXIC),
    ("lf_hyp_pericore",      lf_hyp_pericore,      HYPOXIC),
    ("lf_hyp_t2s_strong",    lf_hyp_t2s_strong,    HYPOXIC),
    ("lf_cel_md_mtr",        lf_cel_md_mtr,        CELLULAR),
    ("lf_cel_md_strong",     lf_cel_md_strong,     CELLULAR),
    ("lf_cel_t2_md",         lf_cel_t2_md,         CELLULAR),
    ("lf_cel_pericore",      lf_cel_pericore,      CELLULAR),
    ("lf_inf_fa_periph",     lf_inf_fa_periph,     INFILTRATIVE),
    ("lf_inf_md_intermed",   lf_inf_md_intermed,   INFILTRATIVE),
    ("lf_bulk_near_median",  lf_bulk_near_median,  BULK),
    ("lf_bulk_periph",       lf_bulk_periph,       BULK),
    ("lf_bulk_rel_median",   lf_bulk_rel_median,   BULK),
    ("lf_ast_fa_mtr",        lf_ast_fa_mtr,        ASTROCYTE),
    ("lf_ast_fa_rd",         lf_ast_fa_rd,         ASTROCYTE),
]

LF_NAMES = [n for n, _, _ in LF_REGISTRY]


# ── Apply all LFs ─────────────────────────────────────────────────
def _apply_lfs(rows):
    """Return {cluster_id: {lf_name: label | ABSTAIN}}."""
    return {row['cluster_id']: {n: fn(row) for n, fn, _ in LF_REGISTRY}
            for row in rows}


# ── Coverage / conflict analysis ──────────────────────────────────
def lf_analysis(L):
    """
    Analyse LF quality without ground truth.

    For each LF returns a dict with:
      coverage     : fraction of clusters where LF voted (not ABSTAIN)
      conflict_rate: fraction of voted clusters where the majority of other
                     LFs disagreed with this LF's vote
      avg_overlap  : mean number of other LFs that voted the same label
                     on the same cluster
      n_votes      : absolute number of clusters where LF voted
    """
    cluster_ids = list(L.keys())
    n = len(cluster_ids)
    out = []
    for name, _, target in LF_REGISTRY:
        votes = [(c, L[c][name]) for c in cluster_ids if L[c][name] != ABSTAIN]
        cov   = len(votes) / n if n > 0 else 0.0
        conflicts, overlaps = 0, []
        for c, lbl in votes:
            others = [L[c][n2] for n2, _, _ in LF_REGISTRY
                      if n2 != name and L[c][n2] != ABSTAIN]
            if others:
                if max(set(others), key=others.count) != lbl:
                    conflicts += 1
                overlaps.append(sum(1 for v in others if v == lbl))
            else:
                overlaps.append(0)
        out.append({
            'lf_name'      : name,
            'target_label' : target,
            'coverage'     : round(cov, 3),
            'conflict_rate': round(conflicts / len(votes), 3) if votes else 0.0,
            'avg_overlap'  : round(sum(overlaps) / len(overlaps), 2) if overlaps else 0.0,
            'n_votes'      : len(votes),
        })
    return out


# ── Weighted vote ─────────────────────────────────────────────────
def weighted_vote(rows, L, weights=None):
    """
    Weighted vote → {cluster_id: {label: probability}}.

    weights : {lf_name: float}  — default 1.0 per LF.
    INFILTRATIVE has ε=0: only appears with at least one explicit vote.
    """
    w = weights or {n: 1.0 for n, _, _ in LF_REGISTRY}
    probas = {}
    for row in rows:
        c      = row['cluster_id']
        scores = dict(_EPS)
        for name, _, _ in LF_REGISTRY:
            vote = L[c][name]
            if vote != ABSTAIN:
                scores[vote] = scores[vote] + w.get(name, 1.0)
        total = sum(scores.values())
        probas[c] = ({lbl: scores[lbl] / total for lbl in LABELS}
                     if total > 1e-9
                     else {lbl: 1.0 / len(LABELS) for lbl in LABELS})
    return probas


# ── Weight estimation from GT ─────────────────────────────────────
def estimate_weights_from_gt(L, gt_labels):
    """
    Estimate LF precision from annotated clusters (single animal).

    L         : {cluster_id: {lf_name: label | ABSTAIN}}
    gt_labels : {cluster_id: label_str}
    Returns   : {lf_name: float}  — falls back to 1.0 when LF never voted on GT.
    """
    weights = {}
    for name, _, _ in LF_REGISTRY:
        correct = voted = 0
        for c, true_lbl in gt_labels.items():
            if c not in L:
                continue
            vote = L[c][name]
            if vote != ABSTAIN:
                voted   += 1
                correct += int(vote == true_lbl)
        weights[name] = (correct / voted) if voted > 0 else 0.5
    return weights


def estimate_weights_from_multi_gt(animal_runs, gt_json):
    """
    Estimate LF weights from multiple annotated animals.

    animal_runs : {animal_id: (centroids, parameters)}
                  centroids = {cluster_id: {feature: value}}
    gt_json     : {animal_id: {str(cluster_id): label_str}}
                  (keys are strings as loaded from JSON)

    Returns     : {lf_name: float}

    Usage
    -----
    import json
    with open("gt_labels.json") as f:
        gt = json.load(f)

    animal_runs = {
        "animal_01": (centroids_01, parameters),
        "animal_02": (centroids_02, parameters),
        ...
    }
    weights = estimate_weights_from_multi_gt(animal_runs, gt)
    """
    # Accumulate correct/voted counts across all GT animals
    correct_by_lf = {name: 0 for name, _, _ in LF_REGISTRY}
    voted_by_lf   = {name: 0 for name, _, _ in LF_REGISTRY}

    for animal_id, (centroids, parameters) in animal_runs.items():
        if animal_id not in gt_json:
            continue
        animal_gt_raw = gt_json[animal_id]                       # {str(c): label}
        animal_gt     = {int(k): v for k, v in animal_gt_raw.items()}

        rows, _, _ = build_feature_table(centroids, parameters)
        L = _apply_lfs(rows)

        for name, _, _ in LF_REGISTRY:
            for c, true_lbl in animal_gt.items():
                if c not in L:
                    continue
                vote = L[c][name]
                if vote != ABSTAIN:
                    voted_by_lf[name]   += 1
                    correct_by_lf[name] += int(vote == true_lbl)

    return {name: (correct_by_lf[name] / voted_by_lf[name])
                   if voted_by_lf[name] > 0 else 0.5
            for name, _, _ in LF_REGISTRY}


# ── Main entry point ──────────────────────────────────────────────
def label_clusters_ws(centroids, parameters,
                      gt_labels=None, weights=None,
                      zone_map=None, **_kwargs):
    """
    Weakly-supervised cluster labelling — replaces classify_three_passes.

    centroids  : {cluster_id: {feature: value}}
    parameters : list of MRI parameter names
    gt_labels  : {cluster_id: label_str}  annotated animals, optional
    weights    : {lf_name: float}         pre-computed weights, optional
    zone_map   : ignored (accepted for call-site compatibility)

    Returns {cluster_id: {
        label, label_p1, probas, score, confidence, entropy,
        is_ambiguous, lf_votes, n_votes, desc,
        anchor_median, anchor_necrotic,
        reclass_note, subcat_note, zone_note  ← always None (compat)
    }}
    """
    rows, anchor_med, anchor_nec = build_feature_table(centroids, parameters)
    L = _apply_lfs(rows)

    if weights is None and gt_labels is not None:
        weights = estimate_weights_from_gt(L, gt_labels)

    probas = weighted_vote(rows, L, weights)

    result = {}
    for row in rows:
        c        = row['cluster_id']
        p        = probas[c]
        sorted_p = sorted(p.values(), reverse=True)
        best_lbl = max(p, key=p.get)
        best_p   = sorted_p[0]
        second_p = sorted_p[1] if len(sorted_p) > 1 else 0.0
        n_votes  = sum(1 for v in L[c].values() if v != ABSTAIN)

        result[c] = {
            'label'          : best_lbl,
            'label_p1'       : best_lbl,
            'probas'         : p,
            'score'          : round(best_p, 3),
            'confidence'     : _confidence(best_p),
            'entropy'        : round(_entropy(p), 3),
            'is_ambiguous'   : second_p >= best_p - 0.15,
            'lf_votes'       : L[c],
            'n_votes'        : n_votes,
            'desc'           : (f"{n_votes}/{len(LF_REGISTRY)} LFs voted — "
                                f"P({best_lbl})={best_p:.2f}"),
            'anchor_median'  : anchor_med,
            'anchor_necrotic': anchor_nec,
            'reclass_note'   : None,
            'subcat_note'    : None,
            'zone_note'      : None,
        }

    return result
