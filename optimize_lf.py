"""
optimize_lf.py — Simulation de corrections LF pour atteindre 80%.

Stratégies testées :
  A. Recalibration des poids sur le nouveau GT
  B. lf_hyp_spatial : T2* <= 0.0 (plus strict)
  C. lf_cel_t2_md   : seuils T2 et MD plus stricts
  D. lf_bulk_rel_median : exclusion T2* négatif
  E. lf_hyp_t2s_strong  : poids augmenté pour C050524 c3
  F. lf_cel_md_strong/cel_t2_md : exclusion quand T2* < -0.5
"""
import sys, os, json, math, itertools
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.labeling_ws import (
    build_feature_table, _apply_lfs, weighted_vote,
    estimate_weights_from_multi_gt,
    LF_REGISTRY, LABELS, ABSTAIN,
    NECROSIS, HYPOXIC, CELLULAR, INFILTRATIVE, BULK, ASTROCYTE,
    _g, _inject_virtual_md,
    _T2S_HYP, _T2_HYP, _DTOPO_HYP_LO, _DTOPO_HYP_HI,
    _T2S_SPATIAL_MAX, _T2S_PERICORE_MAX,
    _T2_NEC, _MD_NEC, _MD_NEC_EXT,
    _T2_CEL, _MD_CEL, _FA_CEL_EXCL,
    _L2_BULK_REL, _T2S_BULK_ABS, _MD_BULK_ABS, _T2_BULK_ABS,
    _FA_AST, _MTR_AST, _RD_AST,
    _T2S_HYP_STRONG, _MD_HYP_EXCL, _T2_HYP_EXCL,
    _DTOPO_PERICORE_HI,
)

GT_JSON   = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labels_Without_hierarchie.json"
GT_FOLDER = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labeling"
LF_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "lf_weights.json")

with open(GT_JSON, encoding="utf-8") as f:
    raw = f.read().replace('"Hypoxic/vascular,', '"Hypoxic/vascular",')
gt_all = json.loads(raw)

# ── Chargement centroids ──────────────────────────────────────────────
def load_all():
    runs = {}
    for animal_id in gt_all:
        cp = os.path.join(GT_FOLDER, animal_id, "V2", "centroids.json")
        if not os.path.exists(cp): continue
        with open(cp) as f:
            raw_c = json.load(f)
        centroids  = {int(k): v for k, v in raw_c.items()}
        parameters = [p for p in next(iter(centroids.values())).keys()
                      if p != "d_topo_norm"]
        runs[animal_id] = (centroids, parameters)
    return runs

all_runs = load_all()

# ── A. Recalibration des poids sur nouveau GT ─────────────────────────
new_weights = estimate_weights_from_multi_gt(all_runs, gt_all)
print("=== A. Poids recalibrés sur nouveau GT ===")
with open(LF_WEIGHTS_FILE) as f:
    old_weights = json.load(f)
for name, _, _ in LF_REGISTRY:
    old = old_weights.get(name, 1.0)
    new = new_weights.get(name, 1.0)
    delta = new - old
    marker = " <<" if abs(delta) > 0.05 else ""
    print(f"  {name:<24}  {old:.3f} -> {new:.3f}  ({delta:+.3f}){marker}")

# ── Évaluation de base ────────────────────────────────────────────────
def evaluate(weights, lf_overrides=None):
    """Retourne (n_correct, n_total, errors_by_label)."""
    true_all, pred_all = [], []
    for animal_id, gt_clusters in gt_all.items():
        if animal_id not in all_runs: continue
        centroids, parameters = all_runs[animal_id]
        gt_int = {int(k): v for k, v in gt_clusters.items()}
        rows, _, _ = build_feature_table(centroids, parameters)
        L = _apply_lfs(rows)
        if lf_overrides:
            for row in rows:
                c = row["cluster_id"]
                for lf_name, lf_fn in lf_overrides.items():
                    L[c][lf_name] = lf_fn(row)
        probas = weighted_vote(rows, L, weights=weights)
        for c_id, true_label in gt_int.items():
            if c_id not in probas: continue
            true_all.append(true_label)
            pred_all.append(max(probas[c_id], key=probas[c_id].get))
    n = len(true_all)
    ok = sum(t == p for t, p in zip(true_all, pred_all))
    by_lbl = {}
    for lbl in LABELS:
        idx = [i for i, t in enumerate(true_all) if t == lbl]
        if idx:
            by_lbl[lbl] = sum(pred_all[i] == lbl for i in idx), len(idx)
    return ok, n, by_lbl

# Baseline
ok0, n0, by0 = evaluate(old_weights)
print(f"\nBaseline (anciens poids) : {ok0}/{n0} = {100*ok0/n0:.1f}%")

# Nouveaux poids seuls
ok_a, n_a, by_a = evaluate(new_weights)
print(f"A. Nouveaux poids seuls  : {ok_a}/{n_a} = {100*ok_a/n_a:.1f}%")

# ── B-F. LF overrides ────────────────────────────────────────────────

# B. lf_hyp_spatial plus strict (T2* <= 0.0 au lieu de 0.3)
def lf_hyp_spatial_B(r):
    dt  = _g(r, "d_topo_norm")
    t2s = _g(r, "T2star")
    if dt is None: return ABSTAIN
    if t2s is not None and t2s > 0.0: return ABSTAIN
    return HYPOXIC if _DTOPO_HYP_LO < dt < _DTOPO_HYP_HI else ABSTAIN

# C. lf_cel_t2_md plus strict (T2 < -0.3 au lieu de -0.2, MD < -0.2 au lieu de -0.1)
def lf_cel_t2_md_C(r):
    t2, md = _g(r, "T2"), _g(r, "DTI-MD")
    if t2 is None or md is None: return ABSTAIN
    t2s = _g(r, "T2star")
    if t2s is not None and t2s < -0.4: return ABSTAIN  # T2* très négatif → hypoxique
    return CELLULAR if (t2 < -0.30 and md < -0.20) else ABSTAIN

# D. lf_bulk_rel_median : exclure si T2* négatif (signal hypoxique)
def lf_bulk_rel_median_D(r):
    l2r = _g(r, "l2_rel")
    if l2r is None: return ABSTAIN
    if l2r >= _L2_BULK_REL: return ABSTAIN
    md  = _g(r, "DTI-MD")
    if md is not None and md < -0.10: return ABSTAIN
    t2s = _g(r, "T2star")
    if t2s is not None and t2s < -0.10: return ABSTAIN  # signal vasculaire → pas Bulk
    return BULK

# E. lf_cel_md_strong : exclure si T2* < -0.4 (signe hypoxique fort)
def lf_cel_md_strong_E(r):
    md = _g(r, "DTI-MD")
    if md is None: return ABSTAIN
    fa = _g(r, "DTI-FA")
    if fa is not None and fa > _FA_CEL_EXCL: return ABSTAIN
    t2s = _g(r, "T2star")
    if t2s is not None and t2s < -0.40: return ABSTAIN  # hypoxique fort → pas purement Cellular
    return CELLULAR if md < -0.30 else ABSTAIN  # seuil MD aussi plus strict

# Simulations combinées
print("\n=== Simulations (nouveaux poids + overrides) ===")
print(f"  {'Combo':<35}  {'Global':>8}  Hyp  Cel  Bulk  Ast")
print("  " + "-"*70)

combos = {
    "baseline (anciens poids)"  : ({}, old_weights),
    "A. nouveaux poids"         : ({}, new_weights),
    "A+B. +hyp_spatial strict"  : ({"lf_hyp_spatial": lf_hyp_spatial_B}, new_weights),
    "A+C. +cel_t2_md strict"    : ({"lf_cel_t2_md": lf_cel_t2_md_C}, new_weights),
    "A+D. +bulk_rel excl T2*"   : ({"lf_bulk_rel_median": lf_bulk_rel_median_D}, new_weights),
    "A+B+C"                     : ({"lf_hyp_spatial": lf_hyp_spatial_B,
                                    "lf_cel_t2_md": lf_cel_t2_md_C}, new_weights),
    "A+B+D"                     : ({"lf_hyp_spatial": lf_hyp_spatial_B,
                                    "lf_bulk_rel_median": lf_bulk_rel_median_D}, new_weights),
    "A+C+D"                     : ({"lf_cel_t2_md": lf_cel_t2_md_C,
                                    "lf_bulk_rel_median": lf_bulk_rel_median_D}, new_weights),
    "A+B+C+D"                   : ({"lf_hyp_spatial": lf_hyp_spatial_B,
                                    "lf_cel_t2_md": lf_cel_t2_md_C,
                                    "lf_bulk_rel_median": lf_bulk_rel_median_D}, new_weights),
    "A+B+C+D+E"                 : ({"lf_hyp_spatial": lf_hyp_spatial_B,
                                    "lf_cel_t2_md": lf_cel_t2_md_C,
                                    "lf_bulk_rel_median": lf_bulk_rel_median_D,
                                    "lf_cel_md_strong": lf_cel_md_strong_E}, new_weights),
}

best_ok, best_name, best_combo = 0, "", {}
for name, (overrides, weights) in combos.items():
    ok, n, by = evaluate(weights, overrides)
    hyp_ok,  hyp_n  = by.get(HYPOXIC,   (0,0))
    cel_ok,  cel_n  = by.get(CELLULAR,  (0,0))
    bulk_ok, bulk_n = by.get(BULK,      (0,0))
    ast_ok,  ast_n  = by.get(ASTROCYTE, (0,0))
    pct = 100*ok/n
    mark = " *** BEST ***" if ok > best_ok else ""
    print(f"  {name:<35}  {ok:>2}/{n} ({pct:>5.1f}%)"
          f"  {hyp_ok}/{hyp_n}  {cel_ok}/{cel_n}  {bulk_ok}/{bulk_n}  {ast_ok}/{ast_n}{mark}")
    if ok > best_ok:
        best_ok, best_name, best_combo = ok, name, (overrides, weights)

print(f"\n>>> Meilleur : {best_name} → {best_ok}/{n} = {100*best_ok/n:.1f}%")
print("\nErreurs restantes avec la meilleure combo :")
overrides_b, weights_b = best_combo
for animal_id, gt_clusters in gt_all.items():
    if animal_id not in all_runs: continue
    centroids, parameters = all_runs[animal_id]
    gt_int = {int(k): v for k, v in gt_clusters.items()}
    rows, _, _ = build_feature_table(centroids, parameters)
    L = _apply_lfs(rows)
    for row in rows:
        c = row["cluster_id"]
        for lf_name, lf_fn in overrides_b.items():
            L[c][lf_name] = lf_fn(row)
    probas = weighted_vote(rows, L, weights=weights_b)
    for c_id, true_label in gt_int.items():
        if c_id not in probas: continue
        pred = max(probas[c_id], key=probas[c_id].get)
        if pred != true_label:
            cen = _inject_virtual_md(centroids[c_id])
            print(f"  {animal_id} c{c_id}: GT={true_label:<22} pred={pred:<22}"
                  f"  T2*={cen.get('T2star',float('nan')):+.2f}"
                  f"  T2={cen.get('T2',float('nan')):+.2f}"
                  f"  MD={cen.get('DTI-MD',float('nan')):+.2f}"
                  f"  dt={cen.get('d_topo_norm',float('nan')):.2f}")
