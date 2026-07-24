"""
optimize_lf2.py — Recherche fine pour atteindre 80%.

Analyse précise des 14 erreurs avec les nouveaux poids.
Test de corrections ciblées, une par une puis combinées.
"""
import sys, os, json, math
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
    _T2_CEL, _MD_CEL, _FA_CEL_EXCL, _MD_CEL_STR,
    _L2_BULK_REL, _T2S_BULK_ABS, _MD_BULK_ABS, _T2_BULK_ABS,
    _FA_AST, _MTR_AST, _RD_AST,
    _T2S_HYP_STRONG, _MD_HYP_EXCL, _T2_HYP_EXCL,
    _DTOPO_PERICORE_HI,
)

GT_JSON   = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labels_Without_hierarchie.json"
GT_FOLDER = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labeling"

with open(GT_JSON, encoding="utf-8") as f:
    raw = f.read().replace('"Hypoxic/vascular,', '"Hypoxic/vascular",')
gt_all = json.loads(raw)

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
new_weights = estimate_weights_from_multi_gt(all_runs, gt_all)

from pipeline.labeling_ws import _EPS, LF_REGISTRY as _LF_REGISTRY

def evaluate(weights, lf_overrides=None, extra_lfs=None):
    """
    lf_overrides  : {lf_name: fn}  — remplace une LF existante du registre
    extra_lfs     : {lf_name: (fn, weight)}  — ajoute une LF hors-registre
    """
    true_all, pred_all, details = [], [], []
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
        # Score manuellement (permet d'ajouter des LFs hors-registre)
        probas = {}
        for row in rows:
            c      = row["cluster_id"]
            scores = dict(_EPS)
            for name, _, _ in _LF_REGISTRY:
                vote = L[c].get(name, ABSTAIN)
                if vote != ABSTAIN:
                    scores[vote] = scores[vote] + weights.get(name, 1.0)
            if extra_lfs:
                for lf_name, (lf_fn, lf_w) in extra_lfs.items():
                    vote = lf_fn(row)
                    if vote != ABSTAIN:
                        scores[vote] = scores[vote] + lf_w
            total = sum(scores.values())
            probas[c] = ({lbl: scores[lbl] / total for lbl in LABELS}
                         if total > 1e-9
                         else {lbl: 1.0 / len(LABELS) for lbl in LABELS})
        for c_id, true_label in gt_int.items():
            if c_id not in probas: continue
            pred = max(probas[c_id], key=probas[c_id].get)
            true_all.append(true_label)
            pred_all.append(pred)
            details.append((animal_id, c_id, true_label, pred,
                            centroids[c_id]))
    n  = len(true_all)
    ok = sum(t == p for t, p in zip(true_all, pred_all))
    by_lbl = {}
    for lbl in LABELS:
        idx = [i for i, t in enumerate(true_all) if t == lbl]
        if idx:
            by_lbl[lbl] = (sum(pred_all[i] == lbl for i in idx), len(idx))
    return ok, n, by_lbl, details

# ── Nouvelles LF candidates ───────────────────────────────────────────

# P1. lf_hyp_pericore : T2* doit être négatif ou neutre (≤ 0.0 au lieu de 0.5)
def lf_hyp_pericore_P1(r):
    dt = _g(r, "d_topo_norm")
    if dt is None: return ABSTAIN
    if not (_DTOPO_HYP_LO < dt < _DTOPO_PERICORE_HI): return ABSTAIN
    t2  = _g(r, "T2");    md  = _g(r, "DTI-MD");   t2s = _g(r, "T2star")
    if (t2 is not None and t2 > _T2_HYP_EXCL) and (md is not None and md > _MD_HYP_EXCL):
        return ABSTAIN
    if t2s is not None and t2s > 0.0: return ABSTAIN  # T2* positif → pas vasculaire
    return HYPOXIC

# P2. lf_hyp_spatial : exclure si MD élevé (signal nécrose partielle pas Hypoxic)
def lf_hyp_spatial_P2(r):
    dt  = _g(r, "d_topo_norm");  t2s = _g(r, "T2star")
    md  = _g(r, "DTI-MD")
    if dt is None: return ABSTAIN
    if t2s is not None and t2s > _T2S_SPATIAL_MAX: return ABSTAIN
    if md  is not None and md  > 0.5:              return ABSTAIN  # MD élevé → nécrose, pas hypoxie
    return HYPOXIC if _DTOPO_HYP_LO < dt < _DTOPO_HYP_HI else ABSTAIN

# P3. lf_bulk_near_median : exclure quand T2* est clairement négatif
def lf_bulk_near_median_P3(r):
    t2s = _g(r, "T2star");  md = _g(r, "DTI-MD");  t2 = _g(r, "T2")
    if t2s is None or md is None or t2 is None: return ABSTAIN
    if t2s < -0.15: return ABSTAIN  # signal vasculaire → pas Bulk
    return BULK if (abs(t2s) < _T2S_BULK_ABS
                    and abs(md)  < _MD_BULK_ABS
                    and abs(t2)  < _T2_BULK_ABS) else ABSTAIN

# P5. lf_bulk_near_core : voxels juste au-dessus du coeur necrotique → Bulk
def lf_bulk_near_core(r):
    dt  = _g(r, "d_topo_norm");  t2s = _g(r, "T2star")
    t2  = _g(r, "T2");           md  = _g(r, "DTI-MD")
    if dt is None: return ABSTAIN
    if dt > 0.10: return ABSTAIN                           # pas au bord du coeur
    if t2s is not None and t2s < -0.10: return ABSTAIN    # signal vasculaire fort → pas Bulk
    if t2 is None or t2 <= 0.30: return ABSTAIN           # T2 doit être élévé (pas nécrose extrême)
    if md is None or md <= 0.30: return ABSTAIN           # MD doit être élévé
    if (t2 is not None and t2 > 1.5) or (md is not None and md > 1.5):
        return ABSTAIN  # profil nécrotique extrême → laisser aux LF nécrose
    return BULK

# P6a. lf_cel_pericore (version MD stricte) : dt[0.05-0.25], T2<-0.20, MD<-0.20, T2*>-0.10
def lf_cel_pericore_v1(r):
    dt  = _g(r, "d_topo_norm");  t2s = _g(r, "T2star")
    t2  = _g(r, "T2");           md  = _g(r, "DTI-MD")
    if dt is None: return ABSTAIN
    if not (0.05 < dt < 0.25): return ABSTAIN
    if t2s is None or t2s < -0.10: return ABSTAIN         # exclut Hypoxic vasculaire
    if t2 is None or t2 > -0.20: return ABSTAIN
    if md is None or md > -0.20: return ABSTAIN
    return CELLULAR

# P6b. lf_cel_pericore (version élargie) : dt[0.05-0.25], T2<-0.20, (MD<-0.20 OR FA>0.50), T2*>-0.10
def lf_cel_pericore_v2(r):
    dt  = _g(r, "d_topo_norm");  t2s = _g(r, "T2star")
    t2  = _g(r, "T2");           md  = _g(r, "DTI-MD")
    fa  = _g(r, "DTI-FA")
    if dt is None: return ABSTAIN
    if not (0.05 < dt < 0.25): return ABSTAIN
    if t2s is None or t2s < -0.10: return ABSTAIN         # exclut Hypoxic vasculaire
    if t2 is None or t2 > -0.20: return ABSTAIN
    # MD restreint OU FA élevée (fibres organisées avec T2 dense)
    restricted_diff = (md is not None and md < -0.20)
    organised_fa    = (fa is not None and fa > 0.50)
    if not (restricted_diff or organised_fa): return ABSTAIN
    return CELLULAR

lf_cel_pericore = lf_cel_pericore_v1  # alias pour compatibilité

# ── Tableau de comparaison ────────────────────────────────────────────
print("=== Simulations ciblées (base = nouveaux poids) ===")
print(f"  {'Combo':<38}  {'Global':>8}  Hyp  Cel  Bulk  Ast")
print("  " + "-"*72)

# combos = (overrides_registre, extra_lfs_hors_registre, weight_mode)
# extra_lfs = {name: (fn, weight)}
combos = {
    "baseline anciens poids"          : ({}, {}, "old"),
    "A. nouveaux poids"               : ({}, {}, "new"),
    # P6a : cel_pericore MD strict (A170122 c3 seulement)
    "A+P6a. cel_pericore_v1 w=0.7"   : ({}, {"lf_cel_pericore_v1": (lf_cel_pericore_v1, 0.7)}, "new"),
    "A+P6a. cel_pericore_v1 w=1.0"   : ({}, {"lf_cel_pericore_v1": (lf_cel_pericore_v1, 1.0)}, "new"),
    # P6b : cel_pericore elargi (A170122 c3 + A170624 c2)
    "A+P6b. cel_pericore_v2 w=1.0"   : ({}, {"lf_cel_pericore_v2": (lf_cel_pericore_v2, 1.0)}, "new"),
    "A+P6b. cel_pericore_v2 w=1.2"   : ({}, {"lf_cel_pericore_v2": (lf_cel_pericore_v2, 1.2)}, "new"),
    # Combinaison P5 + P6b
    "A+P5(1.5)+P6b(1.0)"             : ({}, {"lf_bulk_near_core":  (lf_bulk_near_core,   1.5),
                                              "lf_cel_pericore_v2": (lf_cel_pericore_v2, 1.0)}, "new"),
    # Reference
    "A+P1. pericore T2*<=0"           : ({"lf_hyp_pericore": lf_hyp_pericore_P1}, {}, "new"),
}

import pipeline.labeling_ws as lws
old_weights_dict = {}
lf_weights_path  = os.path.join(os.path.dirname(__file__), "lf_weights.json")
with open(lf_weights_path) as f:
    old_weights_dict = json.load(f)

best_ok, best_name, best_cfg = 0, "", None
results = {}
for name, (overrides, extras, wmode) in combos.items():
    w = old_weights_dict if wmode == "old" else new_weights
    ok, n, by, dets = evaluate(w, overrides, extras if extras else None)
    hyp  = by.get(HYPOXIC,   (0,0))
    cel  = by.get(CELLULAR,  (0,0))
    bulk = by.get(BULK,      (0,0))
    ast  = by.get(ASTROCYTE, (0,0))
    mark = " <<" if ok > best_ok else ""
    print(f"  {name:<38}  {ok:>2}/{n} ({100*ok/n:>5.1f}%)"
          f"  {hyp[0]}/{hyp[1]}  {cel[0]}/{cel[1]}"
          f"  {bulk[0]}/{bulk[1]}  {ast[0]}/{ast[1]}{mark}")
    results[name] = (ok, n, by, dets, overrides, extras, w)
    if ok > best_ok:
        best_ok, best_name, best_cfg = ok, name, (overrides, extras, w)

# ── Détail des erreurs de la meilleure combo ─────────────────────────
print(f"\n>>> Meilleure : {best_name} -> {best_ok}/{n} = {100*best_ok/n:.1f}%")
_, _, _, best_dets, _, _, _ = results[best_name]
print("\n  Erreurs restantes :")
for animal_id, c_id, true_lbl, pred_lbl, cen in best_dets:
    if true_lbl == pred_lbl: continue
    cen2 = _inject_virtual_md(cen)
    print(f"  {animal_id} c{c_id}: GT={true_lbl:<22} pred={pred_lbl:<22}"
          f"  T2*={cen2.get('T2star',float('nan')):+.2f}"
          f"  T2={cen2.get('T2',float('nan')):+.2f}"
          f"  MD={cen2.get('DTI-MD',float('nan')):+.2f}"
          f"  FA={cen2.get('DTI-FA',float('nan')):+.2f}"
          f"  dt={cen2.get('d_topo_norm',float('nan')):.2f}")
