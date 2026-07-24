"""
eval_lf_gt.py — Évaluation des LF sur le GT sans hiérarchie.

Pour chaque souris du GT :
  1. Charge centroids.json (V2)
  2. Applique label_clusters_ws (avec lf_weights.json courant)
  3. Compare la prédiction au label GT

Sorties :
  - Précision globale et par label
  - Matrice de confusion
  - Détail des erreurs Hypoxic/vascular (quelles LF ont voté / abstenu)
  - Analyse LF (couverture, conflits)

Usage :
    python eval_lf_gt.py
"""

import sys, os, json, math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.labeling_ws import (
    build_feature_table, _apply_lfs, weighted_vote,
    label_clusters_ws, lf_analysis,
    LF_REGISTRY, LABELS, ABSTAIN,
    NECROSIS, HYPOXIC, CELLULAR, INFILTRATIVE, BULK,
)

# -- Chemins ----------------------------------------------------------
GT_JSON   = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labels_Without_hierarchie.json"
GT_FOLDER = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labeling"
LF_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "lf_weights.json")

# -- Chargement des poids LF ------------------------------------------
with open(LF_WEIGHTS_FILE) as f:
    lf_weights = json.load(f)

# -- Chargement du GT (avec correction de la virgule manquante) ------─
with open(GT_JSON, encoding="utf-8") as f:
    raw = f.read()
# Corrige le bug de syntaxe : "Hypoxic/vascular, → "Hypoxic/vascular",
raw = raw.replace('"Hypoxic/vascular,', '"Hypoxic/vascular",')
gt_all = json.load(raw if isinstance(raw, dict) else __import__('io').StringIO(raw))
# json.load ne prend pas de str directement
import io
gt_all = json.loads(raw)

# -- Collecte des résultats ------------------------------------------─
all_true, all_pred = [], []

# Pour l'analyse Hypoxic
hyp_errors   = []   # (animal, cluster_id, pred, centroids, lf_votes)
hyp_correct  = []

# Pour l'analyse LF globale
all_L        = {}   # {animal_cluster_key: {lf_name: vote}}
all_gt_flat  = {}   # {animal_cluster_key: true_label}

print("=" * 70)
print("  ÉVALUATION LF — GT sans hiérarchie")
print("=" * 70)

for animal_id, gt_clusters in gt_all.items():
    centroids_path = os.path.join(GT_FOLDER, animal_id, "V2", "centroids.json")
    if not os.path.exists(centroids_path):
        print(f"  [!] {animal_id}: centroids.json absent — ignoré")
        continue

    with open(centroids_path) as f:
        raw_centroids = json.load(f)

    centroids = {int(k): v for k, v in raw_centroids.items()}
    parameters = [p for p in next(iter(centroids.values())).keys()
                  if p != "d_topo_norm"]

    gt_int = {int(k): v for k, v in gt_clusters.items()}

    # Appliquer les LF
    rows, _, _ = build_feature_table(centroids, parameters)
    L = _apply_lfs(rows)
    probas = weighted_vote(rows, L, weights=lf_weights)

    for cluster_id, true_label in gt_int.items():
        if cluster_id not in probas:
            continue
        pred_label = max(probas[cluster_id], key=probas[cluster_id].get)
        all_true.append(true_label)
        all_pred.append(pred_label)

        key = f"{animal_id}__c{cluster_id}"
        all_L[key]       = L[cluster_id]
        all_gt_flat[key] = true_label

        if true_label == HYPOXIC:
            entry = {
                "animal"    : animal_id,
                "cluster"   : cluster_id,
                "pred"      : pred_label,
                "proba"     : probas[cluster_id],
                "centroid"  : centroids[cluster_id],
                "lf_votes"  : L[cluster_id],
            }
            if pred_label == HYPOXIC:
                hyp_correct.append(entry)
            else:
                hyp_errors.append(entry)

# -- Précision globale ------------------------------------------------
n_total   = len(all_true)
n_correct = sum(t == p for t, p in zip(all_true, all_pred))
print(f"\nClusters GT évalués : {n_total}")
print(f"Précision globale   : {n_correct}/{n_total} = {100*n_correct/n_total:.1f}%")

# -- Précision par label ----------------------------------------------
print("\n-- Précision par label ----------------------------------------─")
for lbl in LABELS:
    indices = [i for i, t in enumerate(all_true) if t == lbl]
    if not indices:
        continue
    correct = sum(all_pred[i] == lbl for i in indices)
    print(f"  {lbl:<28} {correct:>2}/{len(indices):>2}  "
          f"({100*correct/len(indices):>5.1f}%)")

# -- Matrice de confusion --------------------------------------------─
print("\n-- Matrice de confusion (lignes=GT, colonnes=pred) ------------─")
present     = sorted(set(all_true) | set(all_pred), key=LABELS.index)
col_w       = 10
_row_label  = "GT / Pred"
header      = f"{_row_label:<28}" + "".join(f"{l[:col_w-1]:>{col_w}}" for l in present)
print(header)
print("-" * len(header))
for true_lbl in present:
    row = f"  {true_lbl:<26}"
    for pred_lbl in present:
        cnt = sum(t == true_lbl and p == pred_lbl
                  for t, p in zip(all_true, all_pred))
        cell = f"{cnt:>{col_w}}" if cnt > 0 else f"{'·':>{col_w}}"
        row += cell
    print(row)

# -- Détail des erreurs Hypoxic --------------------------------------─
print(f"\n-- Hypoxic/vascular : {len(hyp_correct)} corrects / "
      f"{len(hyp_errors)} erreurs ------------------")

if hyp_errors:
    print(f"\n  {'Animal':<26} {'Cl':>3}  {'Prédit':<28}  "
          f"T2*    T2     MD     d_topo")
    print("  " + "-" * 85)
    for e in hyp_errors:
        cen  = e["centroid"]
        md   = cen.get("DTI-MD") or (
            (cen.get("DTI-AD", 0) + 2*cen.get("DTI-RD", 0)) / 3
            if "DTI-AD" in cen else None)
        t2s  = cen.get("T2star", float("nan"))
        t2   = cen.get("T2",     float("nan"))
        dtop = cen.get("d_topo_norm", float("nan"))
        md_s = f"{md:+.2f}" if md is not None else "  n/a"
        print(f"  {e['animal']:<26} {e['cluster']:>3}  {e['pred']:<28}  "
              f"{t2s:+.2f}  {t2:+.2f}  {md_s}  {dtop:.2f}")

    # Votes LF sur les clusters Hypoxic mal classés
    print("\n  Votes LF sur les clusters Hypoxic mal classés :")
    lf_names = [n for n, _, _ in LF_REGISTRY]
    header2  = f"  {'Animal+Cl':<30}" + "".join(f"{n[3:10]:>10}" for n in lf_names)
    print(header2)
    print("  " + "-" * (30 + 10 * len(lf_names)))
    for e in hyp_errors:
        key  = f"{e['animal'][:20]}c{e['cluster']}"
        row  = f"  {key:<30}"
        for n in lf_names:
            v = e["lf_votes"][n]
            if v == ABSTAIN:
                cell = "   ·"
            elif v == HYPOXIC:
                cell = "  HYP"
            else:
                cell = f" {v[:4].upper()}"
            row += f"{cell:>10}"
        print(row)

# -- Analyse LF globale ----------------------------------------------─
print("\n-- Analyse LF (couverture / conflits / overlap) ----------------")
lf_stats = lf_analysis(all_L)
print(f"  {'LF':<24} {'Cov':>6} {'n':>4} {'Conf%':>7} {'Ovlp':>6}  Poids")
print("  " + "-" * 60)
for s in lf_stats:
    w = lf_weights.get(s['lf_name'], 1.0)
    print(f"  {s['lf_name']:<24} {s['coverage']:>6.2f} "
          f"{s['n_votes']:>4} {100*s['conflict_rate']:>6.0f}%  "
          f"{s['avg_overlap']:>5.1f}  {w:.3f}")

# -- Résumé hypoxie --------------------------------------------------─
print("\n-- Couverture LF sur clusters GT=Hypoxic ----------------------─")
hyp_all = hyp_correct + hyp_errors
if hyp_all:
    for n, fn, _ in LF_REGISTRY:
        votes = [e["lf_votes"][n] for e in hyp_all]
        fired_hyp  = sum(v == HYPOXIC  for v in votes)
        fired_other= sum(v != ABSTAIN and v != HYPOXIC for v in votes)
        abstained  = sum(v == ABSTAIN  for v in votes)
        print(f"  {n:<24}  HYP={fired_hyp:>2}  autre={fired_other:>2}  "
              f"abstain={abstained:>2}")

print("\n" + "=" * 70)
