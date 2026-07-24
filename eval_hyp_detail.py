"""
eval_hyp_detail.py — Détail des features MRI sur tous les clusters GT=Hypoxic
et simulation de nouveaux seuils LF.
"""
import sys, os, json, io
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.labeling_ws import (
    build_feature_table, _apply_lfs, weighted_vote,
    LF_REGISTRY, LABELS, ABSTAIN,
    NECROSIS, HYPOXIC, CELLULAR, INFILTRATIVE, BULK,
    _g, _inject_virtual_md,
)

GT_JSON   = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labels_Without_hierarchie.json"
GT_FOLDER = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labeling"
LF_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "lf_weights.json")

with open(LF_WEIGHTS_FILE) as f:
    lf_weights = json.load(f)

with open(GT_JSON, encoding="utf-8") as f:
    raw = f.read()
raw = raw.replace('"Hypoxic/vascular,', '"Hypoxic/vascular",')
gt_all = json.loads(raw)

# ── Collecte de tous les clusters (avec leur vrai label) ─────────────
all_clusters = []   # (animal, c_id, true_label, centroid, parameters)

for animal_id, gt_clusters in gt_all.items():
    cp = os.path.join(GT_FOLDER, animal_id, "V2", "centroids.json")
    if not os.path.exists(cp):
        continue
    with open(cp) as f:
        raw_c = json.load(f)
    centroids  = {int(k): v for k, v in raw_c.items()}
    parameters = [p for p in next(iter(centroids.values())).keys()
                  if p != "d_topo_norm"]
    gt_int = {int(k): v for k, v in gt_clusters.items()}
    for c_id, true_label in gt_int.items():
        if c_id in centroids:
            all_clusters.append((animal_id, c_id, true_label,
                                 centroids[c_id], parameters))

# ── Feature summary des clusters GT=Hypoxic ──────────────────────────
hyp_clusters = [(a, c, cen) for a, c, lbl, cen, _ in all_clusters
                if lbl == HYPOXIC]

print("=" * 72)
print("  FEATURES des clusters GT=Hypoxic/vascular (14 clusters)")
print("=" * 72)
print(f"  {'Animal':<26} {'Cl':>3}  {'T2*':>6} {'T2':>6} {'MD':>6} "
      f"{'FA':>6} {'dtopo':>6}  note")
print("  " + "-" * 70)
for animal, c_id, cen in hyp_clusters:
    cen2 = _inject_virtual_md(cen)
    t2s  = cen2.get("T2star",    float("nan"))
    t2   = cen2.get("T2",        float("nan"))
    md   = cen2.get("DTI-MD",    float("nan"))
    fa   = cen2.get("DTI-FA",    float("nan"))
    dt   = cen2.get("d_topo_norm", float("nan"))
    note = ""
    if t2s != t2s:   note += "no T2* "
    if md  != md:    note += "no MD "
    if dt  != dt:    note += "no dtopo "
    print(f"  {animal:<26} {c_id:>3}  {t2s:>+6.2f} {t2:>+6.2f} {md:>+6.2f} "
          f"{fa:>+6.2f} {dt:>6.2f}  {note}")

# ── Simulation de nouvelles LF hypoxiques ────────────────────────────
# Seuils proposés
_T2S_HYP_NEW      = -0.15   # inchangé
_T2_HYP_NEW       = -0.5    # assoupli (était 0.0)
_T2S_SPATIAL_MAX  =  0.3    # NOUVEAU filtre pour lf_hyp_spatial
_T2S_PERICORE_MAX =  0.5    # NOUVEAU filtre pour lf_hyp_pericore
_T2S_HYP_STRONG   = -0.5    # NOUVEAU seuil fort

_DTOPO_HYP_LO    = 0.03
_DTOPO_HYP_HI    = 0.45
_DTOPO_PERICORE_HI = 0.30
_T2_HYP_EXCL     = 1.5
_MD_HYP_EXCL     = 1.5

def lf_hyp_t2s_t2_NEW(r):
    """T2* bas + T2 pas trop négatif (seuil assoupli à -0.5)."""
    t2s, t2 = _g(r, "T2star"), _g(r, "T2")
    if t2s is None or t2 is None: return ABSTAIN
    return HYPOXIC if (t2s < _T2S_HYP_NEW and t2 >= _T2_HYP_NEW) else ABSTAIN

def lf_hyp_spatial_NEW(r):
    """Zone peri-nécrotique + T2* pas franchement positif."""
    dt  = _g(r, "d_topo_norm")
    t2s = _g(r, "T2star")
    if dt is None: return ABSTAIN
    if t2s is not None and t2s > _T2S_SPATIAL_MAX: return ABSTAIN
    return HYPOXIC if _DTOPO_HYP_LO < dt < _DTOPO_HYP_HI else ABSTAIN

def lf_hyp_pericore_NEW(r):
    """Adjacent au cœur + pas de signal nécrotique extrême + T2* non positif."""
    dt = _g(r, "d_topo_norm")
    if dt is None: return ABSTAIN
    if not (_DTOPO_HYP_LO < dt < _DTOPO_PERICORE_HI): return ABSTAIN
    t2  = _g(r, "T2")
    md  = _g(r, "DTI-MD")
    t2s = _g(r, "T2star")
    if (t2 is not None and t2 > _T2_HYP_EXCL) and (md is not None and md > _MD_HYP_EXCL):
        return ABSTAIN
    if t2s is not None and t2s > _T2S_PERICORE_MAX: return ABSTAIN
    return HYPOXIC

def lf_hyp_t2s_strong_NEW(r):
    """T2* fortement négatif => Hypoxic certain, quelle que soit T2."""
    t2s = _g(r, "T2star")
    if t2s is None: return ABSTAIN
    md = _g(r, "DTI-MD")
    if md is not None and md > _MD_HYP_EXCL: return ABSTAIN
    return HYPOXIC if t2s < _T2S_HYP_STRONG else ABSTAIN

# ── Comparaison avant/après sur tous les clusters ────────────────────
def apply_new_lfs(centroids, parameters):
    """Retourne {c: {lf: vote}} avec les LF originales + les nouvelles."""
    rows, _, _ = build_feature_table(centroids, parameters)
    L_orig = _apply_lfs(rows)
    L_new  = {}
    for row in rows:
        c = row["cluster_id"]
        L_new[c] = dict(L_orig[c])
        L_new[c]["lf_hyp_t2s_t2"]  = lf_hyp_t2s_t2_NEW(row)
        L_new[c]["lf_hyp_spatial"]  = lf_hyp_spatial_NEW(row)
        L_new[c]["lf_hyp_pericore"] = lf_hyp_pericore_NEW(row)
        L_new[c]["lf_hyp_t2s_strong"] = lf_hyp_t2s_strong_NEW(row)
    return rows, L_new

NEW_REGISTRY = LF_REGISTRY + [("lf_hyp_t2s_strong", lf_hyp_t2s_strong_NEW, HYPOXIC)]
new_weights  = dict(lf_weights)
new_weights["lf_hyp_t2s_strong"] = 1.0  # poids provisoire

def weighted_vote_new(rows, L_new):
    eps = {NECROSIS: 0.01, HYPOXIC: 0.01, CELLULAR: 0.01,
           INFILTRATIVE: 0.0, BULK: 0.05}
    probas = {}
    for row in rows:
        c      = row["cluster_id"]
        scores = dict(eps)
        for name, _, _ in NEW_REGISTRY:
            vote = L_new[c].get(name, ABSTAIN)
            if vote != ABSTAIN:
                scores[vote] = scores[vote] + new_weights.get(name, 1.0)
        total = sum(scores.values())
        probas[c] = {lbl: scores[lbl]/total for lbl in LABELS}
    return probas

# ── Évaluation before/after ──────────────────────────────────────────
before_true, before_pred = [], []
after_true,  after_pred  = [], []

for animal_id, gt_clusters in gt_all.items():
    cp = os.path.join(GT_FOLDER, animal_id, "V2", "centroids.json")
    if not os.path.exists(cp):
        continue
    with open(cp) as f:
        raw_c = json.load(f)
    centroids  = {int(k): v for k, v in raw_c.items()}
    parameters = [p for p in next(iter(centroids.values())).keys()
                  if p != "d_topo_norm"]
    gt_int = {int(k): v for k, v in gt_clusters.items()}

    # AVANT
    rows_o, _, _ = build_feature_table(centroids, parameters)
    L_orig = _apply_lfs(rows_o)
    probas_orig = weighted_vote(rows_o, L_orig, weights=lf_weights)

    # APRES
    rows_n, L_new = apply_new_lfs(centroids, parameters)
    probas_new    = weighted_vote_new(rows_n, L_new)

    for c_id, true_label in gt_int.items():
        if c_id not in probas_orig:
            continue
        before_true.append(true_label)
        before_pred.append(max(probas_orig[c_id], key=probas_orig[c_id].get))
        after_true.append(true_label)
        after_pred.append(max(probas_new[c_id],  key=probas_new[c_id].get))

print("\n" + "=" * 72)
print("  SIMULATION — impact des nouveaux seuils")
print("=" * 72)

n = len(before_true)
acc_b = sum(t==p for t,p in zip(before_true, before_pred))
acc_a = sum(t==p for t,p in zip(after_true,  after_pred))
print(f"\n  Precision globale  AVANT : {acc_b}/{n} = {100*acc_b/n:.1f}%")
print(f"  Precision globale  APRES : {acc_a}/{n} = {100*acc_a/n:.1f}%")

print(f"\n  {'Label':<28}  AVANT       APRES")
print("  " + "-" * 55)
for lbl in LABELS:
    idx_b = [i for i,t in enumerate(before_true) if t==lbl]
    idx_a = [i for i,t in enumerate(after_true)  if t==lbl]
    if not idx_b: continue
    cb = sum(before_pred[i]==lbl for i in idx_b)
    ca = sum(after_pred[i]==lbl  for i in idx_a)
    arrow = "  (+)" if ca > cb else ("  (-)" if ca < cb else "")
    print(f"  {lbl:<28}  {cb:>2}/{len(idx_b):>2} ({100*cb/len(idx_b):>5.1f}%)"
          f"   {ca:>2}/{len(idx_a):>2} ({100*ca/len(idx_a):>5.1f}%){arrow}")

print("\n  Detail erreurs Hypoxic APRES :")
for i, (t, p) in enumerate(zip(after_true, after_pred)):
    if t == HYPOXIC and p != HYPOXIC:
        a, c, lbl, cen, _ = all_clusters[i]
        cen2 = _inject_virtual_md(cen)
        print(f"    {a} c{c}: pred={p}  "
              f"T2*={cen2.get('T2star',float('nan')):+.2f}  "
              f"T2={cen2.get('T2',float('nan')):+.2f}  "
              f"MD={cen2.get('DTI-MD',float('nan')):+.2f}  "
              f"dtopo={cen2.get('d_topo_norm',float('nan')):.2f}")

print("\n  Modifications appliquees :")
print("  1. lf_hyp_t2s_t2   : _T2_HYP  0.0  -> -0.5")
print("  2. lf_hyp_spatial   : + filtre T2* <= 0.3")
print("  3. lf_hyp_pericore  : + filtre T2* <= 0.5")
print("  4. lf_hyp_t2s_strong: NOUVELLE LF (T2* < -0.5, ignore T2)")
print("=" * 72)
