# estimate_lf_weights.py
# Reconstruit les centroides depuis les CSV de run existants,
# applique les LFs, affiche l'analyse de couverture/conflit,
# estime les poids depuis les souris GT et sauvegarde lf_weights.json.
#
# Usage (depuis tumor_habitat_V6/) :
#   C:\Users\laboratorio\miniconda3\envs\seg_hab_env\python.exe estimate_lf_weights.py

import os, sys, json
import pandas as pd

# ── Chemins ───────────────────────────────────────────────────────
GT_LABELING_DIR = r"C:\Users\laboratorio\Desktop\projet 1\gt_labeling"
RESULT_SUBDIR   = "V2"
OUTPUT_WEIGHTS  = os.path.join(os.path.dirname(__file__), "lf_weights.json")

# GT file: override with --gt <path> on the command line
_DEFAULT_GT = r"C:\Users\laboratorio\Desktop\projet 1\gt_labels.json"
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--gt", default=_DEFAULT_GT)
_args, _ = _parser.parse_known_args()
GT_LABELS_JSON = _args.gt

# Rend le module pipeline importable
sys.path.insert(0, os.path.dirname(__file__))
from pipeline.labeling_ws import (
    build_feature_table, _apply_lfs, lf_analysis,
    weighted_vote, estimate_weights_from_multi_gt,
    LF_REGISTRY, ABSTAIN, LABELS,
)


# ── Reconstruction des centroides depuis un CSV de run ────────────
def load_centroids_from_csv(csv_path, meta_path):
    """
    Charge les centroides depuis centroids.json (préféré, contient d_topo_norm)
    ou habitats_result.csv en fallback.

    centroids.json est écrit par save_centroids_json() après inject_dtopo_centroids,
    donc il inclut d_topo_norm quand le run a été fait avec segmentation hiérarchique.

    Si centroids.json existe mais contient MOINS de clusters que le CSV
    (run ultérieur a écrasé le fichier), on préfère le CSV pour conserver
    tous les clusters annotés dans gt_labels.json.

    Retourne (centroids, parameters) ou (None, None) si fichiers manquants.
    """
    run_dir   = os.path.dirname(csv_path)
    json_path = os.path.join(run_dir, "centroids.json")

    # ── Load JSON (has d_topo_norm when available) ────────────────
    json_centroids = None
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        json_centroids = {int(k): dict(v) for k, v in raw.items()}

    # ── Load CSV ──────────────────────────────────────────────────
    csv_centroids = None
    csv_params    = None
    if os.path.exists(csv_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        parameters = list(meta.get("scaling_info", {}).keys())
        if parameters:
            df        = pd.read_csv(csv_path)
            feat_cols = [p for p in parameters if p in df.columns]
            if "Habitat" in df.columns and feat_cols:
                csv_centroids = {}
                for cluster_id, group in df.groupby("Habitat"):
                    csv_centroids[int(cluster_id)] = {
                        p: float(group[p].mean()) for p in feat_cols
                    }
                csv_params = feat_cols

    # ── Choose source: prefer JSON but fall back to CSV when JSON
    #    has fewer clusters (stale overwrite from a later short run) ──
    if json_centroids is not None and csv_centroids is not None:
        if len(json_centroids) >= len(csv_centroids):
            centroids  = json_centroids
            parameters = sorted({p for feat in json_centroids.values() for p in feat})
        else:
            print(f"    [!] centroids.json has {len(json_centroids)} clusters "
                  f"but CSV has {len(csv_centroids)} — using CSV "
                  f"(d_topo_norm unavailable for this animal)")
            centroids  = csv_centroids
            parameters = csv_params
    elif json_centroids is not None:
        centroids  = json_centroids
        parameters = sorted({p for feat in json_centroids.values() for p in feat})
    elif csv_centroids is not None:
        centroids  = csv_centroids
        parameters = csv_params
    else:
        return None, None

    return centroids, parameters


# ── Chargement des 10 souris GT ───────────────────────────────────
def load_gt_animals():
    with open(GT_LABELS_JSON, encoding="utf-8") as f:
        gt_json = json.load(f)

    animal_runs = {}
    missing     = []

    for animal_id in gt_json:
        animal_dir = os.path.join(GT_LABELING_DIR, animal_id, RESULT_SUBDIR)
        csv_path   = os.path.join(animal_dir, "habitats_result.csv")
        meta_path  = os.path.join(animal_dir, "run_meta.json")

        centroids, parameters = load_centroids_from_csv(csv_path, meta_path)
        if centroids is None:
            missing.append(animal_id)
            continue

        animal_runs[animal_id] = (centroids, parameters)
        print(f"  OK  {animal_id}  ({len(centroids)} clusters, "
              f"params={parameters})")

    if missing:
        print(f"\n  ATTENTION — fichiers manquants pour : {missing}")

    return animal_runs, gt_json


# ── Analyse de couverture/conflit (sans GT) ───────────────────────
def print_lf_analysis(animal_runs):
    # Construit une L-matrix combinée sur toutes les souris
    combined_L = {}
    for animal_id, (centroids, parameters) in animal_runs.items():
        rows, _, _ = build_feature_table(centroids, parameters)
        L = _apply_lfs(rows)
        for c, votes in L.items():
            combined_L[(animal_id, c)] = votes

    analysis = lf_analysis(combined_L)

    print(f"\n{'LF':<26} {'cible':<22} {'cov':>5} {'conflit':>8} "
          f"{'overlap':>8} {'n':>5}")
    print("-" * 75)
    for r in analysis:
        flag = ""
        if r["coverage"] > 0.80:
            flag = "  !! trop large"
        elif r["coverage"] < 0.05:
            flag = "  !! quasi-inutile"
        elif r["conflict_rate"] > 0.50:
            flag = "  !! conflit eleve"
        print(f"{r['lf_name']:<26} {r['target_label']:<22} "
              f"{r['coverage']:>5.2f} {r['conflict_rate']:>8.2f} "
              f"{r['avg_overlap']:>8.1f} {r['n_votes']:>5}{flag}")

    return combined_L


# ── Estimation des poids depuis GT ────────────────────────────────
def estimate_and_save_weights(animal_runs, gt_json):
    weights = estimate_weights_from_multi_gt(animal_runs, gt_json)

    print(f"\n{'LF':<26} {'poids (précision)':>18}")
    print("-" * 46)
    for name, w in sorted(weights.items(), key=lambda x: x[1]):
        bar   = "#" * int(w * 20)
        note  = "  !! faible -- a revoir" if w < 0.40 else ""
        print(f"{name:<26} {w:>8.3f}  {bar}{note}")

    with open(OUTPUT_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"\nPoids sauvegardes -> {OUTPUT_WEIGHTS}")

    return weights


# ── Normalisation de label : supprime les sous-types entre parenthèses ──
def _base_label(label):
    """
    'Necrosis (hemorrhagic)' → 'Necrosis'
    'Infiltrative (edematous)' → 'Infiltrative'
    'Bulk tumor' → 'Bulk tumor'   (unchanged)
    Lets both methods be compared at the same granularity level.
    """
    return label.split(" (")[0].strip()


# ── LOO-CV : vote pondéré WS ──────────────────────────────────────
def loo_cross_validation(animal_runs, gt_json):
    """
    Leave-one-out cross-validation for the weighted-vote WS labeler.

    For each GT animal i:
      1. Estimate LF weights on ALL OTHER GT animals (training set).
      2. Predict on animal i (test set).

    Reports per-animal accuracy and overall LOO accuracy.
    Returns (correct, total, overall_accuracy).
    """
    animal_ids = [a for a in animal_runs if a in gt_json]
    if len(animal_ids) < 2:
        print("  LOO-CV requires at least 2 GT animals. Skipping.")
        return 0, 0, 0.0

    total_correct = total_count = 0
    errors = []

    print(f"\n{'Animal':<22} {'OK':>4} {'N':>4} {'Acc':>7}  Erreurs")
    print("-" * 60)

    for held_out in animal_ids:
        train_runs = {a: animal_runs[a] for a in animal_ids if a != held_out}

        # Weights trained on all animals except the held-out one
        loo_weights = estimate_weights_from_multi_gt(train_runs, gt_json)

        centroids, parameters = animal_runs[held_out]
        rows, _, _ = build_feature_table(centroids, parameters)
        L          = _apply_lfs(rows)
        probas     = weighted_vote(rows, L, loo_weights)

        animal_gt = {int(k): v for k, v in gt_json[held_out].items()}
        a_correct = a_total = 0
        row_errors = []

        for c, true_lbl in animal_gt.items():
            if c not in probas:
                continue
            pred = max(probas[c], key=probas[c].get)
            # Compare at base-label level so sub-type differences don't penalize WS
            if _base_label(pred) == _base_label(true_lbl):
                a_correct += 1
                total_correct += 1
            else:
                row_errors.append(
                    f"c{c}: pred={pred} true={true_lbl} p={probas[c][pred]:.2f}")
                errors.append(
                    f"  {held_out}  cluster {c}: "
                    f"pred={pred}  true={true_lbl}  p={probas[c][pred]:.2f}")
            a_total += 1
            total_count += 1

        acc = a_correct / a_total if a_total > 0 else 0.0
        err_str = "  " + " | ".join(row_errors) if row_errors else ""
        print(f"{held_out:<22} {a_correct:>4} {a_total:>4} {acc:>7.1%}{err_str}")

    overall = total_correct / total_count if total_count > 0 else 0.0
    print("-" * 60)
    print(f"{'LOO-CV TOTAL':<22} {total_correct:>4} {total_count:>4} {overall:>7.1%}")

    return total_correct, total_count, overall


# ── Baseline : règles 3-passes (déterministe, sans entraînement) ───
def three_pass_accuracy(animal_runs, gt_json):
    """
    Evaluate classify_three_passes (rule-based, no training) against GT.

    features_by_cluster is passed as empty lists so the anchor falls back
    to the min-L2 cluster (same heuristic as the WS median anchor).
    Comparison uses base labels so sub-types don't bias the score.
    Returns (correct, total, overall_accuracy).
    """
    from pipeline.labeling import classify_three_passes

    total_correct = total_count = 0
    errors = []

    print(f"\n{'Animal':<22} {'OK':>4} {'N':>4} {'Acc':>7}  Erreurs")
    print("-" * 60)

    for animal_id, (centroids, parameters) in animal_runs.items():
        if animal_id not in gt_json:
            continue
        animal_gt = {int(k): v for k, v in gt_json[animal_id].items()}

        # Empty feature arrays → anchor selection falls back to min-L2
        features_by_cluster = {c: [] for c in centroids}
        results = classify_three_passes(centroids, parameters, features_by_cluster)

        a_correct = a_total = 0
        row_errors = []

        for c, true_lbl in animal_gt.items():
            if c not in results:
                continue
            pred = results[c]['label']
            if _base_label(pred) == _base_label(true_lbl):
                a_correct += 1
                total_correct += 1
            else:
                row_errors.append(f"c{c}: pred={pred} true={true_lbl}")
                errors.append(
                    f"  {animal_id}  cluster {c}: pred={pred}  true={true_lbl}")
            a_total += 1
            total_count += 1

        acc = a_correct / a_total if a_total > 0 else 0.0
        err_str = "  " + " | ".join(row_errors) if row_errors else ""
        print(f"{animal_id:<22} {a_correct:>4} {a_total:>4} {acc:>7.1%}{err_str}")

    overall = total_correct / total_count if total_count > 0 else 0.0
    print("-" * 60)
    print(f"{'3-PASS TOTAL':<22} {total_correct:>4} {total_count:>4} {overall:>7.1%}")

    return total_correct, total_count, overall


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Chargement des souris GT ===")
    animal_runs, gt_json = load_gt_animals()

    if not animal_runs:
        print("Aucune souris GT chargée. Vérifiez les chemins.")
        sys.exit(1)

    print(f"\n=== Analyse LF (toutes souris GT combinées) ===")
    print("Couverture < 0.05 ou > 0.80, conflit > 0.50 = LF pathologique")
    combined_L = print_lf_analysis(animal_runs)

    print(f"\n=== Estimation des poids (toutes souris, pour lf_weights.json) ===")
    weights = estimate_and_save_weights(animal_runs, gt_json)

    print(f"\n=== LOO-CV — vote pondéré WS ===")
    print("Entraînement sur N-1 souris, évaluation sur la 1 restante.")
    print("Comparaison au niveau du label de base (sans sous-type).")
    loo_ok, loo_n, loo_acc = loo_cross_validation(animal_runs, gt_json)

    print(f"\n=== Baseline — règles 3-passes (déterministe) ===")
    print("Aucun entraînement — même comparaison au niveau label de base.")
    tp_ok, tp_n, tp_acc = three_pass_accuracy(animal_runs, gt_json)

    print("\n" + "=" * 48)
    print("  COMPARAISON FINALE")
    print("=" * 48)
    print(f"  {'Méthode':<28} {'Acc':>7}  (OK/N)")
    print(f"  {'LOO-CV WS (pondéré)':<28} {loo_acc:>7.1%}  ({loo_ok}/{loo_n})")
    print(f"  {'3-passes (règles)':<28} {tp_acc:>7.1%}  ({tp_ok}/{tp_n})")
    winner = "WS pondéré" if loo_acc >= tp_acc else "3-passes"
    print(f"\n  -> Meilleure méthode : {winner}")
    print("=" * 48)
    print("\nTerminé. Utilisez lf_weights.json dans label_clusters_ws(weights=...).")
