"""
test_discovery_synthetic.py
----------------------------
End-to-end test: Classic segmentation -> Discovery + Level-2 sub-clustering
on the synthetic cohort in synthetic_cohort/.

Outputs (in synthetic_cohort/_discovery_test/discovery/):
  multi_mouse_discovery_report.pdf
  multi_mouse_discovery_level2_report.pdf
  gt_confusion_matrix.png   <- GT comparison
  gt_per_mouse_ari.png      <- per-mouse ARI bar chart

Run:
    python test_discovery_synthetic.py
"""

import os
import sys
import glob

# ── Conda env guard ───────────────────────────────────────────────────────────
PYTHON_EXE = r"C:\Users\laboratorio\miniconda3\envs\seg_hab_env\python.exe"
if sys.executable != PYTHON_EXE and os.path.exists(PYTHON_EXE):
    import subprocess
    result = subprocess.run([PYTHON_EXE, __file__] + sys.argv[1:])
    sys.exit(result.returncode)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pipeline.loading    import load_data, robust_scale, resolve_dti_parameters, save_scaling_info
from pipeline.spatial    import prepare_features
from pipeline.clustering import run_clustering, run_clustering_with_size_guard
from pipeline.selection  import select_optimal_k
from pipeline.export     import export_results
from multi_mouse_discovery import run_discovery
from pipeline.joint_clustering import run_joint_pipeline
from config import K_RANGE, N_INIT_GMM, N_REFS_GAP, N_JOBS_GAP, GMM_SILHOUETTE_GUARD, KMEANS_GAP_GUARD

# ── Configuration ─────────────────────────────────────────────────────────────

SYNTHETIC_ROOT = os.path.join(os.path.dirname(__file__), "synthetic_cohort")
OUT_ROOT       = os.path.join(os.path.dirname(__file__), "synthetic_cohort", "_discovery_test")
METHOD         = "gmm"
N_INIT         = N_INIT_GMM

DISC_K_RANGE   = range(2, 7)
DISC_METHOD    = "gmm"
DISC_CRITERION = "elbow"
DISC_MERGE     = "suggest"
SUB_CLUST      = True
K_SUB_MAX      = 3
SUB_MIN_VOX    = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


# ── Step 1: Classic per mouse ─────────────────────────────────────────────────

def run_classic_for_mouse(mouse_dir: str, out_dir: str) -> str | None:
    """Run Classic pipeline on one synthetic mouse using automatic K selection.
    Mirrors what the real app does: select_optimal_k -> run_clustering_with_size_guard.
    Returns path to habitats_result.csv, or None on failure."""
    csv_files = sorted(glob.glob(os.path.join(mouse_dir, "*_roi_data.csv")))
    if not csv_files:
        log(f"  [!] No roi_data.csv found in {mouse_dir}")
        return None

    mouse_name = os.path.basename(mouse_dir)
    log(f"\n{'-'*60}")
    log(f"Classic -- {mouse_name}  ({len(csv_files)} params)")

    try:
        parameters, df_all = load_data(csv_files)
    except Exception as exc:
        log(f"  [!] load_data failed: {exc}")
        return None

    if len(df_all) == 0:
        log("  [!] No voxels loaded")
        return None

    df_scaled, features_scaled, scaling_info = robust_scale(df_all, parameters)
    parameters, _ = resolve_dti_parameters(parameters)
    features_scaled = df_scaled[parameters].fillna(0).to_numpy()

    try:
        features, dtopo_meta = prepare_features(features_scaled, df_all, parameters,
                                                n_init_gmm=N_INIT)
    except Exception as exc:
        log(f"  [!] prepare_features failed ({exc}), using raw features")
        features    = features_scaled
        dtopo_meta  = None

    os.makedirs(out_dir, exist_ok=True)

    try:
        best_k, gmm_cache = select_optimal_k(
            features, method=METHOD, df_all=df_all,
            k_range=K_RANGE, n_refs=N_REFS_GAP, n_jobs=N_JOBS_GAP,
            spatial_weight=0.0, n_init_gmm=N_INIT,
            gmm_silhouette_guard=GMM_SILHOUETTE_GUARD,
            kmeans_gap_guard=KMEANS_GAP_GUARD,
            output_dir=out_dir,
        )
        log(f"  -> K optimal = {best_k}")
    except Exception as exc:
        log(f"  [!] select_optimal_k failed: {exc}")
        return None

    try:
        labels, _, best_k = run_clustering_with_size_guard(
            features, features_scaled, df_all, best_k,
            method=METHOD, spatial_weight=0.0, n_init_gmm=N_INIT,
            parameters=parameters, dtopo_meta=dtopo_meta,
            k_min=min(K_RANGE), gmm_cache=gmm_cache,
        )
    except Exception as exc:
        log(f"  [!] run_clustering_with_size_guard failed: {exc}")
        return None

    df_scaled = df_scaled.copy()
    df_scaled["Habitat"] = labels
    label_dict = {int(l): f"H{l}" for l in np.unique(labels)}

    try:
        export_results(df_scaled, parameters, label_dict, out_dir)
        save_scaling_info(scaling_info, out_dir)
    except Exception as exc:
        log(f"  [!] export_results failed: {exc}")
        return None

    result_csv = os.path.join(out_dir, "habitats_result.csv")
    log(f"  -> {result_csv}")
    return result_csv


# ── Step 3: Ground-truth comparison ──────────────────────────────────────────

# Colour for each biological habitat type
_GT_COLORS = {
    "necrotic"    : "#E74C3C",
    "hypoxic"     : "#F39C12",
    "proliferative": "#2ECC71",
    "viable"      : "#3498DB",
    "edematous"   : "#9B59B6",
}
_DIVMAP_BLUE = LinearSegmentedColormap.from_list(
    "blues2", ["#FFFFFF", "#1565C0"], N=256)


def compare_with_ground_truth(disc_out: str,
                               mouse_dirs: list,
                               csv_paths: list) -> None:
    """
    Compare discovered meta-habitats against per-voxel ground-truth labels.

    Pipeline:
      habitats_result.csv  (X, Y, Slice, Habitat)
      ground_truth.csv     (X, Y, Slice, habitat_type)
        -> merge on (X, Y, Slice)
        -> map Habitat -> meta_habitat via meta_habitat_assignments.csv
        -> compute ARI, Homogeneity, Completeness, V-measure
        -> plot confusion matrix + per-mouse ARI bar chart
    """
    from sklearn.metrics import (adjusted_rand_score, homogeneity_score,
                                  completeness_score, v_measure_score)

    log("\n" + "=" * 60)
    log("STEP 3 -- Ground-truth quantitative comparison")
    log("=" * 60)

    # Load discovery assignments: mouse, cluster_id -> meta_habitat
    asgn_path = os.path.join(disc_out, "meta_habitat_assignments.csv")
    if not os.path.exists(asgn_path):
        log("[GT] meta_habitat_assignments.csv not found -- skipping")
        return
    assignments = pd.read_csv(asgn_path)
    lookup = {(str(row["mouse"]), int(row["cluster_id"])): str(row["meta_habitat"])
              for _, row in assignments.iterrows()}

    all_true: list = []
    all_pred: list = []
    per_mouse: dict = {}   # mouse_name -> DataFrame(habitat_type, meta_habitat)

    for mdir, csv_path in zip(mouse_dirs, csv_paths):
        mouse_name = os.path.basename(os.path.dirname(csv_path))
        gt_path    = os.path.join(mdir, "ground_truth.csv")

        if not os.path.exists(gt_path):
            log(f"  [!] {mouse_name}: no ground_truth.csv")
            continue

        try:
            gt  = pd.read_csv(gt_path)[["X", "Y", "Slice", "habitat_type"]]
            hab = pd.read_csv(csv_path)[["X", "Y", "Slice", "Habitat"]]
        except Exception as exc:
            log(f"  [!] {mouse_name}: load error -- {exc}")
            continue

        merged = hab.merge(gt, on=["X", "Y", "Slice"], how="inner")
        if len(merged) == 0:
            log(f"  [!] {mouse_name}: no voxels matched on (X,Y,Slice)")
            continue

        merged = merged.copy()
        merged["meta_habitat"] = merged["Habitat"].apply(
            lambda c: lookup.get((mouse_name, int(c)))
        )
        merged = merged.dropna(subset=["meta_habitat"])
        if len(merged) == 0:
            continue

        t = merged["habitat_type"].tolist()
        p = merged["meta_habitat"].tolist()

        ari_m = adjusted_rand_score(t, p)
        hom_m = homogeneity_score(t, p)
        per_mouse[mouse_name] = {
            "ari": ari_m, "hom": hom_m,
            "n": len(merged), "df": merged,
        }
        log(f"  {mouse_name:<28} {len(merged):>6,} vox  "
            f"ARI={ari_m:+.3f}  Hom={hom_m:.3f}")

        all_true.extend(t)
        all_pred.extend(p)

    if not all_true:
        log("[GT] No voxels available for comparison.")
        return

    # ── Global metrics ────────────────────────────────────────────────────────
    ari  = adjusted_rand_score(all_true, all_pred)
    hom  = homogeneity_score(all_true, all_pred)
    comp = completeness_score(all_true, all_pred)
    vm   = v_measure_score(all_true, all_pred)

    log(f"\n  ARI          = {ari:.3f}   (1=perfect, 0=random, <0=worse than random)")
    log(f"  Homogeneity  = {hom:.3f}   (each MH contains only one GT type)")
    log(f"  Completeness = {comp:.3f}   (each GT type lives in one MH)")
    log(f"  V-measure    = {vm:.3f}   (harmonic mean of Hom + Comp)")

    # ── Figure 1: confusion matrix ────────────────────────────────────────────
    true_types = sorted(set(all_true))
    pred_types = sorted(set(all_pred),
                        key=lambda x: int(x.replace("MH", "")) if x.startswith("MH") else 99)
    t_idx = {t: i for i, t in enumerate(true_types)}
    p_idx = {p: i for i, p in enumerate(pred_types)}

    mat = np.zeros((len(true_types), len(pred_types)), dtype=int)
    for t, p in zip(all_true, all_pred):
        mat[t_idx[t], p_idx[p]] += 1

    row_sums = mat.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    mat_norm = mat / row_sums

    fig, ax = plt.subplots(figsize=(max(6, len(pred_types) * 1.3 + 2),
                                    max(4, len(true_types) * 0.75 + 2.2)))
    fig.patch.set_facecolor("#F7F8FA")

    im = ax.imshow(mat_norm, cmap=_DIVMAP_BLUE, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pred_types)))
    ax.set_xticklabels(pred_types, fontsize=9)
    ax.set_yticks(range(len(true_types)))

    # Colour y-tick labels by biological type
    ax.set_yticklabels(true_types, fontsize=9)
    for ytick, lbl in zip(ax.get_yticklabels(), true_types):
        ytick.set_color(_GT_COLORS.get(lbl, "#333"))
        ytick.set_fontweight("bold")

    ax.set_xlabel("Discovered meta-habitat", fontsize=10)
    ax.set_ylabel("Ground-truth habitat type", fontsize=10)
    ax.set_title(
        f"Discovery vs Ground Truth\n"
        f"ARI = {ari:.3f}   Hom = {hom:.3f}   "
        f"Comp = {comp:.3f}   V = {vm:.3f}\n"
        f"(row-normalised — each row sums to 100%)",
        fontsize=9.5, fontweight="bold",
    )

    for i, t in enumerate(true_types):
        for j in range(len(pred_types)):
            v   = mat_norm[i, j]
            raw = mat[i, j]
            if raw == 0:
                continue
            ax.text(j, i, f"{raw:,}\n({v*100:.0f}%)",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else "#222")

    fig.colorbar(im, ax=ax, label="Fraction of GT-type voxels assigned to MH",
                 shrink=0.8)
    fig.tight_layout()
    conf_path = os.path.join(disc_out, "gt_confusion_matrix.png")
    fig.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\n  Confusion matrix  -> {conf_path}")

    # ── Figure 2: per-mouse ARI + composition bars ────────────────────────────
    names   = list(per_mouse.keys())
    aris    = [per_mouse[n]["ari"] for n in names]
    homs    = [per_mouse[n]["hom"] for n in names]
    n_voxs  = [per_mouse[n]["n"]   for n in names]

    bar_colors = ["#2ECC71" if a >= 0.5 else "#F39C12" if a >= 0.25
                  else "#E74C3C" for a in aris]

    fig2, axes2 = plt.subplots(2, 1, figsize=(max(9, len(names) * 0.95 + 1.5), 7),
                               gridspec_kw={"height_ratios": [2, 1]})
    fig2.patch.set_facecolor("#F7F8FA")

    # Top: ARI bars
    ax_ari = axes2[0]
    bars = ax_ari.bar(names, aris, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.6)
    ax_ari.bar(names, homs, color="none", edgecolor="#555",
               linewidth=1.2, linestyle="--", label="Homogeneity")
    ax_ari.axhline(ari,  color="#1C1C1E", linestyle="--", linewidth=1.6,
                   label=f"Overall ARI = {ari:.3f}")
    ax_ari.axhline(0,    color="#AAA", linewidth=0.7)
    ax_ari.set_ylim(-0.15, 1.08)
    ax_ari.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_ari.set_ylabel("Score", fontsize=9)
    ax_ari.set_title("Per-mouse: ARI (bars) and Homogeneity (border) vs Ground Truth",
                     fontsize=10, fontweight="bold")
    ax_ari.legend(fontsize=8, framealpha=0.9)
    for bar, nv in zip(bars, n_voxs):
        h = bar.get_height()
        ax_ari.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.03 if h >= 0 else -0.07),
                    f"{nv:,}v", ha="center", fontsize=6.5, color="#555")

    # Annotation: green>=0.5, orange>=0.25, red<0.25
    from matplotlib.patches import Patch
    legend_q = [Patch(facecolor="#2ECC71", label="ARI >= 0.50  (good)"),
                Patch(facecolor="#F39C12", label="ARI >= 0.25  (moderate)"),
                Patch(facecolor="#E74C3C", label="ARI <  0.25  (poor)")]
    ax_ari.legend(handles=legend_q + [
        plt.Line2D([0], [0], color="#1C1C1E", lw=1.6, linestyle="--",
                   label=f"Overall ARI = {ari:.3f}"),
        plt.Line2D([0], [0], color="#555", lw=1.2, linestyle="--",
                   label="Homogeneity (per mouse)"),
    ], fontsize=7.5, framealpha=0.9, ncol=2)

    # Bottom: stacked bar — GT type composition per mouse
    ax_comp = axes2[1]
    x = np.arange(len(names))
    bottoms = np.zeros(len(names))

    for gt_type in true_types:
        fracs = []
        for mn in names:
            df_m = per_mouse[mn]["df"]
            tot  = len(df_m)
            frac = (df_m["habitat_type"] == gt_type).sum() / max(tot, 1)
            fracs.append(frac)
        color = _GT_COLORS.get(gt_type, "#999")
        ax_comp.bar(x, fracs, bottom=bottoms, color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.4, label=gt_type)
        bottoms += np.array(fracs)

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_comp.set_ylabel("GT type fraction", fontsize=9)
    ax_comp.set_ylim(0, 1.05)
    ax_comp.set_title("Ground-truth habitat composition per mouse", fontsize=9)
    ax_comp.legend(title="GT type", fontsize=7, title_fontsize=7,
                   bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)

    fig2.tight_layout()
    ari_path = os.path.join(disc_out, "gt_per_mouse_ari.png")
    fig2.savefig(ari_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log(f"  Per-mouse ARI   -> {ari_path}")

    # ── Figure 3: GT type purity per meta-habitat ─────────────────────────────
    # For each MH, show the fraction of voxels coming from each GT type
    fig3, ax3 = plt.subplots(figsize=(max(6, len(pred_types) * 1.4 + 1.5), 4.5))
    fig3.patch.set_facecolor("#F7F8FA")

    mh_totals = mat.sum(axis=0).astype(float)
    mh_totals[mh_totals == 0] = 1
    mat_col_norm = mat / mh_totals   # col-normalised: GT composition of each MH

    x = np.arange(len(pred_types))
    bottoms = np.zeros(len(pred_types))
    for i, gt_type in enumerate(true_types):
        color = _GT_COLORS.get(gt_type, "#999")
        ax3.bar(x, mat_col_norm[i], bottom=bottoms, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4, label=gt_type)
        bottoms += mat_col_norm[i]

    ax3.set_xticks(x)
    ax3.set_xticklabels(pred_types, fontsize=9)
    ax3.set_ylim(0, 1.08)
    ax3.set_ylabel("Fraction of MH voxels", fontsize=9)
    ax3.set_xlabel("Meta-habitat", fontsize=9)
    ax3.set_title(
        "GT type purity per meta-habitat\n"
        "(ideal: each bar = one colour = one biological type)",
        fontsize=10, fontweight="bold"
    )
    ax3.legend(title="GT habitat type", fontsize=8, title_fontsize=8,
               bbox_to_anchor=(1.01, 1), loc="upper left")
    ax3.axhline(1, color="#AAA", linewidth=0.7)

    fig3.tight_layout()
    purity_path = os.path.join(disc_out, "gt_mh_purity.png")
    fig3.savefig(purity_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    log(f"  MH purity       -> {purity_path}")

    # ── Text summary ──────────────────────────────────────────────────────────
    log("\n  Ideal result: each meta-habitat = one GT type (homogeneous bars)")
    log("  Interpretation guide:")
    log(f"    ARI  {ari:.3f} : " + (
        "good" if ari >= 0.5 else "moderate" if ari >= 0.25 else "poor"
    ) + " -- random baseline = 0")
    log(f"    Hom  {hom:.3f} : " + (
        "MH are biologically pure" if hom >= 0.7
        else "MH mix some GT types" if hom >= 0.4
        else "MH are biologically mixed"
    ))
    log(f"    Comp {comp:.3f} : " + (
        "each GT type is well concentrated" if comp >= 0.7
        else "GT types are spread across MH" if comp >= 0.4
        else "GT types are heavily split"
    ))


# ── Step 4: Joint GT comparison ──────────────────────────────────────────────

def compare_joint_with_gt(jt_results: dict, mouse_dirs: list) -> None:
    """
    Compare Joint clustering outputs against per-voxel ground truth.
    jt_results : {norm_name: run_joint_pipeline result dict}
    mouse_dirs : list of raw synthetic mouse directories (for ground_truth.csv)
    """
    from sklearn.metrics import (adjusted_rand_score, homogeneity_score,
                                  completeness_score, v_measure_score)

    log("\n" + "=" * 60)
    log("STEP 4b -- Joint vs Ground Truth")
    log("=" * 60)

    # GT map: mouse_name -> DataFrame(X, Y, Slice, habitat_type)
    gt_map: dict = {}
    for mdir in mouse_dirs:
        gt_path = os.path.join(mdir, "ground_truth.csv")
        if os.path.exists(gt_path):
            gt_map[os.path.basename(mdir)] = pd.read_csv(gt_path)

    summary_rows: list = []

    for norm, res in jt_results.items():
        df_pooled = res["df_pooled"]
        best_k    = res["best_k"]

        all_true: list = []
        all_pred: list = []
        per_mouse_ari: dict = {}

        for mouse_name, gt_df in gt_map.items():
            df_m = df_pooled[df_pooled["mouse_name"] == mouse_name].copy()
            if len(df_m) == 0:
                continue
            merged = df_m.merge(
                gt_df[["X", "Y", "Slice", "habitat_type"]],
                on=["X", "Y", "Slice"], how="inner"
            )
            if len(merged) == 0:
                continue
            t = merged["habitat_type"].tolist()
            p = merged["Habitat"].astype(str).tolist()
            ari_m = adjusted_rand_score(t, p)
            per_mouse_ari[mouse_name] = ari_m
            all_true.extend(t)
            all_pred.extend(p)

        if not all_true:
            log(f"  [{norm}] No voxels matched.")
            continue

        ari  = adjusted_rand_score(all_true, all_pred)
        hom  = homogeneity_score(all_true, all_pred)
        comp = completeness_score(all_true, all_pred)
        vm   = v_measure_score(all_true, all_pred)
        mean_mouse_ari = float(np.mean(list(per_mouse_ari.values())))

        log(f"\n  [{norm}]  K={best_k}")
        log(f"    ARI global       = {ari:.3f}")
        log(f"    Homogeneity      = {hom:.3f}")
        log(f"    Completeness     = {comp:.3f}")
        log(f"    V-measure        = {vm:.3f}")
        log(f"    Mean per-mouse ARI = {mean_mouse_ari:.3f}")
        for mn, a in per_mouse_ari.items():
            log(f"      {mn:<28} ARI={a:+.3f}")

        summary_rows.append({
            "normalization"   : norm,
            "K"               : best_k,
            "ARI_global"      : round(ari, 3),
            "Homogeneity"     : round(hom, 3),
            "Completeness"    : round(comp, 3),
            "V_measure"       : round(vm, 3),
            "Mean_mouse_ARI"  : round(mean_mouse_ari, 3),
        })

    # ── Comparison table figure ───────────────────────────────────────────────
    if not summary_rows:
        return

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * len(summary_rows)))
    fig.patch.set_facecolor("#F7F8FA")
    ax.axis("off")

    cols  = ["Normalization", "K", "ARI global", "Homogeneity",
             "Completeness", "V-measure", "Mean/mouse ARI"]
    rows  = [[r["normalization"], r["K"], r["ARI_global"], r["Homogeneity"],
              r["Completeness"], r["V_measure"], r["Mean_mouse_ARI"]]
             for r in summary_rows]

    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Colour best value per metric column (green = highest)
    metric_cols = list(range(2, 7))
    for j in metric_cols:
        vals = [float(rows[i][j]) for i in range(len(rows))]
        best_i = int(np.argmax(vals))
        tbl[best_i + 1, j].set_facecolor("#C8E6C9")

    # Header colour
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Joint clustering — normalization comparison vs ground truth\n"
        "(green cell = best score for that metric)",
        fontsize=10, fontweight="bold", pad=8,
    )
    fig.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "synthetic_cohort", "_discovery_test",
        "joint_normalization_comparison.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\n  Comparison table -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 1 -- Classic segmentation (per mouse)")
    log("=" * 60)

    mouse_dirs = sorted([
        d for d in glob.glob(os.path.join(SYNTHETIC_ROOT, "Syn*"))
        if os.path.isdir(d)
    ])
    if not mouse_dirs:
        log(f"[ERROR] No SynM* folders in {SYNTHETIC_ROOT}")
        sys.exit(1)
    log(f"Found {len(mouse_dirs)} synthetic mice.")

    # Keep parallel lists so GT comparison can match dirs to csv paths
    valid_mouse_dirs: list = []
    csv_paths: list = []

    for mdir in mouse_dirs:
        mname   = os.path.basename(mdir)
        out_dir = os.path.join(OUT_ROOT, "classic", mname)
        result  = run_classic_for_mouse(mdir, out_dir)
        if result:
            valid_mouse_dirs.append(mdir)
            csv_paths.append(result)

    log(f"\nClassic done: {len(csv_paths)}/{len(mouse_dirs)} mice OK")

    if len(csv_paths) < 2:
        log("[ERROR] Need at least 2 valid mice for Discovery.")
        sys.exit(1)

    log("\n" + "=" * 60)
    log("STEP 2 -- Multi-mouse Discovery + Level-2 sub-clustering")
    log("=" * 60)

    disc_out = os.path.join(OUT_ROOT, "discovery")
    os.makedirs(disc_out, exist_ok=True)

    try:
        results = run_discovery(
            csv_paths        = csv_paths,
            k_override       = None,
            k_range          = DISC_K_RANGE,
            n_init           = N_INIT,
            method           = DISC_METHOD,
            output_dir       = disc_out,
            log              = log,
            criterion        = DISC_CRITERION,
            merge_mode       = DISC_MERGE,
            cosine_threshold = 0.92,
            sub_clust        = SUB_CLUST,
            k_sub_max        = K_SUB_MAX,
            sub_min_voxels   = SUB_MIN_VOX,
        )
    except Exception as exc:
        import traceback
        log(f"\n[ERROR] Discovery failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── GT comparison (Discovery) ─────────────────────────────────────────────
    compare_with_ground_truth(disc_out, valid_mouse_dirs, csv_paths)

    # ─────────────────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("STEP 4 -- Joint clustering (robust_cohort vs robust_global vs robust)")
    log("=" * 60)

    # Build Joint mice_list from raw roi_data.csv files
    jt_mice = []
    for mdir in mouse_dirs:
        files = sorted(glob.glob(os.path.join(mdir, "*_roi_data.csv")))
        if files:
            jt_mice.append({"name": os.path.basename(mdir), "files": files})
    log(f"Joint mice: {len(jt_mice)}")

    jt_norms = ["robust_recal", "robust_cohort", "robust_global", "robust"]
    jt_results = {}

    for norm in jt_norms:
        log(f"\n--- Joint [{norm}] ---")
        jt_out = os.path.join(OUT_ROOT, f"joint_{norm}")
        os.makedirs(jt_out, exist_ok=True)
        try:
            res = run_joint_pipeline(
                mice_list    = jt_mice,
                method       = METHOD,
                k_range      = range(2, 7),
                n_init_gmm   = N_INIT,
                n_refs       = N_REFS_GAP,
                output_dir   = jt_out,
                log_fn       = log,
                normalization= norm,
            )
            jt_results[norm] = res
            log(f"  K={res['best_k']}  ({len(res['df_pooled'])} voxels)")
        except Exception as exc:
            import traceback
            log(f"  [ERROR] Joint [{norm}] failed: {exc}")
            traceback.print_exc()

    # ── GT comparison for each Joint normalization ────────────────────────────
    if jt_results:
        compare_joint_with_gt(jt_results, mouse_dirs)

    # ── Final summary ─────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("TEST COMPLETE")
    log("=" * 60)
    log(f"Output folder  : {disc_out}")
    log(f"Chosen K       : {results['chosen_k']}")
    log(f"Meta-habitats  : {[s['display_label'] for s in results['meta_stats']]}")
    if results.get("sub_results"):
        active = {k: v for k, v in results["sub_results"].items()
                  if not v.get("skipped")}
        log(f"Sub-clustered  : {len(active)} / {len(results['sub_results'])} MH")
        for mh_idx, res in active.items():
            log(f"  {res['meta_stat']['display_label']}: K_sub={res['sub_k']}")
    log("\nOutputs:")
    log("  multi_mouse_discovery_report.pdf")
    log("  multi_mouse_discovery_level2_report.pdf")
    log("  gt_confusion_matrix.png")
    log("  gt_per_mouse_ari.png")
    log("  gt_mh_purity.png")


if __name__ == "__main__":
    main()
