"""
test_size_guard.py — Exercices run_clustering_with_size_guard with synthetic data.

Scenario 1 (guard fires):
  90 Bulk + 6 Necrosis + 4 Cellular(tiny at 4%)
  → guard detects the 4-voxel Cellular cluster, Cellular ∉ _KEEP_SMALL
  → should reduce k  3 → 2

Scenario 2 (guard silent):
  90 Bulk + 6 Cellular + 4 Necrosis(tiny at 4%)
  → guard detects the 4-voxel cluster, but Necrosis ∈ _KEEP_SMALL
  → should keep k = 3

Run from tumor_habitat_V6/:
  python test_size_guard.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.clustering import run_clustering_with_size_guard

PARAMS = ["T2", "T2star", "DTI-MD", "DTI-FA", "MTR"]
RNG    = np.random.RandomState(42)


def _make_dataset(n_bulk, bulk_feat,
                  n_cluster2, feat2,
                  n_cluster3, feat3):
    """
    Build (features_scaled, df_all) in robust-scaled MRI space.
    Adds ±0.02 Gaussian noise to each centroid to make kmeans realistic.
    """
    def blk(n, centre):
        return RNG.normal(centre, 0.02, size=(n, len(PARAMS)))

    features = np.vstack([
        blk(n_bulk, bulk_feat),
        blk(n_cluster2, feat2),
        blk(n_cluster3, feat3),
    ])
    n_total = len(features)
    df_all  = pd.DataFrame({
        "X":     np.arange(n_total, dtype=float),
        "Y":     np.zeros(n_total),
        "Slice": np.zeros(n_total),
    })
    return features, df_all


def run_scenario(title, expected_k, n_bulk, n_mid, n_tiny,
                 bulk_feat, mid_feat, tiny_feat):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  Layout: {n_bulk} Bulk  |  {n_mid} mid  |  {n_tiny} tiny")
    n_total = n_bulk + n_mid + n_tiny
    print(f"  Tiny fraction: {100*n_tiny/n_total:.1f}%  (threshold 5%)")
    print(sep)

    features, df_all = _make_dataset(
        n_bulk, bulk_feat,
        n_mid,  mid_feat,
        n_tiny, tiny_feat,
    )

    labels, _extra, final_k = run_clustering_with_size_guard(
        features, features, df_all,
        best_k=3,
        method="kmeans",
        spatial_weight=1.0,
        n_init_gmm=20,
        parameters=PARAMS,
        dtopo_meta=None,
        k_min=2,
        min_ratio=0.05,
    )

    print(f"\n  Final k = {final_k}   (expected {expected_k})")
    for c in range(final_k):
        n_c = int((labels == c).sum())
        print(f"    Cluster {c}: {n_c:3d} voxels  ({100*n_c/len(labels):.1f}%)")

    ok = final_k == expected_k
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return ok


# ── Centroid positions in robust-scaled space ─────────────────────
# Bulk tumor: all features near tumour median (zero in robust-scaled)
BULK_FEAT = [0.0,  0.0,  0.0,  0.0,  0.0]

# Necrosis: high T2 (free water), high MD (free diffusion), low FA
NEC_FEAT  = [1.5,  0.0,  1.5, -0.5,  0.0]

# Cellular/proliferative: low MD (restricted diffusion), low T2, moderate MTR
CEL_FEAT  = [-0.5, -0.3, -0.5,  0.1,  0.3]

# Hypoxic/vascular: low T2* (deoxyHb), slightly above-median T2
HYP_FEAT  = [0.1, -0.5,  0.0,  0.0,  0.0]


if __name__ == "__main__":
    results = []

    # Scenario 1: tiny cluster is Cellular → not protected → guard fires → k 3→2
    results.append(run_scenario(
        title      = "SCENARIO 1 — tiny Cellular cluster (NOT protected)",
        expected_k = 2,
        n_bulk = 90, bulk_feat = BULK_FEAT,
        n_mid  =  6, mid_feat  = NEC_FEAT,
        n_tiny =  4, tiny_feat = CEL_FEAT,
    ))

    # Scenario 2: tiny cluster is Necrosis → protected → guard silent → k stays 3
    results.append(run_scenario(
        title      = "SCENARIO 2 — tiny Necrosis cluster (protected)",
        expected_k = 3,
        n_bulk = 90, bulk_feat = BULK_FEAT,
        n_mid  =  6, mid_feat  = CEL_FEAT,
        n_tiny =  4, tiny_feat = NEC_FEAT,
    ))

    # Scenario 3: no small cluster at all → guard is a no-op → k stays 3
    results.append(run_scenario(
        title      = "SCENARIO 3 — no small cluster (guard should be silent)",
        expected_k = 3,
        n_bulk = 60, bulk_feat = BULK_FEAT,
        n_mid  = 25, mid_feat  = CEL_FEAT,
        n_tiny = 15, tiny_feat = NEC_FEAT,   # 15% > 5% → not tiny
    ))

    # Scenario 4: k_min floor — even with small clusters k stops at 2
    results.append(run_scenario(
        title      = "SCENARIO 4 — guard hits k_min=2 floor",
        expected_k = 2,
        n_bulk = 90, bulk_feat = BULK_FEAT,
        n_mid  =  6, mid_feat  = HYP_FEAT,  # Hypoxic — also not protected
        n_tiny =  4, tiny_feat = CEL_FEAT,  # Cellular — also not protected
        # With k_min=2: guard fires at k=3 (tiny Cellular), reduces to k=2.
        # At k=2 both clusters are large enough → accepted.
    ))

    print("\n" + "=" * 62)
    passed = sum(results)
    print(f"  RESULTS: {passed}/{len(results)} scenarios passed")
    print("=" * 62)
    sys.exit(0 if passed == len(results) else 1)
