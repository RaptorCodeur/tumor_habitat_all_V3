"""
test_v5_perf.py — Quick correctness + smoke tests for V5 optimisations.

Tests:
  1. aitchison_dist_matrix  — vectorised vs old loop: identical results
  2. permanova              — known synthetic dataset: p-value in expected range
  3. mmd_pairwise_matrix    — parallel vs sequential: same matrix
  4. silhouette subsampling — select_atlas_k runs without error on small data
"""

import sys, time
import numpy as np

sys.path.insert(0, ".")

PYTHON = sys.executable
SEP    = "-" * 60

def ok(msg):  print(f"  [PASS] {msg}")
def fail(msg, e=None):
    print(f"  [FAIL] {msg}")
    if e: print(f"         {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. aitchison_dist_matrix — vectorised == old loop
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("TEST 1 — Aitchison distance matrix (vectorised vs reference loop)")

from pipeline.inter_animal_stats import clr_transform, aitchison_dist_matrix

rng = np.random.default_rng(0)
pi  = rng.dirichlet(np.ones(5), size=20)   # 20 mice, 5 habitats

# Reference: old loop implementation
def _aitchison_loop(pi_matrix, eps=1e-6):
    clr = clr_transform(pi_matrix, eps)
    n   = len(clr)
    D   = np.zeros((n, n))
    for i in range(n):
        diffs        = clr[i] - clr[i + 1:]
        D[i, i + 1:] = np.sqrt((diffs ** 2).sum(axis=1))
    D += D.T
    return D

D_loop = _aitchison_loop(pi)
D_vec  = aitchison_dist_matrix(pi)

if not np.allclose(D_loop, D_vec, atol=1e-10):
    fail("Vectorised != loop", f"max diff = {np.abs(D_loop - D_vec).max():.2e}")
else:
    ok("Vectorised result matches loop reference (max diff < 1e-10)")

if not np.allclose(D_vec, D_vec.T):
    fail("Matrix not symmetric")
else:
    ok("Matrix is symmetric")

if not np.allclose(np.diag(D_vec), 0):
    fail("Diagonal not zero")
else:
    ok("Diagonal is zero")

# ─────────────────────────────────────────────────────────────────────────────
# 2. permanova — synthetic groups with known structure
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("TEST 2 — PERMANOVA on synthetic data")

from pipeline.inter_animal_stats import permanova

# Two well-separated groups in compositional space (Dirichlet)
# Group A: dominated by habitat 0,  Group B: dominated by habitat 3
rng   = np.random.default_rng(1)
n_per = 15
pi_A  = rng.dirichlet([50, 1, 1, 1], size=n_per)   # concentrated on H0
pi_B  = rng.dirichlet([1, 1, 1, 50], size=n_per)   # concentrated on H3
pi    = np.vstack([pi_A, pi_B])
grp   = np.array([0]*n_per + [1]*n_per)
D_sep = aitchison_dist_matrix(pi)

t0  = time.perf_counter()
res = permanova(D_sep, grp, n_perm=499, seed=42)
dt  = time.perf_counter() - t0

if res["p_value"] > 0.10:
    fail(f"Expected p < 0.10 for well-separated groups, got p={res['p_value']:.3f}")
else:
    ok(f"Well-separated groups: F={res['F']:.2f}, p={res['p_value']:.3f}, R2={res['R2']:.3f}  ({dt:.2f}s)")

# Two similar groups (same Dirichlet) → expect large p-value
pi_C  = rng.dirichlet([5, 5, 5, 5], size=n_per)
pi_D  = rng.dirichlet([5, 5, 5, 5], size=n_per)
D_ns  = aitchison_dist_matrix(np.vstack([pi_C, pi_D]))
res2  = permanova(D_ns, grp, n_perm=499, seed=42)

if res2["p_value"] < 0.01:
    fail(f"Expected p >= 0.01 for similar groups, got p={res2['p_value']:.3f}")
else:
    ok(f"Similar groups (null): F={res2['F']:.2f}, p={res2['p_value']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. mmd_pairwise_matrix — parallel vs sequential
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("TEST 3 — MMD pairwise matrix (parallel vs sequential reference)")

from pipeline.inter_animal_stats import mmd_pairwise_matrix, gaussian_mmd_sq

# Sequential reference
def _mmd_sequential(mice_features, sigma, max_voxels=500):
    n = len(mice_features)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mmd2, _ = gaussian_mmd_sq(mice_features[i], mice_features[j],
                                      sigma, max_voxels)
            D[i, j] = D[j, i] = max(float(mmd2), 0.0)
    return D

rng  = np.random.default_rng(2)
mfs  = [rng.normal(size=(80, 6)) for _ in range(10)]   # 10 mice, 80 voxels, 6 features
sig  = 1.0

t0      = time.perf_counter()
D_seq   = _mmd_sequential(mfs, sig)
dt_seq  = time.perf_counter() - t0

t0      = time.perf_counter()
D_par, _ = mmd_pairwise_matrix(mfs, sigma=sig, max_voxels=500, log=lambda *a: None)
dt_par  = time.perf_counter() - t0

if not np.allclose(D_seq, D_par, atol=1e-8):
    fail("Parallel != sequential", f"max diff = {np.abs(D_seq - D_par).max():.2e}")
else:
    ok(f"Parallel result matches sequential (max diff < 1e-8)")
    ok(f"Sequential: {dt_seq:.3f}s  |  Parallel: {dt_par:.3f}s")

if not np.allclose(D_par, D_par.T):
    fail("MMD matrix not symmetric")
else:
    ok("MMD matrix is symmetric")

# ─────────────────────────────────────────────────────────────────────────────
# 4. select_atlas_k with silhouette subsampling — runs without error
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("TEST 4 — select_atlas_k with silhouette subsampling")

from pipeline.habitat_atlas import select_atlas_k, _SIL_SUBSAMPLE

# Synthetic dataset: 8 mice × 300 voxels = 2400 voxels, 4 features
# Features drawn from 3 Gaussians → K=3 expected
rng      = np.random.default_rng(3)
n_mice   = 8
n_vox    = 300
centers  = np.array([[0,0,0,0], [3,3,0,0], [0,0,3,3]], dtype=float)
feats    = []
mouse_ids = []
for m in range(n_mice):
    c = rng.choice(3, size=n_vox)
    f = centers[c] + rng.normal(scale=0.5, size=(n_vox, 4))
    feats.append(f)
    mouse_ids.extend([m] * n_vox)

features   = np.vstack(feats).astype(np.float32)
mouse_ids  = np.array(mouse_ids)

print(f"  Dataset: {features.shape[0]} voxels, {features.shape[1]} features")
print(f"  _SIL_SUBSAMPLE = {_SIL_SUBSAMPLE}")

t0 = time.perf_counter()
try:
    best_k, bic_dict = select_atlas_k(
        features, mouse_ids,
        k_range=range(2, 6),
        n_init=3,
        seed=42,
        log=lambda msg, *a: print(f"  {msg}"),
    )
    dt = time.perf_counter() - t0
    ok(f"select_atlas_k completed: K={best_k}  ({dt:.2f}s)")
    if best_k not in (2, 3, 4):
        print(f"  [WARN] K={best_k} unexpected for 3-cluster data (may vary)")
    else:
        ok(f"K={best_k} is plausible for 3-cluster synthetic data")
except Exception as e:
    fail("select_atlas_k raised an exception", e)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("All tests passed.")
