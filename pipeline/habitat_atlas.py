"""
pipeline/habitat_atlas.py
Weighted GMM atlas for inter-animal habitat comparison.

Core idea: each mouse contributes exactly VOXELS_PER_MOUSE voxels to the GMM
fit (with replacement if needed), so every mouse has equal influence regardless
of tumour size. The fitted GMM is then frozen; all mice receive soft
(posterior-based) assignments — no re-clustering, no sampling noise.

Note: sklearn's GaussianMixture.fit() does not support sample_weight, so we
implement equal-weight contribution via balanced subsampling instead.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture   import GaussianMixture
from sklearn.metrics   import silhouette_score
from scipy.optimize    import linear_sum_assignment
from scipy.spatial.distance import cdist

from config import RANDOM_SEED, GMM_SILHOUETTE_GUARD

VOXELS_PER_MOUSE = 500   # voxels contributed per mouse to the GMM fit


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _balanced_sample(features: np.ndarray, mouse_ids: np.ndarray,
                     seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Return a balanced feature matrix where each mouse contributes exactly
    VOXELS_PER_MOUSE rows (sampled with replacement if the mouse has fewer).
    This gives each mouse equal weight in the GMM fit without needing
    sample_weight support from sklearn.
    """
    rng    = np.random.default_rng(seed)
    unique = np.unique(mouse_ids)
    parts  = []
    for uid in unique:
        idx    = np.where(mouse_ids == uid)[0]
        chosen = rng.choice(idx, size=VOXELS_PER_MOUSE,
                            replace=len(idx) < VOXELS_PER_MOUSE)
        parts.append(features[chosen])
    return np.vstack(parts)


def _fit_gmm(features: np.ndarray, mouse_ids: np.ndarray,
             k: int, n_init: int, seed: int) -> GaussianMixture:
    bal = _balanced_sample(features, mouse_ids, seed)
    gmm = GaussianMixture(
        n_components=k,
        n_init=n_init,
        covariance_type="full",
        random_state=seed,
        max_iter=500,
    )
    gmm.fit(bal)
    return gmm


def _align_centroids(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Reorder `other` centroids to minimise total distance to `ref` (Hungarian)."""
    cost      = cdist(ref, other)
    _, col    = linear_sum_assignment(cost)
    return other[col]


# ─────────────────────────────────────────────────────────────────────────────
# K selection
# ─────────────────────────────────────────────────────────────────────────────

def _elbow(k_list: list, scores: list) -> int:
    """
    Kneedle elbow: point of maximum perpendicular distance from the line
    connecting the first and last (k, score) values.
    Works for both increasing (Silhouette) and decreasing (BIC) curves.
    """
    if len(k_list) < 3:
        return k_list[int(np.argmax(scores))]
    ka = np.array(k_list,  dtype=float)
    sa = np.array(scores,  dtype=float)
    ka = (ka - ka.min()) / (ka.max() - ka.min() + 1e-10)
    sa = (sa - sa.min()) / (sa.max() - sa.min() + 1e-10)
    d  = np.array([ka[-1] - ka[0], sa[-1] - sa[0]])
    d /= np.linalg.norm(d) + 1e-10
    dists = [abs((ka[i]-ka[0])*d[1] - (sa[i]-sa[0])*d[0]) for i in range(len(k_list))]
    return k_list[int(np.argmax(dists))]


def _icl(gmm: GaussianMixture, features: np.ndarray) -> float:
    """
    Integrated Completed Likelihood = BIC + 2 × classification entropy.
    Penalises fuzzy/overlapping clusters → selects lower K than BIC alone.
    EN = 0 when all posteriors are hard (0 or 1); large when clusters overlap.
    """
    post    = gmm.predict_proba(features)           # (n, K)
    entropy = -np.sum(post * np.log(post + 1e-300)) # classification entropy
    return gmm.bic(features) + 2.0 * entropy


def _plot_atlas_k(k_list, bic_vals, icl_vals, sil_vals,
                  k_bic, k_icl, k_sil, k_final, out_dir: str) -> None:
    """3-panel K-selection figure saved to out_dir/k_selection_atlas.png."""
    import os
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, vals, label, chosen, better in [
        (axes[0], bic_vals, "BIC  (elbow, lower=better)", k_bic,  "min"),
        (axes[1], icl_vals, "ICL  (minimum, lower=better)", k_icl, "min"),
        (axes[2], sil_vals, "Silhouette  (elbow, higher=better)", k_sil, "max"),
    ]:
        ax.plot(k_list, vals, marker='o', color='steelblue')
        ax.axvline(chosen, color='crimson', ls='--', lw=1.5, label=f'Selected K={chosen}')
        ax.axvline(k_final, color='forestgreen', ls='-', lw=2.0,
                   alpha=0.7, label=f'Final K={k_final}')
        ax.set_xlabel("K")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Atlas K selection  —  Final K={k_final}  "
                 f"(BIC-elbow={k_bic} · ICL-min={k_icl} · Sil-elbow={k_sil})",
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "k_selection_atlas.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def select_atlas_k(features: np.ndarray, mouse_ids: np.ndarray,
                   k_range: range, n_init: int = 20,
                   seed: int = RANDOM_SEED,
                   gmm_silhouette_guard: float | None = GMM_SILHOUETTE_GUARD,
                   out_dir: str | None = None,
                   log=print) -> tuple[int, dict]:
    """
    Triple-criterion K selection for the weighted GMM atlas.

    Three criteria evaluated on the full (unsubsampled) feature matrix:
      1. BIC elbow  — knee of the BIC curve (stops decreasing rapidly)
      2. ICL minimum — BIC + 2×entropy; penalises fuzzy clusters → lower K
      3. Silhouette elbow — knee of the Silhouette curve

    Decision logic:
      • Majority vote (≥ 2/3 agree) → chosen K
      • All disagree → ICL (most conservative)
      • Silhouette guard: if Sil@chosen_K < gmm_silhouette_guard → Silhouette elbow

    Returns (best_k, {k: bic}).
    """
    from collections import Counter

    k_list    = list(k_range)
    bic_vals  = []
    icl_vals  = []
    sil_vals  = []
    bic_dict  = {}

    log(f"  Atlas K selection  K={k_list}  ({VOXELS_PER_MOUSE} vox/mouse)")
    log("  Criteria: BIC-elbow · ICL-min · Silhouette-elbow")

    for k in k_list:
        gmm  = _fit_gmm(features, mouse_ids, k, n_init, seed)
        bic  = gmm.bic(features)
        icl  = _icl(gmm, features)
        labs = gmm.predict(features)
        sil  = (silhouette_score(features, labs)
                if len(np.unique(labs)) > 1 else 0.0)

        bic_dict[k] = bic
        bic_vals.append(bic)
        icl_vals.append(icl)
        sil_vals.append(sil)
        log(f"    K={k}  BIC={bic:.1f}  ICL={icl:.1f}  Sil={sil:.3f}")

    # ── Criteria ──────────────────────────────────────────────────────────────
    # BIC elbow: find the knee on the decreasing BIC curve (negate → increasing)
    k_bic = _elbow(k_list, [-v for v in bic_vals])
    k_icl = k_list[int(np.argmin(icl_vals))]
    k_sil = _elbow(k_list, sil_vals)

    log(f"  BIC-elbow={k_bic}  ICL-min={k_icl}  Sil-elbow={k_sil}")

    # ── Majority vote ─────────────────────────────────────────────────────────
    votes = Counter([k_bic, k_icl, k_sil])
    top_k, top_n = votes.most_common(1)[0]
    if top_n >= 2:
        best_k = top_k
        log(f"  Majority vote → K={best_k}  ({top_n}/3 criteria agree)")
    else:
        best_k = k_icl   # all disagree: ICL is most conservative
        log(f"  All criteria disagree → ICL (conservative) K={best_k}")

    # ── Silhouette guard ──────────────────────────────────────────────────────
    sil_at_best = sil_vals[k_list.index(best_k)]
    if gmm_silhouette_guard is not None and sil_at_best < gmm_silhouette_guard:
        log(f"  [!] Sil guard: Sil@K={best_k}={sil_at_best:.3f} < {gmm_silhouette_guard}"
            f" → override with Sil-elbow K={k_sil}")
        best_k = k_sil

    log(f"  ✓ Final K = {best_k}")

    if out_dir:
        try:
            _plot_atlas_k(k_list, bic_vals, icl_vals, sil_vals,
                          k_bic, k_icl, k_sil, best_k, out_dir)
        except Exception:
            pass

    return best_k, bic_dict


# ─────────────────────────────────────────────────────────────────────────────
# WeightedAtlas
# ─────────────────────────────────────────────────────────────────────────────

class WeightedAtlas:
    """
    Frozen GMM atlas trained with equal per-mouse weighting.

    Workflow
    --------
    1. atlas = WeightedAtlas(k).fit(features, mouse_ids, parameters)
    2. desc  = atlas.describe_mouse(features_one_mouse)
       → {'pi', 'sigma2', 'residual', 'n_voxels'}
    3. atlas.bootstrap_stability(features, mouse_ids)  # optional quality check
    """

    def __init__(self, k: int, n_init: int = 20, seed: int = RANDOM_SEED):
        self.k          = k
        self.n_init     = n_init
        self.seed       = seed
        self.gmm_       : GaussianMixture | None = None
        self.centroids_ : np.ndarray | None = None   # (K, n_params)
        self.parameters : list[str]         = []
        self.stability_ : dict              = {}

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray, mouse_ids: np.ndarray,
            parameters: list[str]) -> "WeightedAtlas":
        """
        Train the weighted GMM.  After this call the atlas is frozen.

        Parameters
        ----------
        features   : (n_voxels, n_params)  normalised feature matrix
        mouse_ids  : (n_voxels,)           integer mouse index per voxel
        parameters : list of MRI parameter names
        """
        self.parameters = list(parameters)
        self.gmm_       = _fit_gmm(features, mouse_ids, self.k, self.n_init, self.seed)
        self.centroids_ = self.gmm_.means_.copy()   # (K, n_params)
        return self

    # ── Per-mouse descriptors ─────────────────────────────────────────────────

    def posteriors(self, features: np.ndarray) -> np.ndarray:
        """Soft assignment: P(habitat_k | voxel). Shape (n_voxels, K)."""
        assert self.gmm_ is not None, "Call fit() first."
        return self.gmm_.predict_proba(features)

    def composition(self, features: np.ndarray) -> np.ndarray:
        """
        Soft compositional vector π ∈ Δ^(K-1).
        π_k = mean posterior across all voxels.  Always > 0 (GMM posteriors
        are never exactly 0), so CLR transform is well-defined.
        Returns (K,).
        """
        return self.posteriors(features).mean(axis=0)

    def intra_variance(self, features: np.ndarray) -> np.ndarray:
        """
        Posterior-weighted intra-habitat feature variance per habitat.
        σ²_h = Σ_i w_{ih} ‖x_i − μ_h‖² / Σ_i w_{ih}  (averaged over dims).
        Returns (K,).  High value → voxels in h are spread, not tight.
        """
        post  = self.posteriors(features)    # (n, K)
        var_h = np.zeros(self.k)
        for h in range(self.k):
            w     = post[:, h]
            w_sum = w.sum()
            if w_sum < 1e-10:
                var_h[h] = np.nan
                continue
            w_n    = w / w_sum
            mean_h = (w_n[:, None] * features).sum(axis=0)
            diff   = features - mean_h
            var_h[h] = (w_n[:, None] * diff ** 2).sum(axis=0).mean()
        return var_h

    def atlas_residual(self, features: np.ndarray) -> np.ndarray:
        """
        Posterior-weighted mean Euclidean distance to each centroid.
        r_h = Σ_i w_{ih} * ||x_i − μ_h|| / Σ_i w_{ih}
        Consistent with intra_variance (both use soft posteriors).
        r_h = 0 means perfect atlas fit; NaN only if w_sum < 1e-10.
        Returns (K,).
        """
        post  = self.posteriors(features)    # (n, K)
        res_h = np.zeros(self.k)
        for h in range(self.k):
            w     = post[:, h]
            w_sum = w.sum()
            if w_sum < 1e-10:
                res_h[h] = np.nan
                continue
            diffs    = features - self.centroids_[h]          # (n, d)
            dists    = np.sqrt((diffs ** 2).sum(axis=1))      # (n,)
            res_h[h] = (w * dists).sum() / w_sum
        return res_h

    def describe_mouse(self, features: np.ndarray) -> dict:
        """
        Full descriptor for one mouse.

        Returns
        -------
        dict with keys:
          'pi'       : (K,)  soft composition  ∈ (0,1), sums to 1
          'sigma2'   : (K,)  intra-habitat variance (mean over dims)
          'residual' : (K,)  mean distance to assigned centroid
          'n_voxels' : int
        """
        return {
            "pi"      : self.composition(features),
            "sigma2"  : self.intra_variance(features),
            "residual": self.atlas_residual(features),
            "n_voxels": len(features),
        }

    # ── Stability ─────────────────────────────────────────────────────────────

    def bootstrap_stability(self, features: np.ndarray,
                             mouse_ids: np.ndarray,
                             n_bootstrap: int = 30,
                             log=print) -> dict:
        """
        Assess centroid stability via non-parametric bootstrap over mice.

        Each iteration resamples mice (with replacement), refits a weighted GMM,
        and aligns centroids to the frozen atlas via Hungarian matching.
        Records the L2 shift per centroid across all bootstrap runs.

        Returns
        -------
        dict:
          'mean_shift'  : (K,)  mean L2 shift per habitat
          'max_shift'   : (K,)  max  L2 shift per habitat
          'all_shifts'  : (n_bootstrap, K)  full distribution
          'stable'      : bool  True if mean_shift < 0.1 for all habitats
        """
        assert self.gmm_ is not None, "Call fit() first."
        unique_mice = np.unique(mouse_ids)
        shifts      = []

        log(f"Bootstrap stability ({n_bootstrap} runs) …")
        for b in range(n_bootstrap):
            rng        = np.random.default_rng(self.seed + b + 1)
            boot_mice  = rng.choice(unique_mice, size=len(unique_mice), replace=True)

            # Build bootstrap feature matrix with new sequential mouse ids
            boot_feat, boot_ids = [], []
            for new_id, orig_id in enumerate(boot_mice):
                mask = mouse_ids == orig_id
                boot_feat.append(features[mask])
                boot_ids.append(np.full(int(mask.sum()), new_id))

            boot_feat = np.vstack(boot_feat)
            boot_ids  = np.concatenate(boot_ids)

            gmm_b     = _fit_gmm(boot_feat, boot_ids, self.k,
                                  n_init=5,
                                  seed=int(rng.integers(0, 2 ** 31)))

            aligned   = _align_centroids(self.centroids_, gmm_b.means_)
            shift_b   = np.sqrt(((self.centroids_ - aligned) ** 2).sum(axis=1))
            shifts.append(shift_b)

        shifts = np.array(shifts)   # (n_bootstrap, K)
        self.stability_ = {
            "mean_shift": shifts.mean(axis=0),
            "max_shift" : shifts.max(axis=0),
            "all_shifts": shifts,
            "stable"    : bool((shifts.mean(axis=0) < 0.10).all()),
        }
        log("  Centroid stability (mean L2 shift per habitat):")
        for h, ms in enumerate(self.stability_["mean_shift"]):
            flag = "" if ms < 0.10 else "  [!] UNSTABLE"
            log(f"    H{h}: {ms:.4f}{flag}")
        if self.stability_["stable"]:
            log("  → Atlas is stable (all shifts < 0.10)", "ok")
        else:
            log("  → Some centroids are unstable — interpret with caution", "err")
        return self.stability_
