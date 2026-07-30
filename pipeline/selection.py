# =================================================================
# pipeline/selection.py — Optimal K selection
# Gap Statistic is parallelized with joblib for speed.
# =================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.cluster          import KMeans
from sklearn.metrics          import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics          import pairwise_distances
from scipy.sparse.linalg      import eigsh
from joblib                   import Parallel, delayed

from config import (
    K_RANGE, N_REFS_GAP, N_INIT_GMM, N_JOBS_GAP,
    GMM_SILHOUETTE_GUARD, KMEANS_GAP_GUARD, RANDOM_SEED,
)
from sklearn.mixture import GaussianMixture


# =================================================================
# Shared utilities
# =================================================================

def _inertia_log(X, labels, k):
    """Log of total within-cluster inertia. Used by Gap Statistic."""
    total = 0.0
    for c in range(k):
        mask = labels == c
        if mask.sum() < 2:
            continue
        cluster  = X[mask]
        centroid = cluster.mean(axis=0)
        total   += ((cluster - centroid) ** 2).sum()
    return np.log(total + 1e-10)


def _find_elbow(k_list, scores):
    """
    Kneedle elbow detection: point of maximum perpendicular distance
    from the line connecting the first and last score values.
    More robust than taking the absolute maximum.
    """
    if len(k_list) < 3:
        return k_list[np.argmax(scores)]

    k_arr  = np.array(k_list, dtype=float)
    s_arr  = np.array(scores, dtype=float)
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-10)
    s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-10)

    d = np.array([k_norm[-1] - k_norm[0], s_norm[-1] - s_norm[0]])
    d = d / (np.linalg.norm(d) + 1e-10)

    distances = []
    for i in range(len(k_list)):
        p      = np.array([k_norm[i] - k_norm[0], s_norm[i] - s_norm[0]])
        d_perp = np.abs(p[0] * d[1] - p[1] * d[0])
        distances.append(d_perp)

    return k_list[np.argmax(distances)]


def _gap_one_ref(feat_min, feat_max, shape, k, seed):
    """
    Compute inertia for one random reference dataset.
    Called in parallel by _gap_statistic().
    """
    rng = np.random.default_rng(seed)
    ref = rng.uniform(feat_min, feat_max, size=shape)
    km  = KMeans(n_clusters=k, random_state=None, n_init=5)
    km.fit(ref)
    return _inertia_log(ref, km.labels_, k)


def _gap_statistic(embedding, k_list, feat_min, feat_max,
                   n_refs=N_REFS_GAP, n_jobs=N_JOBS_GAP, truncate_to_k=False):
    """
    Compute Gap Statistic for each k in k_list.
    Reference datasets are generated in parallel (joblib).

    Returns (gap_vals, gap_stds, best_k_gap).
    """
    gap_vals = []
    gap_stds = []

    print(f"\nComputing Gap Statistic (n_refs={n_refs}, n_jobs={n_jobs})...")

    for k in k_list:
        emb = embedding[:, :k] if truncate_to_k else embedding

        km_real = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km_real.fit(emb)
        w_real = _inertia_log(emb, km_real.labels_, k)

        seeds = np.random.SeedSequence(RANDOM_SEED).spawn(n_refs)
        seeds = [int(s.generate_state(1)[0]) % (2**31) for s in seeds]

        w_refs = Parallel(n_jobs=n_jobs)(
            delayed(_gap_one_ref)(
                feat_min[:k] if truncate_to_k else feat_min,
                feat_max[:k] if truncate_to_k else feat_max,
                emb.shape, k, seed
            )
            for seed in seeds
        )

        w_refs = np.array(w_refs)
        gap    = w_refs.mean() - w_real
        sdk    = np.std(w_refs) * np.sqrt(1 + 1 / n_refs)
        gap_vals.append(gap)
        gap_stds.append(sdk)
        print(f"  k={k}  Gap={gap:.4f}  +/-{sdk:.4f}")

    # Tibshirani criterion: smallest k such that Gap(k) >= Gap(k+1) - std(k+1)
    best_k_gap = k_list[-1]
    for i in range(len(k_list) - 1):
        if gap_vals[i] >= gap_vals[i + 1] - gap_stds[i + 1]:
            best_k_gap = k_list[i]
            break

    return gap_vals, gap_stds, best_k_gap


def _plot_dual_criterion(k_list, sil_scores, gap_vals, gap_stds,
                          best_k_sil, best_k_gap, title, output_path):
    """Save a two-panel Silhouette + Gap Statistic figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(k_list, sil_scores, marker='o', color='darkorange')
    for k, s in zip(k_list, sil_scores):
        ax1.annotate(f'{s:.3f}', (k, s),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8)
    ax1.axvline(best_k_sil, color='red', linestyle='--',
                label=f'Elbow k = {best_k_sil}')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette score')
    ax1.set_title('Silhouette -- elbow criterion')
    ax1.legend()

    ax2.errorbar(k_list, gap_vals, yerr=gap_stds,
                 marker='o', color='steelblue',
                 ecolor='lightsteelblue', capsize=4,
                 label='Gap +/- std')
    ax2.axvline(best_k_gap, color='red', linestyle='--',
                label=f'Tibshirani k = {best_k_gap}')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Gap statistic')
    ax2.set_title('Gap statistic -- Tibshirani criterion')
    ax2.legend()

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =================================================================
# Spectral affinity helpers (shared by SRSC selection + clustering)
# =================================================================

def build_spectral_affinity(features_scaled, coords=None,
                              spatial_weight=0.0, sigma_feature=None,
                              sigma_spatial=None, spatial_mode='hadamard'):
    """
    Build a (zero-diagonal) affinity matrix W for spectral clustering.

    spatial_mode:
      'hadamard' (recommended): W = W_feat * (W_spat ** alpha)
          Weighted Hadamard: alpha=1 gives classic gating (W_feat * W_spat),
          alpha=0 gives pure feature clustering (W_spat**0 = 1 everywhere),
          intermediate values continuously modulate spatial regularization.
      'blend' (legacy): W = (1-alpha)*W_feat + alpha*W_spat
    """
    n = features_scaled.shape[0]

    if sigma_feature is None:
        sample = features_scaled if n <= 500 else features_scaled[
            np.random.default_rng(RANDOM_SEED).choice(n, 500, replace=False)]
        dists         = pairwise_distances(sample, metric='euclidean')
        pos_dists     = dists[dists > 0]
        sigma_feature = float(np.median(pos_dists)) if len(pos_dists) > 0 else 1.0

    gamma_feature = 1.0 / (2 * sigma_feature ** 2)
    W_feat = rbf_kernel(features_scaled, gamma=gamma_feature).astype(np.float32)
    np.fill_diagonal(W_feat, 0.0)

    if spatial_weight <= 0 or coords is None:
        return W_feat, sigma_feature, None

    if sigma_spatial is None:
        sp_dists      = pairwise_distances(coords, metric='euclidean')
        pos_sp_dists  = sp_dists[sp_dists > 0]
        sigma_spatial = float(np.median(pos_sp_dists)) if len(pos_sp_dists) > 0 else 1.0

    gamma_spatial = 1.0 / (2 * sigma_spatial ** 2)
    W_spat = rbf_kernel(coords, gamma=gamma_spatial).astype(np.float32)
    np.fill_diagonal(W_spat, 0.0)

    if spatial_mode == 'hadamard':
        W = W_feat * (W_spat ** spatial_weight)
    elif spatial_mode == 'blend':
        W = (1 - spatial_weight) * W_feat + spatial_weight * W_spat
    else:
        raise ValueError(f"Unknown spatial_mode: {spatial_mode!r}")

    return W, sigma_feature, sigma_spatial


def normalized_affinity_eigendecomp(W, k_max):
    """
    Symmetric normalization L_sym = D^-1/2 W D^-1/2, then top-k_max
    eigenvectors sorted by DESCENDING eigenvalue.
    Row-normalized for K-means stability.
    """
    d          = np.maximum(W.sum(axis=1), 1e-10)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    A_sym      = W * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    k_max_eff = min(k_max, A_sym.shape[0] - 1)
    if k_max_eff < k_max:
        print(f"  [!] n_voxels too small for k_max={k_max}, using {k_max_eff}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigenvalues, eigenvectors = eigsh(A_sym, k=k_max_eff, which='LM')

    order        = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    norms        = np.maximum(
        np.linalg.norm(eigenvectors, axis=1, keepdims=True), 1e-10)
    eigvecs_norm = eigenvectors / norms

    return eigenvalues, eigvecs_norm


# =================================================================
# GMM K selection
# =================================================================

def select_k_gmm(features_scaled, k_range=K_RANGE, n_init_gmm=N_INIT_GMM,
                  gmm_silhouette_guard=GMM_SILHOUETTE_GUARD, output_dir=None):
    """
    K selection for GMM using BIC + Silhouette.
    BIC is the primary criterion. Silhouette is shown for reference.

    Safety guard (optional): if Silhouette at BIC-selected K is below
    gmm_silhouette_guard, the Silhouette elbow is used instead.
    Set gmm_silhouette_guard=None to disable.

    Returns (best_k, gmm_cache) where gmm_cache = {k: fitted_GaussianMixture}.
    The cache avoids re-fitting the chosen model in run_clustering.
    """
    k_list     = list(k_range)
    bic_scores = []
    sil_scores = []
    gmm_cache  = {}

    print("\nFitting GMM for k selection...")
    for k in k_list:
        gmm    = GaussianMixture(n_components=k, random_state=RANDOM_SEED,
                                 n_init=n_init_gmm, covariance_type='full',
                                 max_iter=300, init_params='kmeans')
        labels = gmm.fit_predict(features_scaled)
        bic_scores.append(gmm.bic(features_scaled))
        sil_scores.append(silhouette_score(features_scaled, labels))
        gmm_cache[k] = gmm
        print(f"  k={k}  BIC={bic_scores[-1]:.1f}  Silhouette={sil_scores[-1]:.3f}")

    best_k_bic = k_list[np.argmin(bic_scores)]
    best_k_sil = _find_elbow(k_list, sil_scores)
    print(f"\nBest k (BIC)        : {best_k_bic}")
    print(f"Best k (Silhouette) : {best_k_sil}")

    if best_k_bic != best_k_sil:
        print(f"  [!] BIC/Silhouette disagreement -- BIC retained ({best_k_bic})")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_list, bic_scores, marker='o', color='steelblue')
    ax1.axvline(best_k_bic, color='red', linestyle='--',
                label=f'Best k = {best_k_bic}')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC -- lower is better')
    ax1.legend()

    ax2.plot(k_list, sil_scores, marker='o', color='darkorange')
    ax2.axvline(best_k_sil, color='red', linestyle='--',
                label=f'Best k = {best_k_sil}')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette score')
    ax2.set_title('Silhouette -- higher is better')
    ax2.legend()

    plt.suptitle('GMM -- optimal number of tumor habitats', fontsize=13)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, '2_bic_silhouette.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()

    # Safety guard
    sil_at_bic = sil_scores[k_list.index(best_k_bic)]
    if gmm_silhouette_guard is not None and sil_at_bic < gmm_silhouette_guard:
        print(f"  [!] GMM_SILHOUETTE_GUARD: Silhouette at k={best_k_bic} "
              f"is {sil_at_bic:.3f} < {gmm_silhouette_guard}")
        print(f"      Overriding BIC with Silhouette elbow k={best_k_sil}")
        return best_k_sil, gmm_cache

    return best_k_bic, gmm_cache


# =================================================================
# K-means K selection (feature space)
# =================================================================

def select_k_kmeans(features_scaled, k_range=K_RANGE, n_refs=N_REFS_GAP,
                     n_jobs=N_JOBS_GAP, kmeans_gap_guard=KMEANS_GAP_GUARD,
                     output_dir=None):
    """
    K selection for K-means in feature space.
    Dual criterion: Silhouette elbow + Gap Statistic (parallelized).

    Safety guard (optional): if kmeans_gap_guard=True and criteria disagree,
    Silhouette elbow overrides Gap Statistic.
    """
    k_list   = list(k_range)
    feat_min = features_scaled.min(axis=0)
    feat_max = features_scaled.max(axis=0)

    sil_scores = []
    print("\nComputing Silhouette scores (K-means, feature space)...")
    for k in k_list:
        km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(features_scaled)
        sil_scores.append(silhouette_score(features_scaled, labels))
        print(f"  k={k}  Silhouette={sil_scores[-1]:.3f}")

    best_k_sil = _find_elbow(k_list, sil_scores)

    gap_vals, gap_stds, best_k_gap = _gap_statistic(
        features_scaled, k_list, feat_min, feat_max,
        n_refs=n_refs, n_jobs=n_jobs)

    print(f"\nBest k (Silhouette elbow) : {best_k_sil}")
    print(f"Best k (Gap Tibshirani)   : {best_k_gap}")

    if best_k_sil == best_k_gap:
        best_k = best_k_sil
        print(f"-> Both criteria agree: k={best_k} selected")
    elif kmeans_gap_guard:
        best_k = best_k_sil
        print(f"-> KMEANS_GAP_GUARD active: Silhouette elbow k={best_k} used over Gap k={best_k_gap}")
    else:
        best_k = best_k_gap
        print(f"-> Disagreement -- Gap prioritized: k={best_k} selected")
        print(f"   [!] Inspect curves and use K_OVERRIDE if needed")

    if output_dir:
        _plot_dual_criterion(
            k_list, sil_scores, gap_vals, gap_stds,
            best_k_sil, best_k_gap,
            'K-means (feature space) -- optimal number of tumor habitats',
            os.path.join(output_dir, '2_kmeans_silhouette.png'))

    return best_k


# =================================================================
# SRSC K selection (spectral space)
# =================================================================

def select_k_spectral(features_scaled, df_all=None, k_range=K_RANGE,
                       n_refs=N_REFS_GAP, n_jobs=N_JOBS_GAP,
                       spatial_weight=0.0, spatial_mode='hadamard',
                       kmeans_gap_guard=KMEANS_GAP_GUARD, output_dir=None):
    """
    K selection for SRSC using a dual criterion in spectral space.
    A single spectral decomposition is reused for all K candidates.
    Gap Statistic is parallelized with joblib.

    Safety guard (optional): if kmeans_gap_guard=True and criteria disagree,
    Silhouette elbow overrides Gap Statistic.
    """
    k_list = list(k_range)
    k_max  = max(k_list)

    coords = (df_all[['X', 'Y']].values.astype(np.float32)
              if (df_all is not None and spatial_weight > 0) else None)

    print("\nComputing spectral embedding (single pass)...")
    W, sigma_feature, sigma_spatial = build_spectral_affinity(
        features_scaled, coords=coords,
        spatial_weight=spatial_weight, spatial_mode=spatial_mode)

    msg = f"  sigma_feature = {sigma_feature:.4f}"
    msg += f", sigma_spatial = {sigma_spatial:.4f}" if sigma_spatial else " (no spatial term)"
    print(msg)

    _, eigvecs_norm = normalized_affinity_eigendecomp(W, k_max)

    # Silhouette in spectral space
    sil_scores = []
    print("Computing Silhouette scores in spectral space...")
    for k in k_list:
        embedding = eigvecs_norm[:, :k]
        km        = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels    = km.fit_predict(embedding)
        sil_scores.append(silhouette_score(embedding, labels))
        print(f"  k={k}  Silhouette={sil_scores[-1]:.3f}")

    best_k_sil = _find_elbow(k_list, sil_scores)

    feat_min = eigvecs_norm.min(axis=0)
    feat_max = eigvecs_norm.max(axis=0)

    gap_vals, gap_stds, best_k_gap = _gap_statistic(
        eigvecs_norm, k_list, feat_min, feat_max,
        n_refs=n_refs, n_jobs=n_jobs, truncate_to_k=True)

    print(f"\nBest k (Silhouette elbow) : {best_k_sil}")
    print(f"Best k (Gap Tibshirani)   : {best_k_gap}")

    if best_k_sil == best_k_gap:
        best_k = best_k_sil
        print(f"-> Both criteria agree: k={best_k} selected")
    elif kmeans_gap_guard:
        best_k = best_k_sil
        print(f"-> KMEANS_GAP_GUARD active: Silhouette elbow k={best_k} used over Gap k={best_k_gap}")
    else:
        best_k = best_k_gap
        print(f"-> Disagreement -- Gap prioritized: k={best_k} selected")
        print(f"   [!] Inspect curves and use K_OVERRIDE if needed")

    if output_dir:
        _plot_dual_criterion(
            k_list, sil_scores, gap_vals, gap_stds,
            best_k_sil, best_k_gap,
            'SRSC -- optimal number of tumor habitats (spectral space)',
            os.path.join(output_dir, '2_silhouette.png'))

    return best_k


# =================================================================
# Dispatcher
# =================================================================

def select_optimal_k(features_scaled, method='gmm', df_all=None,
                      k_range=K_RANGE, n_refs=N_REFS_GAP, n_jobs=N_JOBS_GAP,
                      spatial_weight=0.0, spatial_mode='hadamard',
                      n_init_gmm=N_INIT_GMM,
                      gmm_silhouette_guard=GMM_SILHOUETTE_GUARD,
                      kmeans_gap_guard=KMEANS_GAP_GUARD,
                      output_dir=None):
    """
    Route K selection to the appropriate function based on method.
    All guard parameters are forwarded from the GUI or config.

    Returns (best_k, gmm_cache) where gmm_cache = {k: fitted_GaussianMixture}
    for method='gmm' (avoids re-fitting the chosen model in run_clustering),
    and {} for other methods.
    """
    if method == 'gmm':
        return select_k_gmm(
            features_scaled, k_range=k_range, n_init_gmm=n_init_gmm,
            gmm_silhouette_guard=gmm_silhouette_guard,
            output_dir=output_dir)

    elif method == 'srsc':
        return select_k_spectral(
            features_scaled, df_all=df_all, k_range=k_range,
            n_refs=n_refs, n_jobs=n_jobs,
            spatial_weight=spatial_weight, spatial_mode=spatial_mode,
            kmeans_gap_guard=kmeans_gap_guard,
            output_dir=output_dir), {}

    elif method == 'kmeans':
        return select_k_kmeans(
            features_scaled, k_range=k_range,
            n_refs=n_refs, n_jobs=n_jobs,
            kmeans_gap_guard=kmeans_gap_guard,
            output_dir=output_dir), {}

    else:
        raise ValueError(f"Unknown method: {method!r}")
