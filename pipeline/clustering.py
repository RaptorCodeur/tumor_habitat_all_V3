# =================================================================
# pipeline/clustering.py — Clustering algorithms
# =================================================================

import numpy as np
import warnings

from sklearn.cluster          import KMeans
from sklearn.metrics          import silhouette_score
from sklearn.mixture          import GaussianMixture

from config import SPATIAL_WEIGHT, SIGMA_FEATURE, N_INIT_GMM, GMM_SILHOUETTE_GUARD, MIN_CLUSTER_SIZE_RATIO, RANDOM_SEED
from pipeline.selection import build_spectral_affinity, normalized_affinity_eigendecomp


def _print_cluster_sizes(labels, k):
    """Print voxel count and percentage for each cluster."""
    print("  Cluster sizes :")
    for c in range(k):
        n_c = (labels == c).sum()
        pct = 100 * n_c / len(labels)
        warning = "  [!] Very small cluster -- possible artifact" if pct < 5 else ""
        print(f"    Cluster {c} : {n_c:>5} voxels ({pct:.1f}%){warning}")


def _cluster_gmm(features_scaled, k, n_init_gmm=N_INIT_GMM, gmm_cache=None):
    """
    Gaussian Mixture Model clustering.
    Returns hard labels, soft probabilities, and a post-clustering
    Silhouette score to diagnose cluster quality.

    gmm_cache : optional {k: fitted_GaussianMixture} from select_k_gmm.
    When the requested k is present, the already-fitted model is reused
    and a full re-training is skipped.
    """
    if gmm_cache and k in gmm_cache:
        gmm = gmm_cache[k]
        print(f"\nRunning GMM (k={k}, reusing model from k-selection)...")
    else:
        print(f"\nRunning GMM (k={k})...")
        gmm = GaussianMixture(
            n_components=k,
            init_params='kmeans',
            random_state=RANDOM_SEED,
            n_init=n_init_gmm,
            covariance_type='full',
            max_iter=300,
        )
        gmm.fit(features_scaled)
    labels = gmm.predict(features_scaled)
    proba  = gmm.predict_proba(features_scaled)
    print(f"  Converged : {gmm.converged_}")
    _print_cluster_sizes(labels, k)

    # Post-clustering Silhouette diagnostic (GMM only)
    final_sil = silhouette_score(features_scaled, labels)
    print(f"\n  Final Silhouette score : {final_sil:.3f}")
    warn_thr = GMM_SILHOUETTE_GUARD if GMM_SILHOUETTE_GUARD is not None else 0.15
    if final_sil < warn_thr:
        print(f"  [!] Low Silhouette ({final_sil:.3f} < {warn_thr}) -- clusters may not be biologically meaningful")
        print("      Consider using K_OVERRIDE with a lower K")

    return labels, {"proba": proba, "model": gmm}


def _cluster_kmeans(features_scaled, k):
    """Standard K-means clustering in robust-scaled feature space."""
    print(f"\nRunning K-means (k={k})...")
    km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(features_scaled)
    _print_cluster_sizes(labels, k)
    return labels, {"model": km}


def _cluster_srsc(features_scaled, df_all, k,
                   sigma_feature=SIGMA_FEATURE,
                   spatial_weight=SPATIAL_WEIGHT,
                   spatial_mode='hadamard'):
    """
    Spatially Regularized Spectral Clustering (SRSC).
    With spatial_weight=0, reduces to plain spectral clustering.
    Uses the shared affinity builder from selection.py for consistency.
    """
    n = features_scaled.shape[0]
    print(f"\nRunning SRSC (k={k}, n_voxels={n}, "
          f"alpha={spatial_weight}, mode={spatial_mode})...")

    mem_mb = (n ** 2 * 4) / 1e6
    print(f"  Affinity matrix : {n}x{n} ~ {mem_mb:.0f} MB")
    if mem_mb > 500:
        print("  [!] Large ROI -- consider subsampling if memory is insufficient.")

    coords = df_all[['X', 'Y']].values.astype(np.float32) if spatial_weight > 0 else None

    W, sf, ss = build_spectral_affinity(
        features_scaled, coords=coords,
        spatial_weight=spatial_weight,
        sigma_feature=sigma_feature,
        spatial_mode=spatial_mode)

    msg = f"  sigma_feature = {sf:.4f}"
    msg += f", sigma_spatial = {ss:.4f}" if ss else " (no spatial term)"
    print(msg)

    print(f"  Computing {k} eigenvectors...")
    _, eigvecs_norm = normalized_affinity_eigendecomp(W, k)

    print("  K-means in spectral space...")
    km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(eigvecs_norm)
    _print_cluster_sizes(labels, k)

    return labels, {
        "eigenvectors" : eigvecs_norm,
        "sigma_feature": sf,
        "sigma_spatial": ss,
    }


def run_clustering(features_scaled, df_all, best_k, method='gmm',
                   spatial_weight=SPATIAL_WEIGHT, n_init_gmm=N_INIT_GMM,
                   gmm_cache=None):
    """
    Dispatcher: routes to the appropriate clustering algorithm.
    Returns (labels array, extra_info dict).
    """
    if method == 'gmm':
        return _cluster_gmm(features_scaled, best_k, n_init_gmm, gmm_cache=gmm_cache)
    elif method == 'srsc':
        return _cluster_srsc(features_scaled, df_all, best_k,
                              spatial_weight=spatial_weight)
    elif method == 'kmeans':
        return _cluster_kmeans(features_scaled, best_k)
    else:
        raise ValueError(f"Unknown method: {method!r}")


# Labels kept even when the cluster is very small (biologically meaningful
# even at low volume — do NOT merge them away).
_KEEP_SMALL = {'Necrosis', 'Infiltrative'}


def run_clustering_with_size_guard(
        features_clust, features_scaled, df_all, best_k, method,
        spatial_weight, n_init_gmm, parameters,
        dtopo_meta=None, k_min=2,
        min_ratio=MIN_CLUSTER_SIZE_RATIO,
        gmm_cache=None):
    """
    Clustering with an automatic small-cluster guard.

    After each clustering attempt, every cluster whose voxel fraction is
    below min_ratio is checked with the WS labeler:
      - Necrosis or Infiltrative → keep (biologically relevant at low volume)
      - Anything else → too small to be meaningful → reduce k by 1 and retry

    Iteration stops when all clusters pass the check, or k reaches k_min.

    Parameters
    ----------
    features_clust  : ndarray — feature matrix used for clustering
                      (may include d_topo column appended by prepare_features)
    features_scaled : ndarray — original MRI-only features (n_voxels × n_params)
                      used to compute centroids for the WS labeler
    df_all          : DataFrame with X, Y, Slice (passed to SRSC)
    best_k          : initial k (from automatic selection or override)
    method          : 'gmm' | 'srsc' | 'kmeans'
    spatial_weight  : SRSC alpha
    n_init_gmm      : GMM restarts
    parameters      : list of MRI parameter names (same order as features_scaled)
    dtopo_meta      : dict from prepare_features / hierarchical_segment —
                      used to inject d_topo_norm into centroids so the WS
                      spatial LFs can distinguish Infiltrative from Bulk tumor
    k_min           : floor for k reduction (never goes below this)
    min_ratio       : voxel-fraction threshold below which a cluster is "small"

    Returns
    -------
    labels      : int ndarray (n_voxels,)
    extra_info  : dict (same as run_clustering)
    final_k     : int — effective k after guard (≤ best_k)
    """
    from pipeline.labeling_ws import label_clusters_ws
    from pipeline.spatial    import inject_dtopo_centroids

    current_k = best_k

    while current_k >= max(k_min, 2):
        labels, extra_info = run_clustering(
            features_clust, df_all, current_k,
            method=method, spatial_weight=spatial_weight,
            n_init_gmm=n_init_gmm, gmm_cache=gmm_cache)

        n_total = len(labels)

        # Build centroids on original MRI features (not d_topo-augmented)
        centroids = {
            c: {p: float(features_scaled[labels == c, i].mean())
                for i, p in enumerate(parameters)}
            for c in range(current_k)
        }
        inject_dtopo_centroids(centroids, labels, dtopo_meta)
        ws = label_clusters_ws(centroids, parameters)

        # Find clusters that are too small AND not biologically protected
        bad = []
        for c in range(current_k):
            n_c   = int((labels == c).sum())
            ratio = n_c / n_total
            if ratio < min_ratio:
                base_lbl = ws[c]['label'].split(' (')[0].strip()
                if base_lbl not in _KEEP_SMALL:
                    bad.append((c, n_c, ratio, ws[c]['label']))

        if not bad:
            if current_k < best_k:
                print(f"\n  [size guard] k={current_k} accepted "
                      f"(reduced from {best_k}: no irrelevant small cluster)")
            return labels, extra_info, current_k

        for c, n_c, ratio, lbl in bad:
            print(f"\n  [size guard] k={current_k}: cluster {c} "
                  f"too small ({n_c} vox, {100*ratio:.1f}% "
                  f"< {100*min_ratio:.0f}%) -- WS label={lbl!r} "
                  f"-> not biologically protected, reducing k")

        if current_k <= max(k_min, 2):
            # Already at the floor — accept rather than going below k_min
            print(f"\n  [size guard] k_min={current_k} reached "
                  f"with {len(bad)} unresolvable small cluster(s) -- accepting")
            return labels, extra_info, current_k

        current_k -= 1

    # Fallback (should not be reached with the floor check above)
    labels, extra_info = run_clustering(
        features_clust, df_all, current_k,
        method=method, spatial_weight=spatial_weight, n_init_gmm=n_init_gmm,
        gmm_cache=gmm_cache)
    print(f"\n  [size guard] k_min={current_k} reached -- keeping final clustering")
    return labels, extra_info, current_k
