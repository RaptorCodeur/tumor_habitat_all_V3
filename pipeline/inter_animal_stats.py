"""
pipeline/inter_animal_stats.py
Statistical tests for inter-animal habitat comparison.

Three complementary layers:
  1. Aitchison distance + PERMANOVA on compositional vectors π
     (does group composition differ?)
  2. MMD pairwise distances + PERMANOVA
     (does the full voxel distribution differ?)
  3. Conditional MMD per habitat
     (is habitat H_k the same biology in both groups?)

PERMDISP is always run alongside PERMANOVA to distinguish location shift
from dispersion difference — in oncology the latter is often the signal.
"""

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats    import chi2


# ═════════════════════════════════════════════════════════════════════════════
# 1. MMD with Gaussian kernel
# ═════════════════════════════════════════════════════════════════════════════

def _median_bandwidth(X: np.ndarray, Y: np.ndarray,
                       subsample: int = 800) -> float:
    """Median heuristic: σ = sqrt(median(‖z_i − z_j‖²) / 2)."""
    rng = np.random.default_rng(0)
    Xs  = X[rng.choice(len(X), min(subsample, len(X)), replace=False)]
    Ys  = Y[rng.choice(len(Y), min(subsample, len(Y)), replace=False)]
    Z   = np.vstack([Xs, Ys])
    sq  = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
    med = float(np.median(sq[sq > 0]))
    return float(np.sqrt(med / 2.0))


def _rbf(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    sq = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return np.exp(-sq / (2.0 * sigma ** 2))


def gaussian_mmd_sq(X: np.ndarray, Y: np.ndarray,
                     sigma: float | None = None,
                     max_voxels: int = 2000) -> tuple[float, float]:
    """
    Unbiased MMD² estimate with Gaussian kernel.

    Large feature matrices are subsampled to `max_voxels` per distribution
    to keep memory O(max_voxels²).

    Returns (mmd_sq, sigma_used).
    """
    rng = np.random.default_rng(42)
    if len(X) > max_voxels:
        X = X[rng.choice(len(X), max_voxels, replace=False)]
    if len(Y) > max_voxels:
        Y = Y[rng.choice(len(Y), max_voxels, replace=False)]

    if sigma is None:
        sigma = _median_bandwidth(X, Y)

    n, m = len(X), len(Y)
    Kxx  = _rbf(X, X, sigma); np.fill_diagonal(Kxx, 0.0)
    Kyy  = _rbf(Y, Y, sigma); np.fill_diagonal(Kyy, 0.0)
    Kxy  = _rbf(X, Y, sigma)

    mmd2 = (Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())
    return float(mmd2), sigma


def mmd_pairwise_matrix(mice_features: list[np.ndarray],
                         sigma: float | None = None,
                         max_voxels: int = 2000,
                         log=print) -> tuple[np.ndarray, float]:
    """
    Compute symmetric pairwise MMD² matrix D[i,j] = MMD²(P_i, P_j).
    Distance matrix = sqrt(max(D, 0)).

    Returns (D_sq, sigma).
    """
    n = len(mice_features)

    # Global bandwidth from pooled subsample
    if sigma is None:
        rng  = np.random.default_rng(0)
        pool = np.vstack([
            f[rng.choice(len(f), min(200, len(f)), replace=False)]
            for f in mice_features
        ])
        sq   = np.sum((pool[:, None] - pool[None, :]) ** 2, axis=-1)
        sigma = float(np.sqrt(np.median(sq[sq > 0]) / 2.0))

    log(f"  MMD bandwidth σ = {sigma:.4f}")

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mmd2, _ = gaussian_mmd_sq(mice_features[i], mice_features[j],
                                       sigma, max_voxels)
            D[i, j] = D[j, i] = max(mmd2, 0.0)
        log(f"  MMD row {i + 1}/{n} done")

    return D, sigma


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray,
                          sigma: float | None = None,
                          n_perm: int = 499,
                          max_voxels: int = 2000,
                          seed: int = 42) -> tuple[float, float, float]:
    """
    Permutation test for H0: MMD²(X, Y) = 0.
    Returns (mmd_sq_observed, p_value, sigma).
    """
    mmd_obs, sigma = gaussian_mmd_sq(X, Y, sigma, max_voxels)
    Z   = np.vstack([X, Y])
    n   = len(X)
    rng = np.random.default_rng(seed)

    null = np.zeros(n_perm)
    for t in range(n_perm):
        perm     = rng.permutation(len(Z))
        null[t], _ = gaussian_mmd_sq(Z[perm[:n]], Z[perm[n:]], sigma, max_voxels)

    p = float((null >= mmd_obs).sum() / n_perm)
    return mmd_obs, p, sigma


def mmd_per_habitat(mice_features: list[np.ndarray],
                     mice_posteriors: list[np.ndarray],
                     group_a: list[int],
                     group_b: list[int],
                     k: int,
                     sigma: float | None = None,
                     n_perm: int = 199,
                     max_voxels: int = 2000,
                     log=print) -> list[dict]:
    """
    Conditional MMD per habitat.

    For each habitat h: pool voxels hard-assigned to h across group A and
    group B separately, then test MMD(group_A_h, group_B_h).
    This checks whether habitat h has the *same biology* in both groups —
    a question the atlas approach cannot answer by construction.

    Returns list of K dicts: {habitat, mmd_sq, p_value, n_a, n_b}.
    """
    results = []
    for h in range(k):
        def _collect(indices):
            parts = []
            for i in indices:
                post = mice_posteriors[i]
                mask = post.argmax(axis=1) == h
                if mask.sum() > 0:
                    parts.append(mice_features[i][mask])
            return np.vstack(parts) if parts else np.empty((0, mice_features[0].shape[1]))

        fa, fb = _collect(group_a), _collect(group_b)
        na, nb = len(fa), len(fb)

        if na < 10 or nb < 10:
            log(f"  H{h}: too few voxels (nA={na}, nB={nb}) — skipped")
            results.append({"habitat": h, "mmd_sq": np.nan, "p_value": np.nan,
                            "n_a": na, "n_b": nb})
            continue

        mmd2, p, sig = mmd_permutation_test(fa, fb, sigma, n_perm, max_voxels)
        log(f"  H{h}: MMD²={mmd2:.4f}  p={p:.3f}  (nA={na}, nB={nb})")
        results.append({"habitat": h, "mmd_sq": mmd2, "p_value": p,
                        "n_a": na, "n_b": nb, "sigma": sig})
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. PERMANOVA + PERMDISP
# ═════════════════════════════════════════════════════════════════════════════

def _ss_total(D2: np.ndarray) -> float:
    return float(D2.sum() / (2 * len(D2)))


def _ss_within(D2: np.ndarray, groups: np.ndarray) -> float:
    ss = 0.0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n_g = len(idx)
        if n_g < 2:
            continue
        ss += D2[np.ix_(idx, idx)].sum() / (2 * n_g)
    return float(ss)


def permanova(dist_matrix: np.ndarray,
              group_labels: np.ndarray,
              n_perm: int = 999,
              seed: int = 42) -> dict:
    """
    One-way PERMANOVA (McArdle & Anderson 2001) on any symmetric distance matrix.

    Parameters
    ----------
    dist_matrix  : (n, n) distance matrix (NOT squared)
    group_labels : (n,) integer group per sample

    Returns
    -------
    dict: F, p_value, R2 (effect size), ss_between, ss_within,
          df_groups, df_residual, n_perm
    """
    D2     = dist_matrix ** 2
    n      = len(D2)
    groups = np.asarray(group_labels)
    n_g    = len(np.unique(groups))

    ss_tot = _ss_total(D2)
    ss_w   = _ss_within(D2, groups)
    ss_b   = ss_tot - ss_w
    df_b   = n_g - 1
    df_w   = n - n_g

    F_obs  = (ss_b / df_b) / (ss_w / df_w) if (df_w > 0 and ss_w > 1e-14) else np.nan
    R2     = ss_b / ss_tot if ss_tot > 1e-14 else np.nan

    rng  = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for t in range(n_perm):
        pg      = rng.permutation(groups)
        ss_w_p  = _ss_within(D2, pg)
        ss_b_p  = ss_tot - ss_w_p
        null[t] = ((ss_b_p / df_b) / (ss_w_p / df_w)
                   if ss_w_p > 1e-14 else 0.0)

    p_val = float('nan') if np.isnan(F_obs) else float((null >= F_obs).sum() / n_perm)

    return {
        "F"          : float(F_obs),
        "p_value"    : p_val,
        "R2"         : float(R2),
        "ss_between" : float(ss_b),
        "ss_within"  : float(ss_w),
        "df_groups"  : df_b,
        "df_residual": df_w,
        "n_perm"     : n_perm,
    }


def permdisp(dist_matrix: np.ndarray,
             group_labels: np.ndarray,
             n_perm: int = 999,
             seed: int = 42) -> dict:
    """
    PERMDISP — test for homogeneity of group dispersions (betadisper).

    Computes distance-to-group-centroid for each sample via PCoA (metric MDS),
    then runs a permutation F-test on those distances.

    p < 0.05 → groups differ in DISPERSION, not just location.
    This means the heterogeneity itself is the biological signal.
    """
    from sklearn.manifold import MDS

    n   = len(dist_matrix)
    mds = MDS(n_components=min(n - 1, 8), dissimilarity="precomputed",
              random_state=42, normalized_stress=False)
    coords = mds.fit_transform(dist_matrix)   # (n, ndim)
    groups = np.asarray(group_labels)

    def _dist_to_centroid(coords, groups):
        d = np.zeros(len(coords))
        for g in np.unique(groups):
            idx       = groups == g
            centroid  = coords[idx].mean(axis=0)
            d[idx]    = np.sqrt(((coords[idx] - centroid) ** 2).sum(axis=1))
        return d

    obs_d     = _dist_to_centroid(coords, groups)
    unique_g  = np.unique(groups)
    gd_obs    = [obs_d[groups == g] for g in unique_g]
    grand_m   = obs_d.mean()
    ss_b_obs  = sum(len(gd) * (gd.mean() - grand_m) ** 2 for gd in gd_obs)
    ss_w_obs  = sum(((gd - gd.mean()) ** 2).sum()       for gd in gd_obs)
    df_b, df_w = len(unique_g) - 1, n - len(unique_g)
    F_obs     = (ss_b_obs / df_b) / (ss_w_obs / df_w) if ss_w_obs > 1e-14 else np.nan

    rng  = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for t in range(n_perm):
        pg   = rng.permutation(groups)
        pd   = _dist_to_centroid(coords, pg)
        gd_p = [pd[pg == g] for g in unique_g]
        gm   = pd.mean()
        ssb  = sum(len(gd) * (gd.mean() - gm) ** 2 for gd in gd_p)
        ssw  = sum(((gd - gd.mean()) ** 2).sum()    for gd in gd_p)
        null[t] = (ssb / df_b) / (ssw / df_w) if ssw > 1e-14 else 0.0

    p_val = float('nan') if np.isnan(F_obs) else float((null >= F_obs).sum() / n_perm)

    return {
        "F"             : float(F_obs),
        "p_value"       : p_val,
        "interpretation": ("Groups differ in DISPERSION — heterogeneity IS the signal"
                           if (not np.isnan(p_val) and p_val < 0.05) else
                           "No significant difference in dispersion"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. Compositional statistics (Aitchison / CLR + Dirichlet regression)
# ═════════════════════════════════════════════════════════════════════════════

def clr_transform(pi_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Centered log-ratio transform.  Adds eps pseudo-count to handle zeros
    (soft GMM posteriors are already > 0, but σ² and residuals may be NaN).
    Returns (n, K) array.
    """
    pi     = np.where(np.isnan(pi_matrix), eps, pi_matrix) + eps
    pi     = pi / pi.sum(axis=1, keepdims=True)
    log_pi = np.log(pi)
    return log_pi - log_pi.mean(axis=1, keepdims=True)


def aitchison_dist_matrix(pi_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Pairwise Aitchison distances: D[i,j] = ‖CLR(π_i) − CLR(π_j)‖₂.
    """
    clr = clr_transform(pi_matrix, eps)
    n   = len(clr)
    D   = np.zeros((n, n))
    for i in range(n):
        diffs      = clr[i] - clr[i + 1:]
        D[i, i + 1:] = np.sqrt((diffs ** 2).sum(axis=1))
    D += D.T
    return D


# ── Dirichlet regression ─────────────────────────────────────────────────────

def _build_design_matrix(group_labels: np.ndarray,
                          covariates: np.ndarray | None = None) -> np.ndarray:
    """Intercept + group dummies (reference = last group) + optional covariates."""
    n        = len(group_labels)
    unique_g = np.unique(group_labels)
    n_g      = len(unique_g)
    G        = np.zeros((n, n_g - 1))
    for j, g in enumerate(unique_g[:-1]):
        G[:, j] = (group_labels == g).astype(float)
    parts = [np.ones((n, 1)), G]
    if covariates is not None:
        parts.append(np.atleast_2d(covariates)
                     if covariates.ndim == 1 else covariates)
    return np.hstack(parts)


def _dirichlet_nll(params: np.ndarray,
                    X: np.ndarray,
                    pi: np.ndarray,
                    k: int,
                    l2: float = 1e-3) -> float:
    """
    Negative log-likelihood of Dirichlet regression (log-link).
    Includes L2 regularisation on non-intercept weights to stabilise
    small-n fits.
    """
    n_feat = X.shape[1]
    B      = params.reshape(n_feat, k)
    log_a  = np.clip(X @ B, -10.0, 10.0)      # (n, K)
    alpha  = np.exp(log_a)                      # (n, K)  concentrations

    ll = (gammaln(alpha.sum(axis=1))
          - gammaln(alpha).sum(axis=1)
          + ((alpha - 1.0) * np.log(np.clip(pi, 1e-12, 1.0))).sum(axis=1)
          ).sum()

    # L2 penalty on non-intercept params only
    reg = l2 * (B[1:] ** 2).sum()
    return float(-ll + reg)


def _init_params(X: np.ndarray, pi: np.ndarray, k: int) -> np.ndarray:
    """
    Warm-start: set intercept so that fitted α matches group means of π.
    Non-intercept params start at 0.
    """
    n_feat   = X.shape[1]
    B0       = np.zeros((n_feat, k))
    # Intercept: log(mean(π)) — places the Dirichlet mean at the data mean
    B0[0, :] = np.log(np.clip(pi.mean(axis=0), 1e-6, 1.0))
    return B0.ravel()


def dirichlet_regression(pi_matrix: np.ndarray,
                          group_labels: np.ndarray,
                          covariates: np.ndarray | None = None,
                          l2: float = 1e-3,
                          log=print) -> dict:
    """
    Dirichlet regression: π_i ~ Dirichlet(exp(X_i β)).

    Tests whether group membership significantly explains habitat composition.
    Reports per-habitat log-ratio coefficients (group vs reference).

    Parameters
    ----------
    pi_matrix   : (n_mice, K)  soft compositional vectors from WeightedAtlas
    group_labels: (n_mice,)    integer group index (last group = reference)
    covariates  : (n_mice, p)  optional continuous covariates
    l2          : L2 regularisation strength

    Returns
    -------
    dict:
      p_value     : float  (likelihood-ratio test, full vs null model)
      lr_stat     : float
      df          : int
      group_coef  : (n_groups-1, K) log-ratio coefficients per habitat
      unique_groups: array of group ids (last = reference)
      ll_full, ll_null, fitted_alpha (n, K)
    """
    n, k = pi_matrix.shape
    pi   = np.clip(pi_matrix, 1e-10, 1.0)
    pi   = pi / pi.sum(axis=1, keepdims=True)

    groups   = np.asarray(group_labels)
    unique_g = np.unique(groups)

    X_full = _build_design_matrix(groups, covariates)
    X_null = X_full[:, :1]                        # intercept only

    kw = dict(method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})

    log("  Fitting null Dirichlet model …")
    p0_null = _init_params(X_null, pi, k)
    r_null  = minimize(_dirichlet_nll, p0_null, args=(X_null, pi, k, l2), **kw)
    ll_null = float(-r_null.fun)

    log("  Fitting full Dirichlet model …")
    p0_full = _init_params(X_full, pi, k)
    r_full  = minimize(_dirichlet_nll, p0_full, args=(X_full, pi, k, l2), **kw)
    ll_full = float(-r_full.fun)

    lr_stat = 2.0 * (ll_full - ll_null)
    df_lr   = (X_full.shape[1] - 1) * k
    p_val   = float(1.0 - chi2.cdf(max(lr_stat, 0.0), df_lr))

    B_hat       = r_full.x.reshape(X_full.shape[1], k)
    fitted_alpha = np.exp(np.clip(X_full @ B_hat, -10.0, 10.0))
    n_g_eff      = len(unique_g) - 1
    group_coef   = B_hat[1:1 + n_g_eff, :]    # (n_groups-1, K)

    log(f"  LR stat={lr_stat:.2f}  df={df_lr}  p={p_val:.4f}")

    return {
        "p_value"     : p_val,
        "lr_stat"     : float(lr_stat),
        "df"          : df_lr,
        "ll_full"     : ll_full,
        "ll_null"     : ll_null,
        "group_coef"  : group_coef,
        "fitted_alpha": fitted_alpha,
        "unique_groups": unique_g,
        "B_hat"       : B_hat,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. Witness function (MMD interpretability)
# ═════════════════════════════════════════════════════════════════════════════

def witness_2d(X: np.ndarray, Y: np.ndarray,
               sigma: float,
               param_i: int, param_j: int,
               parameters: list[str],
               n_grid: int = 40) -> dict:
    """
    2-D witness function f*(z) = E_X[k(z,x)] − E_Y[k(z,y)] evaluated on
    a grid over the plane (param_i, param_j).

    Positive region → P_X > P_Y  (group A enriched here in feature space).
    Negative region → P_Y > P_X  (group B enriched).

    Returns dict: gi, gj, GI, GJ, witness (2-D array), param names.
    """
    xi, xj = X[:, param_i], X[:, param_j]
    yi, yj = Y[:, param_i], Y[:, param_j]
    all_i  = np.concatenate([xi, yi])
    all_j  = np.concatenate([xj, yj])
    margin = 0.5

    gi = np.linspace(all_i.min() - margin, all_i.max() + margin, n_grid)
    gj = np.linspace(all_j.min() - margin, all_j.max() + margin, n_grid)
    GI, GJ = np.meshgrid(gi, gj)
    grid   = np.column_stack([GI.ravel(), GJ.ravel()])

    X2 = X[:, [param_i, param_j]]
    Y2 = Y[:, [param_i, param_j]]
    s2 = 2.0 * sigma ** 2

    Kzx = np.exp(-((grid[:, None] - X2[None]) ** 2).sum(-1) / s2)
    Kzy = np.exp(-((grid[:, None] - Y2[None]) ** 2).sum(-1) / s2)
    w   = Kzx.mean(axis=1) - Kzy.mean(axis=1)

    return {
        "gi"     : gi, "gj": gj,
        "GI"     : GI, "GJ": GJ,
        "witness": w.reshape(n_grid, n_grid),
        "param_i": parameters[param_i],
        "param_j": parameters[param_j],
    }


def top_witness_pairs(X: np.ndarray, Y: np.ndarray,
                       sigma: float,
                       parameters: list[str],
                       top_n: int = 3) -> list[tuple[int, int, float]]:
    """
    Score every parameter pair by max |witness| and return the top_n pairs.
    Used to automatically pick which 2-D projections are most discriminating.

    Returns list of (param_i, param_j, score) sorted by score descending.
    """
    d      = len(parameters)
    scored = []
    for i in range(d):
        for j in range(i + 1, d):
            w = witness_2d(X, Y, sigma, i, j, parameters, n_grid=20)
            scored.append((i, j, float(np.abs(w["witness"]).max())))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_n]


# ═════════════════════════════════════════════════════════════════════════════
# 5. Per-habitat test on π (Mann-Whitney + Bonferroni)
# ═════════════════════════════════════════════════════════════════════════════

def pi_per_habitat_tests(pi_matrix: np.ndarray,
                          group_labels: np.ndarray,
                          log=print) -> list[dict]:
    """
    For each habitat h, test whether the soft proportion π_h differs between
    groups using a non-parametric Mann-Whitney U (2 groups) or Kruskal-Wallis
    (>2 groups), with Bonferroni correction for K simultaneous comparisons.

    Works at the mouse level (n = number of mice), so power matches reality —
    no voxel-level inflation.

    Returns list of K dicts:
      habitat, stat, p_raw, p_adj, reject (p_adj < 0.05),
      median_per_group {group_id: median_pi}
    """
    from scipy.stats import mannwhitneyu, kruskal

    groups   = np.asarray(group_labels)
    unique_g = np.unique(groups)
    k        = pi_matrix.shape[1]

    results  = []
    raw_pvals = []

    for h in range(k):
        pi_h = pi_matrix[:, h]
        if len(unique_g) == 2:
            a    = pi_h[groups == unique_g[0]]
            b    = pi_h[groups == unique_g[1]]
            stat, p = mannwhitneyu(a, b, alternative='two-sided')
        else:
            groups_data = [pi_h[groups == g] for g in unique_g]
            stat, p = kruskal(*groups_data)

        raw_pvals.append(float(p))
        results.append({
            "habitat"         : h,
            "stat"            : float(stat),
            "p_raw"           : float(p),
            "median_per_group": {int(g): float(np.median(pi_h[groups == g]))
                                 for g in unique_g},
        })

    # Bonferroni correction
    for i, r in enumerate(results):
        p_adj      = min(raw_pvals[i] * k, 1.0)
        r["p_adj"] = p_adj
        r["reject"] = p_adj < 0.05

    log(f"  Per-habitat Mann-Whitney (Bonferroni x{k}):")
    for r in results:
        stars = ("*" if r["p_adj"] < 0.05 else
                 "~" if r["p_raw"] < 0.05 else "ns")
        log(f"    H{r['habitat']}: p_raw={r['p_raw']:.3f}  "
            f"p_adj={r['p_adj']:.3f}  {stars}")

    return results
