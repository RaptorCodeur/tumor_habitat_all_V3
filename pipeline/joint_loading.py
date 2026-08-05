# =================================================================
# pipeline/joint_loading.py — Multi-mouse loading with normalization choice
# =================================================================

import os
import numpy as np
import pandas as pd

from pipeline.loading import load_data, resolve_dti_parameters

_PARAM_MAP = {
    'DTI-AD': 'DTI-AD', 'DTI-FA': 'DTI-FA', 'DTI-MD': 'DTI-MD',
    'DTI-RD': 'DTI-RD', 'T2star': 'T2star', 'MTR': 'MTR', 'T2': 'T2',
}


# ------------------------------------------------------------------
# CL normalization
# ------------------------------------------------------------------

def cl_stats_per_mouse(files_list, parameters):
    """
    Compute CL-based normalization statistics (median + IQR) per parameter
    from Roi_Name == 'CL' rows.
    Returns {param: {'median': float, 'iqr': float, 'n': int}}.
    """
    raw = {p: [] for p in parameters}
    for file_path in files_list:
        raw_name = os.path.splitext(os.path.basename(file_path))[0]
        col_name = next((p for p in _PARAM_MAP if p in raw_name), None)
        if col_name not in parameters:
            continue
        df      = pd.read_csv(file_path)
        cl_rows = df[df['Roi_Name'] == 'CL']
        if len(cl_rows) == 0:
            continue
        raw[col_name].extend(cl_rows['Value'].dropna().tolist())

    stats = {}
    print(f"\n-- CL normalization statistics --")
    print(f"{'Parameter':<12} {'CL median':>12} {'CL IQR':>10} {'n':>6}")
    print("-" * 44)
    for p in parameters:
        vals = np.array(raw[p])
        if len(vals) == 0:
            stats[p] = {'median': 0.0, 'iqr': 1.0, 'n': 0}
            print(f"{p:<12} {'(no CL data — defaulting to 0/1)':>38}")
            continue
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr      = max(q75 - q25, 1e-8)
        stats[p] = {'median': float(np.median(vals)), 'iqr': iqr, 'n': int(len(vals))}
        print(f"{p:<12} {stats[p]['median']:>12.4f} {iqr:>10.4f} {len(vals):>6}")
    return stats


def cl_normalize(df_tumor, parameters, cl_stats):
    """
    (value - CL_median) / CL_IQR
    0 = contralateral level, +1 = one CL-IQR above healthy tissue.
    """
    df_scaled = df_tumor.copy()
    for p in parameters:
        s = cl_stats.get(p, {'median': 0.0, 'iqr': 1.0})
        df_scaled[p] = (df_tumor[p] - s['median']) / s['iqr']
    features = np.nan_to_num(df_scaled[parameters].values, nan=0.0)
    return df_scaled, features


def cl_hybrid_normalize(df_tumor, parameters, cl_stats):
    """
    CL-hybrid: center by CL_median (per-mouse), scale by tumor_IQR (per-mouse).

    normalized = (value - CL_median) / tumor_IQR

    0 = contralateral tissue level (biological zero preserved).
    Unit = within-tumor IQR — avoids CL IQR amplification while keeping
    parameters on comparable scales across mice and across parameters.
    """
    df_scaled = df_tumor.copy()
    print(f"\n-- CL-hybrid normalization  (CL centering + tumor-IQR scaling) --")
    print(f"{'Parameter':<12} {'CL median':>12} {'Tumor IQR':>12} {'n CL':>6}")
    print("-" * 46)
    for p in parameters:
        s         = cl_stats.get(p, {'median': 0.0, 'iqr': 1.0})
        cl_median = s['median']
        col       = df_tumor[p].dropna()
        q25, q75  = float(col.quantile(0.25)), float(col.quantile(0.75))
        tumor_iqr = max(q75 - q25, 1e-8)
        df_scaled[p] = (df_tumor[p] - cl_median) / tumor_iqr
        print(f"{p:<12} {cl_median:>12.4f} {tumor_iqr:>12.4f} {s.get('n', 0):>6}")
    features = np.nan_to_num(df_scaled[parameters].values, nan=0.0)
    return df_scaled, features


def cl_center_only(df_tumor, parameters, cl_stats):
    """
    First pass of cl_global: subtract CL_median but do NOT scale yet.
    The global IQR is applied after all mice are pooled.
    """
    df_centered = df_tumor.copy()
    for p in parameters:
        s = cl_stats.get(p, {'median': 0.0, 'iqr': 1.0})
        df_centered[p] = df_tumor[p] - s['median']
    return df_centered


# ------------------------------------------------------------------
# Per-mouse robust normalization
# ------------------------------------------------------------------

def robust_scale_per_mouse(df_tumor, parameters):
    """
    (value - tumor_median) / tumor_IQR  — computed per mouse on its own tumor.
    0 = that mouse's tumor median. Units are not directly comparable
    across mice with different tumor composition, but the shape of each
    mouse's feature distribution is preserved for joint clustering.
    """
    df_scaled = df_tumor.copy()
    print(f"\n-- Per-mouse robust scaling --")
    print(f"{'Parameter':<12} {'Tumor median':>14} {'Tumor IQR':>12}")
    print("-" * 42)
    for p in parameters:
        col    = df_tumor[p].dropna()
        median = float(col.median())
        q75, q25 = float(col.quantile(0.75)), float(col.quantile(0.25))
        iqr    = max(q75 - q25, 1e-8)
        df_scaled[p] = (df_tumor[p] - median) / iqr
        print(f"{p:<12} {median:>14.4f} {iqr:>12.4f}")
    features = np.nan_to_num(df_scaled[parameters].values, nan=0.0)
    return df_scaled, features


# ------------------------------------------------------------------
# Recalibrated normalization
# ------------------------------------------------------------------

def robust_recal_scale(df_tumor, parameters, grand_median: dict):
    """
    Recalibrated normalization: (x - grand_median[p]) / mouse_IQR[p]

    grand_median[p]  : cohort grand-median for parameter p
                       (median of per-mouse tumor medians, equal weight per mouse)
    mouse_IQR[p]     : intra-tumor IQR of THIS mouse

    Properties:
    • Shared origin  — all mice use the same reference (grand_median),
                       so a globally necrotic tumor stays negative relative
                       to a globally healthy one.
    • Per-mouse scale — IQR is mouse-specific, preserving intra-tumor structure
                        regardless of tumor size.
    • No CL required — purely data-driven, works without contralateral ROI.
    """
    df_scaled = df_tumor.copy()
    for p in parameters:
        col    = df_tumor[p].dropna()
        q75, q25 = float(col.quantile(0.75)), float(col.quantile(0.25))
        iqr    = max(q75 - q25, 1e-8)
        gmed   = grand_median.get(p, float(col.median()))
        df_scaled[p] = (df_tumor[p] - gmed) / iqr
    features = np.nan_to_num(df_scaled[parameters].values, nan=0.0)
    return df_scaled, features


# ------------------------------------------------------------------
# Joint dataset loader
# ------------------------------------------------------------------

def load_joint_dataset(mice_list, normalization='cl', log_fn=print):
    """
    Load N mice, normalize each, find common parameters, pool all voxels.

    Parameters
    ----------
    mice_list     : list of {'name': str, 'files': [path, ...]}
    normalization : 'cl'            — (val - CL_median) / CL_IQR
                                      (0 = contralateral tissue; CL IQR as unit — can
                                      amplify noise if CL region is small/homogeneous)
                   'cl_hybrid'      — (val - CL_median) / tumor_IQR  [recommended]
                                      (0 = contralateral; unit = within-tumor IQR;
                                      prevents CL-IQR amplification while preserving
                                      biological zero; requires contralateral ROI)
                   'cl_global'      — (val - CL_median_per_mouse) / global_tumor_IQR
                                      CL centering (stable biological zero) + global IQR
                                      computed once on all pooled CL-centered voxels.
                                      Decouples the biological anchor from cohort composition.
                                      Requires contralateral ROI.
                   'robust'         — (val - tumor_median) / tumor_IQR per mouse
                                      (removes inter-mouse biological differences)
                   'robust_global'  — ONE RobustScaler fitted on all pooled voxels,
                                      then applied to every mouse identically.
                                      Preserves inter-group differences while keeping
                                      features on comparable scales.
    log_fn        : callable for progress messages

    Returns
    -------
    common_params : list[str]  — parameters present in all mice
    df_pooled     : DataFrame with columns X, Y, Slice, *params, mouse_id, mouse_name
    """
    _norm_labels = {
        'cl':             "CL normalization (0 = contralateral)",
        'cl_hybrid':      "CL-hybrid: CL centering + tumor-IQR scaling (recommended with CL ROI)",
        'cl_global':      "CL-global: CL centering (per-mouse) + global-IQR scaling (pooled)",
        'robust':         "per-mouse robust scaling (removes inter-mouse differences)",
        'robust_global':  "global robust scaling (ONE scaler on all pooled voxels)",
        'robust_cohort':  "cohort-median: center by grand-median, scale by cohort IQR",
        'robust_recal':   "recalibrated: center by grand-median (median of medians), "
                          "scale by per-mouse IQR — preserves inter-mouse biology "
                          "with per-mouse intra-tumor structure",
    }
    log_fn(f"Normalization strategy: {_norm_labels.get(normalization, normalization)}")

    all_dfs          = []
    all_params       = []
    _per_mouse_meds  = []   # collected for robust_cohort / robust_recal grand-median
    _raw_dfs         = []   # raw DataFrames for robust_recal second-pass scaling

    for i, mouse in enumerate(mice_list):
        name  = mouse['name']
        files = mouse['files']
        log_fn(f"[{name}] Loading data…")

        parameters, df_tumor = load_data(files)
        parameters, _        = resolve_dti_parameters(parameters)

        log_fn(f"[{name}] {len(parameters)} params, {len(df_tumor)} voxels")

        if normalization == 'cl':
            stats        = cl_stats_per_mouse(files, parameters)
            df_scaled, _ = cl_normalize(df_tumor, parameters, stats)
        elif normalization == 'cl_hybrid':
            stats        = cl_stats_per_mouse(files, parameters)
            df_scaled, _ = cl_hybrid_normalize(df_tumor, parameters, stats)
        elif normalization == 'cl_global':
            # CL centering only — global IQR scaling applied after pooling
            stats     = cl_stats_per_mouse(files, parameters)
            df_scaled = cl_center_only(df_tumor, parameters, stats)
        elif normalization in ('robust_global', 'robust_cohort'):
            # Store raw df — global/cohort scaling applied after pooling below
            df_scaled = df_tumor.copy()
            if normalization == 'robust_cohort':
                _per_mouse_meds.append(
                    {p: float(df_tumor[p].median()) for p in parameters}
                )
        elif normalization == 'robust_recal':
            # Two-pass: collect raw data + per-mouse medians first,
            # apply recalibrated scaling after grand-median is known
            df_scaled = df_tumor.copy()
            _per_mouse_meds.append(
                {p: float(df_tumor[p].median()) for p in parameters}
            )
            _raw_dfs.append(df_tumor)
        else:
            df_scaled, _ = robust_scale_per_mouse(df_tumor, parameters)

        df_scaled['mouse_id']   = i
        df_scaled['mouse_name'] = name
        all_dfs.append(df_scaled)
        all_params.append(parameters)

    if not all_params:
        raise ValueError("load_joint_dataset: mice_list is empty — no mouse to load")

    # Intersect parameter sets, preserving order from first mouse
    common_set      = set(all_params[0])
    for p in all_params[1:]:
        common_set &= set(p)
    all_common_cols = [p for p in all_params[0] if p in common_set]
    common_params   = list(all_common_cols)

    if not common_params:
        raise ValueError("No common MRI parameters found across selected mice.")

    log_fn(f"Common parameters: {', '.join(common_params)}")

    keep      = ['X', 'Y', 'Slice'] + all_common_cols + ['mouse_id', 'mouse_name']
    df_pooled = pd.concat([df[[c for c in keep if c in df.columns]] for df in all_dfs],
                          ignore_index=True)

    # Global robust scaling: fit ONE scaler on all pooled voxels, apply to all
    if normalization == 'robust_global':
        from sklearn.preprocessing import RobustScaler
        X_all    = df_pooled[common_params].values.astype(np.float64)
        scaler   = RobustScaler().fit(X_all)
        X_scaled = scaler.transform(X_all)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        for j, p in enumerate(common_params):
            df_pooled[p] = X_scaled[:, j]
        log_fn(f"Global RobustScaler fitted on {len(X_all)} voxels — "
               f"medians: { {p: round(float(scaler.center_[j]),3) for j,p in enumerate(common_params)} }")

    # Recalibrated: (x - grand_median) / mouse_IQR  — applied mouse by mouse
    if normalization == 'robust_recal':
        # Compute grand median (equal weight per mouse)
        grand_median = {}
        for p in common_params:
            pm_vals = [m.get(p, np.nan) for m in _per_mouse_meds]
            grand_median[p] = float(np.nanmedian(pm_vals))

        print(f"\n-- Recalibrated normalization (robust_recal) --")
        print(f"{'Parameter':<12} {'Grand median':>14}  "
              f"(median of {len(_per_mouse_meds)} per-mouse medians)")
        print("-" * 50)
        for p in common_params:
            pm_vals = [m.get(p, np.nan) for m in _per_mouse_meds]
            print(f"{p:<12} {grand_median[p]:>14.4f}  "
                  f"[range {np.nanmin(pm_vals):.3f} – {np.nanmax(pm_vals):.3f}]")

        # Re-scale each mouse using grand_median as center + its own IQR
        scaled_dfs = []
        for idx, raw_df in enumerate(_raw_dfs):
            name_i = all_dfs[idx]["mouse_name"].iloc[0]
            df_s, _ = robust_recal_scale(raw_df, common_params, grand_median)
            df_s["mouse_id"]   = all_dfs[idx]["mouse_id"].iloc[0]
            df_s["mouse_name"] = name_i
            scaled_dfs.append(df_s)

        keep = ["X", "Y", "Slice"] + common_params + ["mouse_id", "mouse_name"]
        df_pooled = pd.concat(
            [df[[c for c in keep if c in df.columns]] for df in scaled_dfs],
            ignore_index=True
        )
        df_pooled[common_params] = df_pooled[common_params].fillna(0.0)

    # Cohort-median: center by grand-median (equal weight per mouse), scale by cohort IQR
    if normalization == 'robust_cohort':
        grand_median = {}
        for p in common_params:
            pm_vals = [m.get(p, np.nan) for m in _per_mouse_meds]
            grand_median[p] = float(np.nanmedian(pm_vals))

        print(f"\n-- Cohort-median normalization (robust_cohort) --")
        print(f"{'Parameter':<12} {'Grand median':>14}  "
              f"(median of {len(_per_mouse_meds)} per-mouse medians)")
        print("-" * 50)
        for p in common_params:
            print(f"{p:<12} {grand_median[p]:>14.4f}")

        for p in common_params:
            df_pooled[p] = df_pooled[p] - grand_median[p]

        print(f"\n{'Parameter':<12} {'Cohort IQR':>12}  (after grand-median centering)")
        print("-" * 30)
        for p in common_params:
            vals = df_pooled[p].dropna().values.astype(np.float64)
            q25, q75   = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            cohort_iqr = max(q75 - q25, 1e-8)
            df_pooled[p] = df_pooled[p] / cohort_iqr
            print(f"{p:<12} {cohort_iqr:>12.4f}")

        df_pooled[common_params] = df_pooled[common_params].fillna(0.0)

    # CL-global: divide all CL-centered voxels by the global IQR (computed on pooled data)
    if normalization == 'cl_global':
        log_fn("\n-- CL-global: applying global IQR scaling --")
        log_fn(f"{'Parameter':<12} {'Global IQR':>12}")
        log_fn("-" * 26)
        for p in common_params:
            vals = df_pooled[p].dropna().values.astype(np.float64)
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            g_iqr    = max(q75 - q25, 1e-8)
            df_pooled[p] = df_pooled[p] / g_iqr
            log_fn(f"{p:<12} {g_iqr:>12.4f}")
        df_pooled[common_params] = df_pooled[common_params].fillna(0.0)

    log_fn(f"Pooled: {len(df_pooled)} voxels from {len(mice_list)} mice")
    return common_params, df_pooled
