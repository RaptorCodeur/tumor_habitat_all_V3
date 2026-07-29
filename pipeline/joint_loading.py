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
        q75, q25  = float(col.quantile(0.75)), float(col.quantile(0.25))
        tumor_iqr = max(q75 - q25, 1e-8)
        df_scaled[p] = (df_tumor[p] - cl_median) / tumor_iqr
        print(f"{p:<12} {cl_median:>12.4f} {tumor_iqr:>12.4f} {s.get('n', 0):>6}")
    features = np.nan_to_num(df_scaled[parameters].values, nan=0.0)
    return df_scaled, features


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
        'cl':            "CL normalization (0 = contralateral)",
        'cl_hybrid':     "CL-hybrid: CL centering + tumor-IQR scaling (recommended with CL ROI)",
        'robust':        "per-mouse robust scaling (removes inter-mouse differences)",
        'robust_global': "global robust scaling (ONE scaler on all pooled voxels)",
    }
    log_fn(f"Normalization strategy: {_norm_labels.get(normalization, normalization)}")

    all_dfs    = []
    all_params = []

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
        elif normalization == 'robust_global':
            # Store raw df — global scaling applied after pooling below
            df_scaled = df_tumor.copy()
        else:
            df_scaled, _ = robust_scale_per_mouse(df_tumor, parameters)

        df_scaled['mouse_id']   = i
        df_scaled['mouse_name'] = name
        all_dfs.append(df_scaled)
        all_params.append(parameters)

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

    log_fn(f"Pooled: {len(df_pooled)} voxels from {len(mice_list)} mice")
    return common_params, df_pooled
