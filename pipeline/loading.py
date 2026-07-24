# =================================================================
# pipeline/loading.py — Data loading and robust scaling
# =================================================================

import os
import numpy as np
import pandas as pd


def load_data(files_list):
    """
    Load and merge all CSV files by coordinates (X, Y, Slice).
    Returns tumor voxels (ALL) only.
    T2star is matched before T2 to avoid partial filename match.
    """
    param_map = {
        'DTI-AD': 'DTI-AD',
        'DTI-FA': 'DTI-FA',
        'DTI-MD': 'DTI-MD',
        'DTI-RD': 'DTI-RD',
        'T2star': 'T2star',
        'MTR':    'MTR',
        'T2':     'T2',
    }
    df_all = None

    for file_path in files_list:
        raw_name = os.path.splitext(os.path.basename(file_path))[0]
        col_name = next((p for p in param_map if p in raw_name), raw_name)
        print(f"Loading : {raw_name}  ->  column '{col_name}'")

        df_temp  = pd.read_csv(file_path)
        vals_all = df_temp[df_temp['Roi_Name'] == 'ALL'][
            ['X', 'Y', 'Slice', 'Value']].copy()
        vals_all = vals_all.rename(columns={'Value': col_name})

        if df_all is None:
            df_all = vals_all
        else:
            df_all = df_all.merge(
                vals_all[['X', 'Y', 'Slice', col_name]],
                on=['X', 'Y', 'Slice'], how='left')

    parameters = [c for c in df_all.columns if c not in ['X', 'Y', 'Slice']]
    print(f"\nTumor voxels (ALL) : {len(df_all)}")
    print(f"Detected parameters : {parameters}")

    for p in parameters:
        n_nan = int(df_all[p].isna().sum())
        if n_nan > 0:
            print(f"  [!] {p} : {n_nan} voxel(s) with NaN after merge "
                  f"({100*n_nan/len(df_all):.1f}%) — will be treated as 0.0 (tumour median)")
    return parameters, df_all


def robust_scale(df_all, parameters):
    """
    Robust intra-tumor normalization: (x - median) / IQR on ALL voxels.
    Insensitive to extreme voxels (necrosis, artifacts).
    If IQR = 0 for a parameter the column is set to 0.
    scaling_info is returned for reproducibility across animals.
    """
    df_scaled    = df_all.copy()
    scaling_info = {}

    for p in parameters:
        col      = df_all[p].dropna()
        median_p = col.median()
        q75, q25 = col.quantile(0.75), col.quantile(0.25)
        iqr_p    = q75 - q25
        scaling_info[p] = {"median": median_p, "iqr": iqr_p}

        if iqr_p == 0:
            print(f"  [!] {p} : IQR = 0 -> column set to 0 (constant distribution)")
            df_scaled[p] = 0.0
        else:
            df_scaled[p] = (df_all[p] - median_p) / iqr_p

    features_scaled = df_scaled[parameters].values
    features_scaled = np.nan_to_num(features_scaled, nan=0.0)

    print("\n-- Robust scaling (intra-tumor ALL) --")
    print(f"{'Parameter':<12} {'Median':>10} {'IQR':>10}")
    print("-" * 35)
    for p, info in scaling_info.items():
        print(f"{p:<12} {info['median']:>10.4f} {info['iqr']:>10.4f}")

    return df_scaled, features_scaled, scaling_info


def resolve_dti_parameters(parameters):
    """
    Remove DTI-MD when both DTI-AD and DTI-RD are present (MD is a linear
    combination of the two and adds no information while biasing distances).

    If DTI-AD or DTI-RD is absent, MD is kept as the only diffusivity metric.
    Returns (resolved_parameters, mode) where mode is 'AD_RD' or 'MD'.
    """
    has_ad = 'DTI-AD' in parameters
    has_rd = 'DTI-RD' in parameters
    has_md = 'DTI-MD' in parameters

    if has_ad and has_rd and has_md:
        resolved = [p for p in parameters if p != 'DTI-MD']
        print("DTI resolution : AD + RD present -> DTI-MD dropped (redundant)")
        return resolved, 'AD_RD'

    if has_ad and has_rd and not has_md:
        print("DTI resolution : AD + RD present, no MD -> using AD_RD rules")
        return parameters, 'AD_RD'

    print("DTI resolution : DTI-MD used (AD or RD absent)")
    return parameters, 'MD'


def load_cl_stats(files_list, parameters):
    """
    Extract contralateral (CL) statistics from the same CSV files used for
    the tumor, by filtering Roi_Name == 'CL'.

    Returns {param: {'mean': float, 'std': float, 'median': float, 'n': int}}.
    Parameters with no CL rows are silently omitted from the result.
    """
    param_map = {
        'DTI-AD': 'DTI-AD', 'DTI-FA': 'DTI-FA', 'DTI-MD': 'DTI-MD',
        'DTI-RD': 'DTI-RD', 'T2star': 'T2star', 'MTR': 'MTR', 'T2': 'T2',
    }
    all_vals = {p: [] for p in parameters}

    print("\n-- Extracting CL reference (Roi_Name == 'CL') --")
    for file_path in files_list:
        raw_name = os.path.splitext(os.path.basename(file_path))[0]
        col_name = next((p for p in param_map if p in raw_name), None)
        if col_name not in parameters:
            continue

        df      = pd.read_csv(file_path)
        cl_rows = df[df['Roi_Name'] == 'CL']
        if len(cl_rows) == 0:
            print(f"  [!] {col_name:<10} : no 'CL' ROI found in {os.path.basename(file_path)}")
            continue

        all_vals[col_name].extend(cl_rows['Value'].dropna().tolist())
        print(f"  {col_name:<10} : {len(cl_rows)} CL voxels")

    cl_stats = {}
    print(f"\n{'Parameter':<12} {'CL mean':>10} {'CL std':>10} {'CL median':>10} {'n':>6}")
    print("-" * 42)
    for param in parameters:
        vals = np.array(all_vals[param])
        if len(vals) == 0:
            continue
        mean   = float(np.mean(vals))
        std    = max(float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0, 1e-8)
        median = float(np.median(vals))
        cl_stats[param] = {'mean': mean, 'std': std, 'median': median,
                           'n': int(len(vals))}
        print(f"{param:<12} {mean:>10.4f} {std:>10.4f} {median:>10.4f} {len(vals):>6}")

    return cl_stats
