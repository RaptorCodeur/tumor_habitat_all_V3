"""
Quick test: regenerate joint_report.pdf from existing output files
without re-running the full pipeline.
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\laboratorio\tumor_habitat_all_V2')

OUTPUT_DIR = r'C:\Users\laboratorio\Desktop\Projet 1\dossier_test\test_2307\joint_clustering_7_only2'

MOUSE_NAMES = [
    'A030122_R71F_10s_HFD',
    'A030722_R11F_20s_CD',
]

# --- Load centroids ---
with open(os.path.join(OUTPUT_DIR, 'centroids.json')) as f:
    centroids_raw = json.load(f)
centroids = {int(k): v for k, v in centroids_raw.items()}
best_k = len(centroids)
common_params = list(next(iter(centroids.values())).keys())

# --- Load + pool per-mouse CSVs ---
frames = []
mice_list = []
for i, name in enumerate(MOUSE_NAMES):
    csv_path = os.path.join(OUTPUT_DIR, name, 'habitats_result.csv')
    df = pd.read_csv(csv_path)
    df['mouse_id']   = i
    df['mouse_name'] = name
    # Rename 'Habitat' to integer if it isn't already
    df['Habitat'] = df['Habitat'].astype(int)
    frames.append(df)
    mice_list.append({'name': name})

df_pooled = pd.concat(frames, ignore_index=True)

# --- Call the updated generate function ---
from pipeline.joint_clustering import generate_joint_report_pdf

generate_joint_report_pdf(
    df_pooled    = df_pooled,
    common_params= common_params,
    mice_list    = mice_list,
    centroids    = centroids,
    best_k       = best_k,
    normalization= 'robust',
    method       = 'joint',
    output_dir   = OUTPUT_DIR,
    log_fn       = print,
)

print("\nDone — opening PDF...")
os.startfile(os.path.join(OUTPUT_DIR, 'joint_report.pdf'))
