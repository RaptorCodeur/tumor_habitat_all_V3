import os, sys, glob, json
sys.path.insert(0, r'C:\Users\laboratorio\tumor_habitat_all_V3')
from pipeline.joint_clustering import run_joint_pipeline, export_per_mouse as _jt_export

base     = r'C:\Users\laboratorio\Desktop\Projet 1\complete_data'
out_base = r'C:\Users\laboratorio\Desktop\Projet 1\dossier_test\test_08\test_0308\test_sync\real_mice_norm_comparison'

selected = [
    'A170924_R15NM_20s_HFD',
    'F030622_R5F_10s_CD',
    'H030321_R4MM_10s_HFD',
    'A080322_R12_20s_HFD',
    'C290722_R13M_20s_HFD',
    'H150622_R6F_20s_CD',
    'A030722_R11F_20s_CD',
    'H300622_R8F_10s_CD',
    'A291221_R63F_10s_HFD',
    'A030122_R71F_10s_HFD',
    'A040122_R64F_10s_HFD',
    'G061221_R55M_10s_HFD',
]

mice_list = []
for m in selected:
    csvs = glob.glob(os.path.join(base, m, '*.csv'))
    mice_list.append({'name': m, 'files': csvs})

print('Mice:', len(mice_list))

methods = ['robust', 'robust_global', 'cl_hybrid', 'cl_global']
summary = {}

for norm in methods:
    out_dir = os.path.join(out_base, norm)
    os.makedirs(out_dir, exist_ok=True)
    print()
    print('=' * 60)
    print('NORMALIZATION:', norm)
    print('=' * 60)
    try:
        res = run_joint_pipeline(
            mice_list    = mice_list,
            method       = 'gmm',
            k_range      = (2, 8),
            n_init_gmm   = 10,
            n_refs       = 20,
            output_dir   = out_dir,
            log_fn       = print,
            normalization= norm,
        )
        k = res['best_k']
        centroids = res['centroids']
        params = res['parameters']
        _jt_export(res['df_pooled'], params, mice_list, out_dir, k,
                   normalization=norm, log_fn=print)
        summary[norm] = {'K': k, 'params': params, 'centroids': centroids}
        print('>>> DONE  K =', k)
    except Exception as e:
        import traceback
        traceback.print_exc()
        summary[norm] = {'K': None, 'error': str(e)}

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
for norm, r in summary.items():
    k = r.get('K')
    print('  {:18}  K={}'.format(norm, k if k else 'ERROR'))

# Save centroids for each method
with open(os.path.join(out_base, 'summary.json'), 'w') as f:
    # centroids keys are int — convert
    out = {}
    for norm, r in summary.items():
        if r.get('K'):
            out[norm] = {
                'K': r['K'],
                'params': r['params'],
                'centroids': {str(k): v for k, v in r['centroids'].items()},
            }
    json.dump(out, f, indent=2)
print('summary.json saved.')
