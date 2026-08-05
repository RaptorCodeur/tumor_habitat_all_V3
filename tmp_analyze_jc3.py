import json, os, numpy as np, pandas as pd

base = r'C:\Users\laboratorio\Desktop\Projet 1\dossier_test\test_08\test_0408'
norms = ['recal','cohort','robust']

for norm in norms:
    d = os.path.join(base, f'joint_clustering_3_{norm}')
    c = json.load(open(os.path.join(d, 'centroids.json')))
    k = len(c)
    params = [p for p in list(c['0'].keys()) if p != 'd_topo_norm']
    print(f'=== {norm}  K={k} ===')
    for h in sorted(c.keys(), key=int):
        vals = c[h]
        mtr = vals.get('MTR',0); ad = vals.get('DTI-AD',0)
        rd = vals.get('DTI-RD',0); fa = vals.get('DTI-FA',0); t2 = vals.get('T2',0)
        if mtr < -0.8 and ad > 2:
            bio = 'NECROSIS'
        elif ad < -0.8 and rd < -0.5:
            bio = 'Dense/viable'
        elif fa > 1.0 and rd < 0:
            bio = 'Fibrose/anisotrope'
        elif t2 > 0.4 and mtr > 0.3:
            bio = 'Viable proliferatif'
        elif rd > 0.4 and mtr < 0:
            bio = 'Hypoxique/oedeme'
        elif abs(mtr) < 0.2 and abs(ad) < 0.3 and abs(rd) < 0.3:
            bio = 'Moyen/neutre'
        else:
            bio = 'Mixte'
        print('  H%s: MTR=%+.2f AD=%+.2f FA=%+.2f RD=%+.2f T2=%+.2f  => %s' % (
            h, mtr, ad, fa, rd, t2, bio))

    print('  Composition par souris:')
    for mdir in sorted(os.listdir(d)):
        csv = os.path.join(d, mdir, 'habitats_result.csv')
        if not os.path.isfile(csv):
            continue
        df = pd.read_csv(csv)
        if 'Habitat' not in df.columns:
            continue
        counts = df['Habitat'].value_counts().sort_index()
        total = len(df)
        comp = '  '.join(f'H{h}:{100*n/total:.0f}%' for h, n in counts.items())
        print(f'    {mdir[:24]}: {comp}')

    # consistency score
    means = []
    for mdir in sorted(os.listdir(d)):
        csv = os.path.join(d, mdir, 'habitats_result.csv')
        if not os.path.isfile(csv):
            continue
        df = pd.read_csv(csv)
        p_cols = [p for p in params if p in df.columns]
        if p_cols:
            means.append(df[p_cols].mean().values)
    if means:
        mat = np.array(means)
        score = mat.std(axis=0, ddof=0).mean()
        print(f'  Consistency score: {score:.3f}')
    print()
