import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from pipeline.labeling_ws import estimate_weights_from_multi_gt

GT_JSON   = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labels_Without_hierarchie.json"
GT_FOLDER = r"C:\Users\laboratorio\Desktop\Projet 1\gt_labeling"

with open(GT_JSON, encoding="utf-8") as f:
    raw = f.read().replace('"Hypoxic/vascular,', '"Hypoxic/vascular",')
gt_all = json.loads(raw)

runs = {}
for animal_id in gt_all:
    cp = os.path.join(GT_FOLDER, animal_id, "V2", "centroids.json")
    if not os.path.exists(cp): continue
    with open(cp) as f:
        raw_c = json.load(f)
    centroids  = {int(k): v for k, v in raw_c.items()}
    parameters = [p for p in next(iter(centroids.values())).keys() if p != "d_topo_norm"]
    runs[animal_id] = (centroids, parameters)

w = estimate_weights_from_multi_gt(runs, gt_all)
print(json.dumps(w, indent=2))
