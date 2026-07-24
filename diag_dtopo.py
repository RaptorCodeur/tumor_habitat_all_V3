import os, json

GT_LABELING_DIR = r"C:\Users\laboratorio\Desktop\projet 1\gt_labeling"
GT_LABELS_JSON  = r"C:\Users\laboratorio\Desktop\projet 1\gt_labels.json"
RESULT_SUBDIR   = "V2"

with open(GT_LABELS_JSON, encoding="utf-8") as f:
    gt_json = json.load(f)

print("d_topo_norm par label GT :")
print(f"{'animal / cluster':<45} {'GT label':<28} {'d_topo'}")
print("-" * 90)

by_label = {}
for animal_id, labels in gt_json.items():
    json_path = os.path.join(GT_LABELING_DIR, animal_id, RESULT_SUBDIR, "centroids.json")
    if not os.path.exists(json_path):
        continue
    with open(json_path, encoding="utf-8") as f:
        centroids = {int(k): v for k, v in json.load(f).items()}
    for cid_str, lbl in labels.items():
        cid = int(cid_str)
        if cid not in centroids:
            continue
        dt = centroids[cid].get("d_topo_norm")
        tag = f"{animal_id} / {cid}"
        if dt is not None:
            print(f"  {tag:<43} {lbl:<28} {dt:.4f}")
            by_label.setdefault(lbl, []).append(dt)
        else:
            print(f"  {tag:<43} {lbl:<28} N/A")

print()
print("Statistiques par label :")
for lbl, vals in sorted(by_label.items()):
    if not vals:
        continue
    mn = min(vals)
    mx = max(vals)
    av = sum(vals) / len(vals)
    print(f"  {lbl:<28} n={len(vals):2d}  min={mn:.3f}  max={mx:.3f}  mean={av:.3f}")
