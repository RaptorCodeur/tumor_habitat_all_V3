"""Standalone Napari launcher -- run in its own process to avoid
Tkinter/Qt conflicts in the Jupyter kernel."""
import sys
import os
import json
import numpy as np
import nibabel as nib
import napari

# Import colours from config so this file stays in sync with the main app.
# Falls back to a built-in palette if config cannot be imported (e.g. when
# launched outside the project directory).
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import HABITAT_COLORS
except Exception:
    HABITAT_COLORS = [
        (255,   0,   0, 200), (  0, 200, 255, 200), (  0, 255,   0, 200),
        (255, 255,   0, 200), (255,   0, 255, 200), (255, 140,   0, 200),
        (100,   0, 255, 200), (  0, 200, 130, 200), ( 30,  80, 255, 200),
        (210,  80,   0, 200),
    ]

def main(nifti_path, mask_path, labels_path):
    nii_obj = nib.load(nifti_path)
    img_ref = np.nan_to_num(nii_obj.get_fdata(), nan=0.0)
    img_ref = np.squeeze(img_ref)
    img_display = img_ref[..., 0] if img_ref.ndim == 4 else img_ref

    # NIfTI voxels are stored as (x, y, z); Napari's slider controls axis 0.
    # Transpose to (z, y, x) so the slice axis drives the slider.
    if img_display.ndim == 3:
        img_display = img_display.transpose(2, 1, 0)

    mask_obj = nib.load(mask_path)
    habitats_mask = np.asarray(mask_obj.get_fdata()).astype(int)
    if habitats_mask.ndim == 3:
        habitats_mask = habitats_mask.transpose(2, 1, 0)

    with open(labels_path) as f:
        habitat_labels = {int(k): v for k, v in json.load(f).items()}

    viewer = napari.Viewer(title='Tumor Habitat Segmentation -- V2')
    viewer.add_image(img_display, name='Reference MRI',
                     colormap='gray', opacity=1.0)

    color_dict = {0: (0, 0, 0, 0)}
    for hab_id in habitat_labels:
        color_dict[hab_id + 1] = tuple(
            c / 255 for c in HABITAT_COLORS[hab_id % len(HABITAT_COLORS)])

    for hab_id, label_name in habitat_labels.items():
        mask = (habitats_mask == (hab_id + 1)).astype(int)
        rgba = tuple(c / 255 for c in HABITAT_COLORS[hab_id % len(HABITAT_COLORS)])
        layer = viewer.add_labels(mask, name=f'H{hab_id} -- {label_name}', opacity=0.8)
        layer.color_mode = 'direct'
        layer.color = {0: (0, 0, 0, 0), 1: rgba}
        layer.refresh()

    napari.run()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])