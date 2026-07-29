# =================================================================
# app.py — Graphical interface for Tumor Habitat Segmentation V2
#
# Thread safety rules:
#   - All tk.Variable values are read in _run() (main thread) and
#     passed as a plain dict to the secondary thread.
#   - All GUI updates from the secondary thread go through self.queue.
#   - listen_to_queue() polls the queue every 100ms on the main thread.
#   - The labeling dialog is opened by the main thread via the queue.
#   - Napari is launched after Tkinter closes (Qt cannot coexist with Tk).
# =================================================================

import os
import sys
import json
import threading
import queue
import subprocess
import traceback

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from config import (
    K_RANGE, CLUSTERING_METHOD, SPATIAL_WEIGHT, N_INIT_GMM, N_REFS_GAP,
    GMM_SILHOUETTE_GUARD, KMEANS_GAP_GUARD, N_JOBS_GAP,
    APP_THEME, HABITAT_COLORS_HEX,
    MRF_ENABLED, MRF_LAMBDA, MRF_CONFIDENCE_THR,
    HIERARCHICAL_ENABLED,
    DTOPO_JOINT_WEIGHT, DTOPO_MIN_NECROSIS_FRACTION,
)
from pipeline.spatial import (
    prepare_features, analyze_spatial_coherence,
    mrf_boundary_smooth, hierarchical_segment,
    inject_dtopo_centroids, compute_zone_dtopo, compute_necrotic_dtopo,
)
from pipeline.loading    import load_data, robust_scale, load_cl_stats, resolve_dti_parameters
from pipeline.selection  import select_optimal_k
from pipeline.clustering import run_clustering, run_clustering_with_size_guard
from pipeline.labeling   import show_habitat_map, _label_dialog
from pipeline.labeling_ws import label_clusters_ws
from pipeline.export     import export_results, generate_report_pdf, save_habitat_mask, save_run_meta, save_centroids_json
from compare_v4          import run_comparison_v4
from i18n                import t, get_lang, set_lang, display_label, get_method_info
from visualization.plots import (
    plot_correlation, plot_habitat_barchart,
    plot_heatmap_profiles, plot_radar_profiles,
    plot_gmm_probabilities,
)
from pipeline.joint_clustering import run_joint_pipeline, export_per_mouse as _jt_export_per_mouse
from visualize_joint_boundaries import main as _generate_boundary_analysis
from comparative_pipeline import run_comparative_pdf
from multi_mouse_discovery import (
    find_common_parameters, build_centroid_matrix,
    find_optimal_meta_k, run_meta_clustering,
    compute_meta_stats, generate_pdf as _disc_generate_pdf,
    fig_k_selection, fig_pca_scatter, fig_profiles_heatmap,
    fig_meta_profiles_bar, fig_crosstab,
    fig_mouse_index, fig_summary_table,
    fig_radar_per_mh, fig_radar_combined,
    fig_segmentation_maps,
    SIZE_FLOOR as _DISC_SIZE_FLOOR,
)


# Silence tk.Variable __del__ warnings (harmless on Windows + Jupyter)
def _silent_variable_del(self):
    pass
tk.Variable.__del__ = _silent_variable_del


# =================================================================
# Visual themes — select via APP_THEME in config.py
# =================================================================

_THEMES = {
    'dark': {
        "bg"              : "#1A1D23",
        "bg_card"         : "#21262F",
        "bg_input"        : "#2A3140",
        "bg_cluster"      : "#252B36",
        "accent"          : "#4A90D9",
        "accent_dark"     : "#3A78C2",
        "text"            : "#DDE4EE",
        "text_sub"        : "#7A8899",
        "text_mono"       : "#A8B8CC",
        "separator"       : "#2E3849",
        "log_bg"          : "#10131A",
        "log_fg"          : "#B8C8DC",
        "log_ok"          : "#4CAF7D",
        "log_err"         : "#E05252",
        "btn_bg"          : "#2A3140",
        "btn_active_bg"   : "#2E3849",
        "btn_disabled_bg" : "#2E3A50",
        "btn_disabled_fg" : "#4A5A70",
        "card_border"     : 0,
        "sec_btn_border"  : 0,
        "font_family"     : "Helvetica",
        "font_ui"         : ("Helvetica", 10),
        "font_bold"       : ("Helvetica", 10, "bold"),
        "font_section"    : ("Helvetica", 11, "bold"),
        "font_small"      : ("Helvetica", 9),
        "font_log"        : ("Courier", 9),
        "pad_x"           : 16,
        "pad_y"           : 8,
    },
    'light': {
        "bg"              : "#F7F8FA",
        "bg_card"         : "#FFFFFF",
        "bg_input"        : "#FFFFFF",
        "bg_cluster"      : "#F0F2F5",
        "accent"          : "#3A6EA5",
        "accent_dark"     : "#2B5180",
        "text"            : "#1C1C1E",
        "text_sub"        : "#6B7280",
        "text_mono"       : "#4B5563",
        "separator"       : "#DDE1E7",
        "log_bg"          : "#1E2228",
        "log_fg"          : "#C8D3E0",
        "log_ok"          : "#5CB85C",
        "log_err"         : "#E05252",
        "btn_bg"          : "#DDE1E7",
        "btn_active_bg"   : "#CBD2DA",
        "btn_disabled_bg" : "#9BAEC2",
        "btn_disabled_fg" : "#D0D8E4",
        "card_border"     : 1,
        "sec_btn_border"  : 1,
        "font_family"     : "Helvetica",
        "font_ui"         : ("Helvetica", 10),
        "font_bold"       : ("Helvetica", 10, "bold"),
        "font_section"    : ("Helvetica", 11, "bold"),
        "font_small"      : ("Helvetica", 9),
        "font_log"        : ("Courier", 9),
        "pad_x"           : 16,
        "pad_y"           : 8,
    },
}

THEME = _THEMES.get(APP_THEME, _THEMES['dark'])


def _apply_theme(root):
    """Configure ttk.Style globally. Call once after Tk() is created."""
    style = ttk.Style(root)
    style.theme_use("clam")

    bg      = THEME["bg"]
    bg_card = THEME["bg_card"]
    bg_in   = THEME["bg_input"]
    accent  = THEME["accent"]
    acc_dk  = THEME["accent_dark"]
    text    = THEME["text"]
    tsub    = THEME["text_sub"]
    sep     = THEME["separator"]
    ff      = THEME["font_family"]
    btn_bg  = THEME["btn_bg"]
    btn_act = THEME["btn_active_bg"]
    dis_bg  = THEME["btn_disabled_bg"]
    dis_fg  = THEME["btn_disabled_fg"]
    cborder = THEME["card_border"]
    sborder = THEME["sec_btn_border"]

    root.configure(bg=bg)

    style.configure("TFrame",         background=bg)
    style.configure("Card.TFrame",    background=bg_card, relief="flat",
                    borderwidth=cborder)
    style.configure("TLabel",         background=bg, foreground=text,
                    font=(ff, 10))
    style.configure("Sub.TLabel",     background=bg, foreground=tsub,
                    font=(ff, 9))
    style.configure("Section.TLabel", background=bg, foreground=text,
                    font=(ff, 11, "bold"))

    style.configure("TNotebook",
                    background=bg, borderwidth=0, tabmargins=[0, 5, 0, 0])
    style.configure("TNotebook.Tab",
                    background=sep, foreground=tsub,
                    font=(ff, 10), padding=[14, 6])
    style.map("TNotebook.Tab",
              background=[("selected", bg_card)],
              foreground=[("selected", accent)])

    style.configure("TButton",
                    background=btn_bg, foreground=text,
                    font=(ff, 10), padding=[8, 5],
                    relief="flat", borderwidth=0)
    style.map("TButton",
              background=[("active", btn_act), ("pressed", btn_act)],
              relief=[("pressed", "flat")])

    style.configure("Primary.TButton",
                    background=accent, foreground="#FFFFFF",
                    font=(ff, 10, "bold"), padding=[10, 8],
                    relief="flat", borderwidth=0)
    style.map("Primary.TButton",
              background=[("active", acc_dk), ("disabled", dis_bg)],
              foreground=[("disabled", dis_fg)])

    style.configure("Secondary.TButton",
                    background=bg_in, foreground=accent,
                    font=(ff, 10), padding=[8, 5],
                    relief="flat", borderwidth=sborder)
    style.map("Secondary.TButton",
              background=[("active", sep)])

    style.configure("TEntry",
                    fieldbackground=bg_in, foreground=text, font=(ff, 10),
                    bordercolor=sep, lightcolor=sep, darkcolor=sep,
                    insertcolor=text, padding=[6, 4])

    style.configure("TSpinbox",
                    fieldbackground=bg_in, foreground=text, font=(ff, 10),
                    bordercolor=sep, arrowcolor=tsub, padding=[6, 4])

    style.configure("TRadiobutton", background=bg, foreground=text, font=(ff, 10))
    style.map("TRadiobutton",
              background=[("active", bg)],
              foreground=[("active", accent)])

    style.configure("TCheckbutton", background=bg, foreground=text, font=(ff, 10))
    style.map("TCheckbutton",
              background=[("active", bg)],
              foreground=[("active", accent)])

    style.configure("TScale",
                    background=bg, troughcolor=sep, sliderlength=16)

    style.configure("TScrollbar",
                    background=sep, troughcolor=bg, borderwidth=0,
                    arrowsize=12, arrowcolor=tsub)

    style.configure("TSeparator", background=sep)


# =================================================================
# Help window for "?" buttons
# =================================================================

def _show_method_help(method_key):
    """Open a non-modal Toplevel with method description, advantages, limits."""
    info = get_method_info(method_key)
    if not info:
        return

    win = tk.Toplevel()
    win.title(info.get("title", t("help_method_info")))
    win.geometry("520x440")
    win.resizable(True, True)
    win.configure(bg=THEME["bg"])
    win.lift()
    win.attributes('-topmost', True)

    canvas    = tk.Canvas(win, bg=THEME["bg"], highlightthickness=0)
    scrollbar = ttk.Scrollbar(win, orient='vertical', command=canvas.yview)
    frame     = ttk.Frame(canvas)

    frame.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    pad = {"padx": 16, "pady": 4}

    ttk.Label(frame, text=info["title"],
              style="Section.TLabel").pack(anchor='w', padx=16, pady=(16, 8))

    for section_key, info_key in [
        ("help_what",  "description"),
        ("help_why",   "advantages"),
        ("help_watch", "limits"),
    ]:
        tk.Label(frame, text=t(section_key),
                 font=(THEME["font_family"], 10, "bold"),
                 fg=THEME["accent"],
                 bg=THEME["bg"]).pack(anchor='w', **pad)
        tk.Label(frame, text=info.get(info_key, ""),
                 wraplength=460, justify='left',
                 font=THEME["font_small"],
                 fg=THEME["text_sub"],
                 bg=THEME["bg"]).pack(anchor='w', padx=24, pady=(0, 10))

    ttk.Button(frame, text=t("close"),
               style="Secondary.TButton",
               command=win.destroy).pack(pady=12)


def _show_dtopo_help_window():
    """Standalone popup explaining d_topo weight and min-fraction parameters."""
    win = tk.Toplevel()
    win.title(t("wintitle_dtopo_help"))
    win.geometry("560x560")
    win.resizable(True, True)
    win.configure(bg=THEME["bg"])
    win.lift()
    win.attributes('-topmost', True)

    canvas    = tk.Canvas(win, bg=THEME["bg"], highlightthickness=0)
    scrollbar = ttk.Scrollbar(win, orient='vertical', command=canvas.yview)
    frame     = ttk.Frame(canvas)
    frame.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    def section(title, body):
        tk.Label(frame, text=title,
                 font=(THEME["font_family"], 10, "bold"),
                 fg=THEME["accent"], bg=THEME["bg"]
                 ).pack(anchor='w', padx=16, pady=(12, 2))
        tk.Label(frame, text=body,
                 wraplength=490, justify='left',
                 font=THEME["font_small"],
                 fg=THEME["text_sub"], bg=THEME["bg"]
                 ).pack(anchor='w', padx=24, pady=(0, 6))

    tk.Label(frame, text="d_topo spatial feature",
             font=(THEME["font_family"], 12, "bold"),
             fg=THEME["text"], bg=THEME["bg"]
             ).pack(anchor='w', padx=16, pady=(16, 4))

    section("What is it?",
            "d_topo_norm is a spatial feature computed in Pass 2 of the joint pipeline. "
            "It measures the topological distance of each voxel from the necrotic core "
            "(identified in Pass 1), then normalises that distance per mouse using the "
            "median and IQR of the distance distribution.\n\n"
            "Low d_topo_norm ≈ spatially close to necrosis.\n"
            "High d_topo_norm ≈ spatially far (tumour periphery).")

    section("Why can it cause problems?",
            "In mice with little or no necrosis, voxels are still assigned a d_topo value. "
            "Because the feature is scaled per mouse, it becomes noisy and can pull voxels "
            "into the 'necrosis' habitat based on their map position rather than their "
            "MRI tissue values — especially when d_topo weight = 1.0 (equal to MRI features).")

    section("Weight  (0 = off, 1 = equal to MRI features)",
            "Controls how much d_topo influences cluster assignment relative to MRI features.\n\n"
            "• 1.0 — d_topo has equal influence to DTI-AD, DTI-FA, MTR, etc.\n"
            "• 0.3 — d_topo acts as a tiebreaker; MRI tissue values dominate.\n"
            "• 0.0 — d_topo is completely ignored (same as not computing it).\n\n"
            "Recommended: 0.3.  Increase only if you specifically want spatial "
            "anatomy to drive habitat boundaries.")

    section("Min necrosis fraction  (%)",
            "Per-mouse safety guard. A mouse must have at least this percentage of its "
            "voxels assigned to the necrosis cluster (Pass 1) for d_topo to be activated "
            "for that mouse. If the fraction is below the threshold, d_topo is set to 0 "
            "for that mouse — voxel assignment uses MRI features only.\n\n"
            "• 5% (default) — excludes mice with very little or no necrosis.\n"
            "• 10% — stricter; only mice with clearly established necrosis use d_topo.\n"
            "• 1% — very permissive; almost every mouse will use d_topo.\n\n"
            "After running, open the 'ℹ d_topo impact' tab in the results window "
            "to see the per-mouse decision table.")

    ttk.Button(frame, text=t("close"),
               style="Secondary.TButton",
               command=win.destroy).pack(pady=14)


# =================================================================
# Atlas covariable parsing
# =================================================================

def _parse_atlas_covariates(names, rows, log=print):
    """
    Convert raw text entries (one row per mouse) into a float numpy matrix.

    Rules per column:
    - All values parseable as float  → continuous, used as-is
    - Parseable as YYYY-MM-DD dates  → converted to integer days since earliest
    - Otherwise (text categories)    → one-hot dummies (last category = reference)

    Returns None if names is empty, else ndarray of shape (n_mice, p).
    """
    if not names or not rows:
        return None

    import numpy as np
    import pandas as pd

    n = len(rows)
    columns = []   # list of (col_name, float_list)

    for j, name in enumerate(names):
        raw = [row[j] if j < len(row) else "" for row in rows]

        # Numeric?
        try:
            col = [float(v) if v != "" else float("nan") for v in raw]
            columns.append((name, col))
            log(f"  Covariable '{name}': numerique ({col})")
            continue
        except ValueError:
            pass

        # Date?
        try:
            dates = pd.to_datetime(raw, format="%Y-%m-%d", errors="raise")
            days  = (dates - dates.min()).days.values.astype(float)
            columns.append((name, days.tolist()))
            log(f"  Covariable '{name}': dates -> jours {days.tolist()}")
            continue
        except Exception:
            pass

        # Categorical dummies
        unique_vals = list(dict.fromkeys(v for v in raw if v != ""))
        if len(unique_vals) < 2:
            log(f"  Covariable '{name}': ignoree (valeur unique ou vide)", "err")
            continue
        ref = unique_vals[-1]
        for cat in unique_vals[:-1]:
            col = [1.0 if v == cat else 0.0 for v in raw]
            col_name = f"{name}[{cat}]"
            columns.append((col_name, col))
        log(f"  Covariable '{name}': categorielle — {len(unique_vals)-1} dummy(s), "
            f"reference='{ref}'")

    if not columns:
        return None

    mat = np.array([c for _, c in columns], dtype=float).T   # (n_mice, p)
    col_names = [cn for cn, _ in columns]
    log(f"  Matrice covariables: {mat.shape[0]} souris x {mat.shape[1]} colonnes "
        f"({', '.join(col_names)})")
    return mat


# =================================================================
# Main application
# =================================================================

class HabitatApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(t("app_title"))
        self.geometry("660x760")
        self.minsize(520, 580)
        self.resizable(True, True)
        _apply_theme(self)

        # --- tk variables (read ONLY in main thread via _run()) ---
        self.csv_files    = []
        self.nifti_path   = tk.StringVar(value="")
        self.method       = tk.StringVar(value=CLUSTERING_METHOD)
        self.k_min        = tk.IntVar(value=min(K_RANGE))
        self.k_max        = tk.IntVar(value=max(K_RANGE))
        self.k_override   = tk.StringVar(value="")
        self.spatial_w    = tk.DoubleVar(value=SPATIAL_WEIGHT)
        self.n_refs       = tk.IntVar(value=N_REFS_GAP)
        self.n_init_gmm   = tk.IntVar(value=N_INIT_GMM)
        self.output_name  = tk.StringVar(value="V2")
        self.output_dir   = tk.StringVar(value="")

        _gmm_default      = GMM_SILHOUETTE_GUARD is not None
        _kmeans_default   = bool(KMEANS_GAP_GUARD)
        self.gmm_guard          = tk.BooleanVar(value=_gmm_default)
        self.kmeans_guard       = tk.BooleanVar(value=_kmeans_default)
        self.mrf_enabled        = tk.BooleanVar(value=MRF_ENABLED)
        self.hierarchical_enabled = tk.BooleanVar(value=HIERARCHICAL_ENABLED)

        self.queue          = queue.Queue()
        self._napari_params = None

        # --- Comparison tab variables ---
        # Inter-animal
        self._ia_animals      = []
        self._ia_list_inner   = None
        self._ia_method       = tk.StringVar(value=CLUSTERING_METHOD)
        self._ia_k_min        = tk.IntVar(value=min(K_RANGE))
        self._ia_k_max        = tk.IntVar(value=max(K_RANGE))
        self._ia_k_override   = tk.StringVar(value="")
        self._ia_spatial_w    = tk.DoubleVar(value=SPATIAL_WEIGHT)
        self._ia_n_refs       = tk.IntVar(value=N_REFS_GAP)
        self._ia_n_init_gmm   = tk.IntVar(value=N_INIT_GMM)
        self._ia_gmm_guard          = tk.BooleanVar(value=_gmm_default)
        self._ia_kmeans_guard       = tk.BooleanVar(value=_kmeans_default)
        self._ia_mrf_enabled        = tk.BooleanVar(value=MRF_ENABLED)
        self._ia_hierarchical_enabled = tk.BooleanVar(value=HIERARCHICAL_ENABLED)
        self._ia_output_name  = tk.StringVar(value="inter_animal")
        self._ia_output_dir   = tk.StringVar(value="")
        self.btn_ia_run       = None

        # Inter-method
        self._im_csv_files    = []
        self._im_use_gmm      = tk.BooleanVar(value=True)
        self._im_use_srsc     = tk.BooleanVar(value=False)
        self._im_use_kmeans   = tk.BooleanVar(value=False)
        self._im_k_min        = tk.IntVar(value=min(K_RANGE))
        self._im_k_max        = tk.IntVar(value=max(K_RANGE))
        self._im_k_override   = tk.StringVar(value="")
        self._im_spatial_w    = tk.DoubleVar(value=SPATIAL_WEIGHT)
        self._im_n_refs       = tk.IntVar(value=N_REFS_GAP)
        self._im_n_init_gmm   = tk.IntVar(value=N_INIT_GMM)
        self._im_gmm_guard          = tk.BooleanVar(value=_gmm_default)
        self._im_kmeans_guard       = tk.BooleanVar(value=_kmeans_default)
        self._im_mrf_enabled        = tk.BooleanVar(value=MRF_ENABLED)
        self._im_hierarchical_enabled = tk.BooleanVar(value=HIERARCHICAL_ENABLED)
        self._im_output_name  = tk.StringVar(value="inter_method")
        self._im_output_dir   = tk.StringVar(value="")
        self._im_lbl_csv      = None
        self._im_lst_csv      = None
        self.btn_im_run       = None

        # Discovery tab
        self._disc_animals    = []
        self._disc_list_inner = None
        self._disc_method     = tk.StringVar(value=CLUSTERING_METHOD)
        self._disc_k_max      = tk.IntVar(value=max(K_RANGE))
        self._disc_n_init     = tk.IntVar(value=N_INIT_GMM)
        self._disc_meta_k     = tk.IntVar(value=0)
        self._disc_output_name     = tk.StringVar(value="discovery")
        self._disc_output_dir      = tk.StringVar(value="")
        self._disc_compare_enabled = tk.BooleanVar(value=False)
        self._disc_group_names     = [tk.StringVar(value="Control"),
                                       tk.StringVar(value="Treated")]
        self._disc_gnames_frame    = None
        self.btn_disc_run          = None

        # Joint clustering tab
        self._jt_animals      = []
        self._jt_list_inner   = None
        self._jt_method       = tk.StringVar(value='gmm')
        self._jt_k_min        = tk.IntVar(value=min(K_RANGE))
        self._jt_k_max        = tk.IntVar(value=max(K_RANGE))
        self._jt_k_override   = tk.StringVar(value="")
        self._jt_n_init_gmm   = tk.IntVar(value=N_INIT_GMM)
        self._jt_n_refs       = tk.IntVar(value=N_REFS_GAP)
        self._jt_gmm_guard    = tk.BooleanVar(value=_gmm_default)
        self._jt_kmeans_guard = tk.BooleanVar(value=_kmeans_default)
        self._jt_normalization         = tk.StringVar(value='robust')
        self._jt_dtopo_weight          = tk.DoubleVar(value=DTOPO_JOINT_WEIGHT)
        self._jt_dtopo_min_frac        = tk.DoubleVar(value=DTOPO_MIN_NECROSIS_FRACTION)
        self._jt_dtopo_min_frac_pct    = tk.IntVar(value=int(round(DTOPO_MIN_NECROSIS_FRACTION * 100)))
        self._jt_output_name       = tk.StringVar(value="joint_clustering")
        self._jt_output_dir        = tk.StringVar(value="")
        self._jt_compare_enabled   = tk.BooleanVar(value=False)
        self._jt_group_names       = [tk.StringVar(value="Control"),
                                       tk.StringVar(value="Treated")]
        self._jt_gnames_frame      = None
        self.btn_jt_run            = None

        # Comparative analysis tab
        self._cmp_animals      = []
        self._cmp_list_inner   = None
        self._cmp_method       = tk.StringVar(value='gmm')
        self._cmp_k_min        = tk.IntVar(value=min(K_RANGE))
        self._cmp_k_max        = tk.IntVar(value=max(K_RANGE))
        self._cmp_k_override   = tk.StringVar(value="")
        self._cmp_n_init_gmm   = tk.IntVar(value=N_INIT_GMM)
        self._cmp_n_refs       = tk.IntVar(value=N_REFS_GAP)
        self._cmp_normalization = tk.StringVar(value='robust_global')
        self._cmp_output_name  = tk.StringVar(value="comparative_analysis")
        self._cmp_output_dir   = tk.StringVar(value="")
        self.btn_cmp_run       = None

        # Atlas comparison tab
        self._atl_animals      = []
        self._atl_list_inner   = None
        self._atl_k_min        = tk.IntVar(value=2)
        self._atl_k_max        = tk.IntVar(value=7)
        self._atl_k_override   = tk.StringVar(value="")
        self._atl_n_init       = tk.IntVar(value=20)
        self._atl_n_bootstrap  = tk.IntVar(value=30)
        self._atl_n_perm       = tk.IntVar(value=999)
        self._atl_normalization = tk.StringVar(value='robust_global')
        self._atl_output_name  = tk.StringVar(value="atlas_comparison")
        self._atl_output_dir   = tk.StringVar(value="")
        self._atl_group_names  = [tk.StringVar(value="Control"),
                                   tk.StringVar(value="Treated")]
        self._atl_covar_names  = []   # list of tk.StringVar — names of covariables
        self.btn_atl_run       = None

        self._notebook      = None   # reference kept for tab-restore on rebuild
        self._active_tab    = 0

        self._build_ui()
        self.listen_to_queue()

        self.method.trace_add('write', self._on_method_change)
        self._on_method_change()

    # ------------------------------------------------------------------
    # Language toggle
    # ------------------------------------------------------------------

    def _set_lang(self, lang: str):
        if lang != get_lang():
            set_lang(lang)
            self._rebuild_ui()

    def _rebuild_ui(self):
        """Destroy and recreate the entire UI in the new language, preserving state."""
        if self._notebook is not None:
            try:
                self._active_tab = self._notebook.index('current')
            except Exception:
                pass

        for w in self.winfo_children():
            w.destroy()

        self._notebook = None
        self.title(t("app_title"))
        self._build_ui()

        # Restore selected notebook tab
        if self._notebook is not None:
            try:
                self._notebook.select(self._active_tab)
            except Exception:
                pass

        # Restore file list
        if self.csv_files and hasattr(self, 'lst_csv'):
            for f in self.csv_files:
                self.lst_csv.insert('end', os.path.basename(f))
            self.lbl_csv.config(
                text=t("n_files_selected", n=len(self.csv_files)),
                foreground=THEME["text"])

        # Restore discovery animals list
        if self._disc_animals and self._disc_list_inner:
            self._disc_refresh_list()

        # Restore joint clustering animals list
        if self._jt_animals and self._jt_list_inner:
            self._jt_refresh_list()

        # Restore comparative animals list
        if self._cmp_animals and self._cmp_list_inner:
            self._cmp_refresh_list()

        # Restore atlas animals list
        if self._atl_animals and self._atl_list_inner:
            self._atl_refresh_list()

        # Restore inter-method file list
        if self._im_csv_files and hasattr(self, '_im_lst_csv') and self._im_lst_csv:
            for f in self._im_csv_files:
                self._im_lst_csv.insert('end', os.path.basename(f))
            if self._im_lbl_csv:
                self._im_lbl_csv.config(
                    text=t("n_files_selected", n=len(self._im_csv_files)),
                    foreground=THEME["text"])

        self.method.trace_add('write', self._on_method_change)
        self._on_method_change()

    # ------------------------------------------------------------------
    # Method change handler
    # ------------------------------------------------------------------

    def _on_method_change(self, *_):
        if not hasattr(self, '_frame_srsc'):
            return

        m = self.method.get()

        self._frame_srsc.pack_forget()
        self._frame_gmm.pack_forget()
        self._frame_nrefs.pack_forget()
        self._frame_kmeans_guard.pack_forget()

        if m == 'srsc':
            self._frame_srsc.pack(fill='x', padx=14, pady=3)
        if m == 'gmm':
            self._frame_gmm.pack(fill='x', padx=14, pady=3)
        if m in ('srsc', 'kmeans'):
            self._frame_nrefs.pack(fill='x', padx=14, pady=3)
        if m in ('kmeans', 'srsc'):
            self._frame_kmeans_guard.pack(fill='x', padx=14, pady=3)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        header = tk.Frame(self, bg=THEME["bg_card"], height=48)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(header,
                 text=t("header_title"),
                 font=(THEME["font_family"], 13, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text"],
                 ).pack(side='left', padx=(18, 4))
        tk.Label(header,
                 text="V2",
                 font=(THEME["font_family"], 11),
                 bg=THEME["bg_card"], fg=THEME["accent"],
                 ).pack(side='left')

        # Language selector buttons
        lang_frame = tk.Frame(header, bg=THEME["bg_card"])
        lang_frame.pack(side='right', padx=12)
        current = get_lang()
        for code, label in [("en", "EN"), ("fr", "FR"), ("es", "ES")]:
            active = (code == current)
            tk.Button(lang_frame,
                      text=label,
                      font=(THEME["font_family"], 9, "bold"),
                      bg=THEME["accent"] if active else THEME["bg_card"],
                      fg="#FFFFFF" if active else THEME["accent"],
                      activebackground=THEME.get("accent", "#4A90D9"),
                      activeforeground="#FFFFFF",
                      relief="flat", bd=0, padx=8, pady=4,
                      cursor="hand2",
                      command=lambda c=code: self._set_lang(c),
                      ).pack(side='left', padx=1)

        # Bottom section packed FIRST so it stays visible when window is small
        bottom = ttk.Frame(self)
        bottom.pack(side='bottom', fill='x', padx=14, pady=(4, 10))

        # Dynamic action area — shows context-aware button(s) for the active tab
        self._action_frame = ttk.Frame(bottom)
        self._action_frame.pack(fill='x', pady=(0, 4))

        # Tabs 0-2 (Files / Method / Output): main analysis
        self._action_main = ttk.Frame(self._action_frame)
        self.btn_run = ttk.Button(self._action_main, text=t("run_analysis"),
                                   style="Primary.TButton",
                                   command=self._run)
        self.btn_run.pack(fill='x', pady=(0, 3))
        self.btn_napari = ttk.Button(self._action_main, text=t("open_napari"),
                                      style="Secondary.TButton",
                                      command=self._open_napari,
                                      state='disabled')
        self.btn_napari.pack(fill='x')

        # Tab 3 (Comparison): one button per sub-tab
        self._action_ia = ttk.Frame(self._action_frame)
        self.btn_ia_run = ttk.Button(self._action_ia, text=t("run_ia"),
                                      style="Primary.TButton",
                                      command=self._run_inter_animal)
        self.btn_ia_run.pack(fill='x')

        self._action_im = ttk.Frame(self._action_frame)
        self.btn_im_run = ttk.Button(self._action_im, text=t("run_im"),
                                      style="Primary.TButton",
                                      command=self._run_inter_method)
        self.btn_im_run.pack(fill='x')

        # Tab 4 (Discovery)
        self._action_disc = ttk.Frame(self._action_frame)
        self.btn_disc_run = ttk.Button(self._action_disc, text=t("run_discovery"),
                                        style="Primary.TButton",
                                        command=self._run_discovery)
        self.btn_disc_run.pack(fill='x')

        # Tab 5 (Joint)
        self._action_joint = ttk.Frame(self._action_frame)
        self.btn_jt_run = ttk.Button(self._action_joint, text=t("run_joint"),
                                      style="Primary.TButton",
                                      command=self._run_joint)
        self.btn_jt_run.pack(fill='x')

        # Tab 6 (Comparative)
        self._action_cmp = ttk.Frame(self._action_frame)
        self.btn_cmp_run = ttk.Button(self._action_cmp, text=t("run_comparative"),
                                       style="Primary.TButton",
                                       command=self._run_comparative)
        self.btn_cmp_run.pack(fill='x')

        # Tab 7 (Atlas)
        self._action_atlas = ttk.Frame(self._action_frame)
        self.btn_atl_run = ttk.Button(self._action_atlas, text=t("run_atlas"),
                                       style="Primary.TButton",
                                       command=self._run_atlas)
        self.btn_atl_run.pack(fill='x')

        # Show the main-analysis buttons by default; tab change swaps frames
        self._action_main.pack(fill='x')

        tk.Label(bottom, text=t("progress"),
                 font=(THEME["font_family"], 9),
                 bg=THEME["bg"], fg=THEME["text_sub"],
                 ).pack(anchor='w', pady=(0, 2))

        self.log = scrolledtext.ScrolledText(
            bottom, height=8,
            font=THEME["font_log"],
            state='disabled',
            bg=THEME["log_bg"],
            fg=THEME["log_fg"],
            insertbackground=THEME["log_fg"],
            selectbackground=THEME["accent"],
            relief="flat", borderwidth=0,
            padx=10, pady=6)
        self.log.pack(fill='both', expand=True)

        self.log.tag_config("ok",  foreground=THEME["log_ok"])
        self.log.tag_config("err", foreground=THEME["log_err"])

        # Notebook fills the middle (packed after bottom)
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=14, pady=(8, 4))
        self._notebook = notebook

        self.tab1 = ttk.Frame(notebook)
        self.tab4 = ttk.Frame(notebook)
        self.tab5 = ttk.Frame(notebook)
        self.tab6 = ttk.Frame(notebook)
        self.tab7 = ttk.Frame(notebook)
        self.tab8 = ttk.Frame(notebook)

        notebook.add(self.tab1, text=t("tab_clustering"))
        notebook.add(self.tab4, text=t("tab_comparison"))
        notebook.add(self.tab5, text=t("tab_discovery"))
        notebook.add(self.tab6, text=t("tab_joint"))
        notebook.add(self.tab7, text=t("tab_compare"))
        notebook.add(self.tab8, text=t("tab_atlas"))

        self._build_tab_clustering(self.tab1)
        self._build_tab_comparison(self.tab4)
        self._build_tab_discovery(self.tab5)
        self._build_tab_joint(self.tab6)
        self._build_tab_comparative(self.tab7)
        self._build_tab_atlas(self.tab8)

        # Swap action buttons when the user switches tabs
        _action_map = {
            0: self._action_main,
            2: self._action_disc,
            3: self._action_joint,
            4: self._action_cmp,
            5: self._action_atlas,
        }

        _all_action_frames = (self._action_main, self._action_ia, self._action_im,
                              self._action_disc, self._action_joint,
                              self._action_cmp, self._action_atlas)

        def _active_comp_frame():
            try:
                return (self._action_ia
                        if self._comp_sub.index('current') == 0
                        else self._action_im)
            except Exception:
                return self._action_ia

        def _on_tab_change(event=None):
            try:
                idx = notebook.index('current')
            except Exception:
                idx = 0
            for f in _all_action_frames:
                f.pack_forget()
            if idx == 1:
                _active_comp_frame().pack(fill='x')
            else:
                _action_map.get(idx, self._action_main).pack(fill='x')

        notebook.bind('<<NotebookTabChanged>>', _on_tab_change)
        self._comp_sub.bind('<<NotebookTabChanged>>',
                            lambda e: _on_tab_change()
                            if notebook.index('current') == 1 else None)
        self._setup_global_scroll()

    # ------------------------------------------------------------------
    # Global scroll helper
    # ------------------------------------------------------------------

    def _setup_global_scroll(self):
        """
        One bind_all that catches <MouseWheel> and arrow keys from ANY widget.
        Walks up the widget tree to find the nearest scrollable tk.Canvas and
        scrolls it — so the wheel works even when the cursor is over a Button,
        Label, Entry, etc. inside the canvas.
        """
        def _find_canvas(widget):
            w = widget
            while w is not None:
                if isinstance(w, tk.Canvas):
                    try:
                        if w.cget('yscrollcommand'):
                            return w
                    except tk.TclError:
                        pass
                try:
                    w = w.master
                except Exception:
                    break
            return None

        def _on_wheel(event):
            canvas = _find_canvas(event.widget)
            if canvas:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_arrow(event):
            wclass = event.widget.winfo_class()
            if wclass in ('Entry', 'TEntry', 'Spinbox', 'TSpinbox',
                          'Text', 'TCombobox', 'ScrolledText'):
                return
            canvas = _find_canvas(event.widget)
            if canvas:
                delta = -1 if event.keysym in ('Up', 'Prior') else 1
                units = "pages" if event.keysym in ('Prior', 'Next') else "units"
                canvas.yview_scroll(delta, units)

        self.bind_all("<MouseWheel>", _on_wheel)
        self.bind_all("<Up>",   _on_arrow)
        self.bind_all("<Down>", _on_arrow)
        self.bind_all("<Prior>", _on_arrow)
        self.bind_all("<Next>",  _on_arrow)

    # ------------------------------------------------------------------
    # Clustering tab  (Files + Method + Output réunis)
    # ------------------------------------------------------------------

    def _build_tab_clustering(self, parent):
        px  = THEME["pad_x"]
        pad = {"padx": px, "pady": THEME["pad_y"]}

        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        # ── Section : Fichiers ────────────────────────────────────────
        ttk.Label(inner, text=t("csv_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f1 = ttk.Frame(inner)
        f1.pack(fill='x', padx=px)
        self.lbl_csv = ttk.Label(f1, text=t("no_file"), style="Sub.TLabel")
        self.lbl_csv.pack(side='left', fill='x', expand=True)
        ttk.Button(f1, text="📁", style="Secondary.TButton", width=3,
                   command=self._select_csv_folder).pack(side='right', padx=(0, 2))
        ttk.Button(f1, text=t("browse"), style="Secondary.TButton",
                   command=self._select_csv).pack(side='right')

        self.lst_csv = tk.Listbox(
            inner, height=5,
            font=THEME["font_log"],
            bg=THEME["bg_input"], fg=THEME["text"],
            selectbackground=THEME["accent"], selectforeground="#FFFFFF",
            relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=THEME["separator"],
            activestyle="none")
        self.lst_csv.pack(fill='x', padx=px, pady=(4, 6))

        ttk.Label(inner, text=t("nifti_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        f2 = ttk.Frame(inner)
        f2.pack(fill='x', padx=px)
        ttk.Entry(f2, textvariable=self.nifti_path,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f2, text=t("browse"), style="Secondary.TButton",
                   command=self._select_nifti).pack(side='right', padx=(6, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=10)

        # ── Section : Méthode ─────────────────────────────────────────
        ttk.Label(inner, text=t("method_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        for val, desc_key in [
            ("gmm",    "gmm_desc"),
            ("srsc",   "srsc_desc"),
            ("kmeans", "kmeans_desc"),
        ]:
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=2)
            ttk.Radiobutton(f, text=t(desc_key),
                            variable=self.method, value=val).pack(side='left', anchor='w')
            ttk.Button(f, text="?", width=2, style="Secondary.TButton",
                       command=lambda v=val: _show_method_help(v)
                       ).pack(side='left', padx=(8, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)
        ttk.Label(inner, text=t("common_params"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)

        def row(label, widget_fn):
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=4)
            ttk.Label(f, text=label, width=30).pack(side='left')
            widget_fn(f)

        row(t("k_min"),
            lambda f: ttk.Spinbox(f, from_=2, to=10,
                                   textvariable=self.k_min, width=6).pack(side='left'))
        row(t("k_max"),
            lambda f: ttk.Spinbox(f, from_=3, to=15,
                                   textvariable=self.k_max, width=6).pack(side='left'))
        row(t("k_override"),
            lambda f: ttk.Entry(f, textvariable=self.k_override, width=6).pack(side='left'))

        self._frame_srsc = ttk.Frame(inner)
        ttk.Label(self._frame_srsc, text=t("srsc_params"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)
        sf = ttk.Frame(self._frame_srsc)
        sf.pack(fill='x', padx=px, pady=4)
        ttk.Label(sf, text=t("spatial_weight"), width=30).pack(side='left')
        self.lbl_sw = ttk.Label(sf, text=f"{SPATIAL_WEIGHT:.2f}", width=5)
        ttk.Scale(sf, from_=0.0, to=1.0, variable=self.spatial_w,
                  orient='horizontal', length=160,
                  command=lambda v: self.lbl_sw.config(
                      text=f"{float(v):.2f}")).pack(side='left')
        self.lbl_sw.pack(side='left')
        f_sig = ttk.Frame(self._frame_srsc)
        f_sig.pack(fill='x', padx=px, pady=2)
        ttk.Label(f_sig, text=t("sigma_feature"), width=30).pack(side='left')
        self.lbl_sigma = ttk.Label(f_sig, text="—", style="Sub.TLabel")
        self.lbl_sigma.pack(side='left')

        self._frame_gmm = ttk.Frame(inner)
        ttk.Label(self._frame_gmm, text=t("gmm_params"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)
        f_init = ttk.Frame(self._frame_gmm)
        f_init.pack(fill='x', padx=px, pady=4)
        ttk.Label(f_init, text=t("n_init_gmm"), width=30).pack(side='left')
        ttk.Spinbox(f_init, from_=1, to=50, textvariable=self.n_init_gmm,
                    width=6).pack(side='left')
        f_guard_gmm = ttk.Frame(self._frame_gmm)
        f_guard_gmm.pack(fill='x', padx=px, pady=2)
        ttk.Checkbutton(f_guard_gmm,
                        text=t("gmm_guard", thr=GMM_SILHOUETTE_GUARD),
                        variable=self.gmm_guard).pack(anchor='w')

        self._frame_nrefs = ttk.Frame(inner)
        f_nr = ttk.Frame(self._frame_nrefs)
        f_nr.pack(fill='x', padx=px)
        ttk.Label(f_nr, text=t("n_refs"), width=30).pack(side='left')
        ttk.Spinbox(f_nr, from_=5, to=200, textvariable=self.n_refs,
                    width=6).pack(side='left')

        self._frame_kmeans_guard = ttk.Frame(inner)
        ttk.Checkbutton(self._frame_kmeans_guard,
                        text=t("kmeans_guard"),
                        variable=self.kmeans_guard).pack(anchor='w', padx=px + 4)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)
        ttk.Label(inner, text=t("spatial_options"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(0, 4))
        ttk.Checkbutton(inner, text=t("hierarchical_opt"),
                        variable=self.hierarchical_enabled,
                        ).pack(anchor='w', padx=px + 4, pady=2)
        ttk.Checkbutton(inner,
                        text=t("mrf_opt") + f"  (λ={MRF_LAMBDA}, thr={MRF_CONFIDENCE_THR})",
                        variable=self.mrf_enabled,
                        ).pack(anchor='w', padx=px + 4, pady=2)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=10)

        # ── Section : Output ──────────────────────────────────────────
        ttk.Label(inner, text=t("output_folder"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Entry(inner, textvariable=self.output_name,
                  width=20).pack(anchor='w', padx=px)
        ttk.Label(inner, text=t("output_folder_hint"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(2, 0))

        ttk.Label(inner, text=t("output_dir"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        f_out = ttk.Frame(inner)
        f_out.pack(fill='x', padx=px)
        ttk.Entry(f_out, textvariable=self.output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f_out, text=t("browse"), style="Secondary.TButton",
                   command=self._select_output_dir).pack(side='right', padx=(6, 0))
        ttk.Label(inner, text=t("output_dir_hint"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(4, 12))

    # ------------------------------------------------------------------
    # Comparative Analysis tab
    # ------------------------------------------------------------------

    def _build_tab_comparative(self, parent):
        px  = THEME["pad_x"]
        pad = {"padx": px, "pady": THEME["pad_y"]}

        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        ttk.Label(inner,
                  text=t("cmp_title"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("cmp_desc"),
                  style="Sub.TLabel",
                  wraplength=520).pack(anchor='w', padx=px, pady=(0, 10))

        # ── Mouse list ───────────────────────────────────────────────
        ttk.Label(inner, text=t("mice_raw_csv"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        list_frame = ttk.Frame(inner, style="Card.TFrame")
        list_frame.pack(fill='x', padx=px, pady=(0, 4))
        self._cmp_list_inner = list_frame

        if not self._cmp_animals:
            self._cmp_add_mouse()
            self._cmp_add_mouse()
        else:
            self._cmp_refresh_list()

        ttk.Button(inner, text=t("add_mouse"), style="Secondary.TButton",
                   command=self._cmp_add_mouse).pack(anchor='w', padx=px, pady=(0, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Normalization (joint step only) ───────────────────────────
        ttk.Label(inner, text=t("joint_norm_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Radiobutton(
            inner,
            text=t("norm_robust_global_jt_radio"),
            variable=self._cmp_normalization, value='robust_global'
        ).pack(anchor='w', padx=px + 4, pady=1)
        ttk.Radiobutton(
            inner,
            text=t("norm_robust_jt_radio"),
            variable=self._cmp_normalization, value='robust'
        ).pack(anchor='w', padx=px + 4, pady=1)
        ttk.Radiobutton(
            inner,
            text=t("norm_cl_hybrid_jt_radio"),
            variable=self._cmp_normalization, value='cl_hybrid'
        ).pack(anchor='w', padx=px + 4, pady=1)
        ttk.Radiobutton(
            inner,
            text=t("norm_cl_jt_radio"),
            variable=self._cmp_normalization, value='cl'
        ).pack(anchor='w', padx=px + 4, pady=(1, 6))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Clustering method ────────────────────────────────────────
        ttk.Label(inner, text=t("method_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        for val, key in [("gmm", "gmm_desc"), ("kmeans", "kmeans_desc")]:
            ttk.Radiobutton(inner, text=t(key),
                            variable=self._cmp_method,
                            value=val).pack(anchor='w', padx=px + 4, pady=1)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── K parameters ─────────────────────────────────────────────
        ttk.Label(inner, text=t("k_params_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        def crow(label, widget_fn):
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=3)
            ttk.Label(f, text=label, width=30).pack(side='left')
            widget_fn(f)

        crow(t("k_min_label"),
             lambda f: ttk.Spinbox(f, from_=2, to=10,
                                    textvariable=self._cmp_k_min,
                                    width=6).pack(side='left'))
        crow(t("k_max_label"),
             lambda f: ttk.Spinbox(f, from_=3, to=15,
                                    textvariable=self._cmp_k_max,
                                    width=6).pack(side='left'))
        crow(t("k_override"),
             lambda f: ttk.Entry(f, textvariable=self._cmp_k_override,
                                  width=6).pack(side='left'))
        crow(t("gmm_n_init"),
             lambda f: ttk.Spinbox(f, from_=1, to=50,
                                    textvariable=self._cmp_n_init_gmm,
                                    width=6).pack(side='left'))
        crow(t("n_refs"),
             lambda f: ttk.Spinbox(f, from_=5, to=200,
                                    textvariable=self._cmp_n_refs,
                                    width=6).pack(side='left'))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Output ───────────────────────────────────────────────────
        ttk.Label(inner, text=t("output_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f_name = ttk.Frame(inner)
        f_name.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_name, text=t("pdf_name_label"), width=30).pack(side='left')
        ttk.Entry(f_name, textvariable=self._cmp_output_name,
                  width=22).pack(side='left')

        f_dir = ttk.Frame(inner)
        f_dir.pack(fill='x', padx=px, pady=3)
        ttk.Entry(f_dir, textvariable=self._cmp_output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f_dir, text=t("browse"), style="Secondary.TButton",
                   command=self._cmp_select_output).pack(side='right', padx=(6, 0))
        ttk.Label(inner,
                  text=t("defaults_to_first_mouse"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(2, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)

    # ── Comparative animal card helpers ──────────────────────────────────────

    def _cmp_add_mouse(self):
        self._cmp_animals.append({"files": []})
        self._cmp_refresh_list()

    def _cmp_remove_mouse(self, idx):
        if len(self._cmp_animals) <= 1:
            return
        self._cmp_animals.pop(idx)
        self._cmp_refresh_list()

    def _cmp_refresh_list(self):
        if self._cmp_list_inner is None:
            return
        for w in self._cmp_list_inner.winfo_children():
            w.destroy()
        for i, animal in enumerate(self._cmp_animals):
            self._cmp_build_card(i, animal)

    def _cmp_build_card(self, i, animal):
        card = ttk.Frame(self._cmp_list_inner)
        card.pack(fill='x', padx=4, pady=2)

        ttk.Label(card, text=t("mouse_n", n=i + 1),
                  font=THEME["font_bold"], width=9).pack(side='left', padx=(4, 2))

        n   = len(animal["files"])
        lbl = ttk.Label(card,
                        text=t("n_files", n=n) if n else t("no_files"),
                        style="Sub.TLabel", width=12)
        lbl.pack(side='left', padx=6)
        animal["_lbl"] = lbl

        ttk.Button(card, text=t("browse"), style="Secondary.TButton",
                   command=lambda a=animal: self._cmp_browse(a)
                   ).pack(side='left', padx=2)
        ttk.Button(card, text="📁", style="Secondary.TButton", width=3,
                   command=lambda a=animal: self._browse_animal_folder(a)
                   ).pack(side='left', padx=2)
        ttk.Button(card, text="✕", style="Secondary.TButton", width=2,
                   state='normal' if len(self._cmp_animals) > 1 else 'disabled',
                   command=lambda idx=i: self._cmp_remove_mouse(idx)
                   ).pack(side='right', padx=4)

    def _cmp_browse(self, animal):
        files = filedialog.askopenfilenames(
            title=t("dlg_select_csv_mouse"),
            filetypes=[("CSV files", "*.csv")])
        if files:
            animal["files"] = list(files)
            if animal.get("_lbl"):
                animal["_lbl"].config(
                    text=t("n_files", n=len(files)),
                    foreground=THEME["text"])

    def _cmp_select_output(self):
        path = filedialog.askdirectory(title=t("dlg_select_output"))
        if path:
            self._cmp_output_dir.set(path)

    # ── Comparative run ──────────────────────────────────────────────────────

    def _run_comparative(self):
        mice_list = []
        for i, animal in enumerate(self._cmp_animals):
            if not animal["files"]:
                continue
            name = os.path.basename(os.path.dirname(animal["files"][0]))
            mice_list.append({"name": name, "files": list(animal["files"])})

        if not mice_list:
            messagebox.showerror(t("err_title"),
                                 "Select CSV files for at least one mouse.")
            return

        default_dir = os.path.dirname(os.path.dirname(mice_list[0]["files"][0]))
        if not self._confirm_output_dir(self._cmp_output_dir, self._cmp_output_name, default_dir):
            return

        out_dir = self._cmp_output_dir.get()
        if not out_dir:
            out_dir = default_dir

        params = {
            "mice"      : mice_list,
            "method"    : self._cmp_method.get(),
            "k_range"   : range(self._cmp_k_min.get(), self._cmp_k_max.get() + 1),
            "k_override": self._cmp_k_override.get().strip(),
            "n_init"    : self._cmp_n_init_gmm.get(),
            "n_refs"    : self._cmp_n_refs.get(),
            "norm"      : self._cmp_normalization.get(),
            "out_dir"   : out_dir,
            "pdf_name"  : self._cmp_output_name.get() or "comparative_analysis",
        }

        self.btn_cmp_run.config(state='disabled', text=t("running"))
        self._write_log(f"[Compare] Starting — {len(mice_list)} mice")
        threading.Thread(target=self._run_comparative_pipeline,
                         args=(params,), daemon=True).start()

    def _run_comparative_pipeline(self, p):
        def log(msg, tag=None):
            self.queue.put({"type": "log", "text": f"[Compare] {msg}", "tag": tag})

        try:
            result = run_comparative_pdf(
                mice_data  = p["mice"],
                method     = p["method"],
                k_range    = p["k_range"],
                k_override = p["k_override"],
                n_init     = p["n_init"],
                n_refs     = p["n_refs"],
                norm       = p["norm"],
                out_dir    = p["out_dir"],
                pdf_name   = p["pdf_name"],
                log        = log,
            )
            self.queue.put({
                "type"    : "cmp_results",
                "pdf_path": result["pdf_path"],
                "summary" : (
                    f"{result['n_mice']} mice  |  "
                    f"Classic K={list(result['classic_ks'].values())}  |  "
                    f"Discovery K={result['disc_k']}  |  "
                    f"Joint K={result['joint_k']}"
                ),
            })
        except Exception as e:
            log(f"Error: {e}", "err")
        finally:
            self.queue.put({"type": "cmp_done"})

    def _build_tab_files(self, parent):
        pad = {"padx": THEME["pad_x"], "pady": THEME["pad_y"]}

        ttk.Label(parent, text=t("csv_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f1 = ttk.Frame(parent)
        f1.pack(fill='x', padx=THEME["pad_x"])
        self.lbl_csv = ttk.Label(f1, text=t("no_file"),
                                  style="Sub.TLabel")
        self.lbl_csv.pack(side='left', fill='x', expand=True)
        ttk.Button(f1, text=t("browse"),
                   style="Secondary.TButton",
                   command=self._select_csv).pack(side='right')

        self.lst_csv = tk.Listbox(
            parent, height=6,
            font=THEME["font_log"],
            bg=THEME["bg_input"],
            fg=THEME["text"],
            selectbackground=THEME["accent"],
            selectforeground="#FFFFFF",
            relief="flat", borderwidth=0,
            highlightthickness=1,
            highlightbackground=THEME["separator"],
            activestyle="none")
        self.lst_csv.pack(fill='x', padx=THEME["pad_x"], pady=(4, 12))

        ttk.Label(parent, text=t("nifti_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        f2 = ttk.Frame(parent)
        f2.pack(fill='x', padx=THEME["pad_x"])
        ttk.Entry(f2, textvariable=self.nifti_path,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f2, text=t("browse"),
                   style="Secondary.TButton",
                   command=self._select_nifti).pack(side='right', padx=(6, 0))

    def _build_tab_method(self, parent):
        # Scrollable wrapper so content is never clipped on small/high-DPI screens
        _canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        _sb     = ttk.Scrollbar(parent, orient='vertical', command=_canvas.yview)
        inner   = ttk.Frame(_canvas)
        inner.bind("<Configure>",
                   lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")))
        _win = _canvas.create_window((0, 0), window=inner, anchor='nw')
        _canvas.bind("<Configure>",
                     lambda e: _canvas.itemconfig(_win, width=e.width))
        _canvas.configure(yscrollcommand=_sb.set)
        _canvas.pack(side='left', fill='both', expand=True)
        _sb.pack(side='right', fill='y')

        def _scroll(e):
            _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        _canvas.bind("<MouseWheel>", _scroll)
        inner.bind("<MouseWheel>", _scroll)

        pad = {"padx": THEME["pad_x"], "pady": THEME["pad_y"]}
        px  = THEME["pad_x"]

        ttk.Label(inner, text=t("method_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        for val, desc_key in [
            ("gmm",    "gmm_desc"),
            ("srsc",   "srsc_desc"),
            ("kmeans", "kmeans_desc"),
        ]:
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=2)
            ttk.Radiobutton(f, text=t(desc_key),
                            variable=self.method,
                            value=val).pack(side='left', anchor='w')
            ttk.Button(f, text="?", width=2,
                       style="Secondary.TButton",
                       command=lambda v=val: _show_method_help(v)
                       ).pack(side='left', padx=(8, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=10)
        ttk.Label(inner, text=t("common_params"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)

        def row(p, label, widget_fn):
            f = ttk.Frame(p)
            f.pack(fill='x', padx=px, pady=4)
            ttk.Label(f, text=label, width=30).pack(side='left')
            widget_fn(f)
            return f

        row(inner, t("k_min"),
            lambda f: ttk.Spinbox(f, from_=2, to=10,
                                   textvariable=self.k_min,
                                   width=6).pack(side='left'))
        row(inner, t("k_max"),
            lambda f: ttk.Spinbox(f, from_=3, to=15,
                                   textvariable=self.k_max,
                                   width=6).pack(side='left'))
        row(inner, t("k_override"),
            lambda f: ttk.Entry(f, textvariable=self.k_override,
                                 width=6).pack(side='left'))

        self._frame_srsc = ttk.Frame(inner)
        ttk.Label(self._frame_srsc, text=t("srsc_params"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)
        sf = ttk.Frame(self._frame_srsc)
        sf.pack(fill='x', padx=px, pady=4)
        ttk.Label(sf, text=t("spatial_weight"), width=30).pack(side='left')
        self.lbl_sw = ttk.Label(sf, text=f"{SPATIAL_WEIGHT:.2f}", width=5)
        ttk.Scale(sf, from_=0.0, to=1.0, variable=self.spatial_w,
                  orient='horizontal', length=160,
                  command=lambda v: self.lbl_sw.config(
                      text=f"{float(v):.2f}")).pack(side='left')
        self.lbl_sw.pack(side='left')

        f_sig = ttk.Frame(self._frame_srsc)
        f_sig.pack(fill='x', padx=px, pady=2)
        ttk.Label(f_sig, text=t("sigma_feature"), width=30).pack(side='left')
        self.lbl_sigma = ttk.Label(f_sig, text="—", style="Sub.TLabel")
        self.lbl_sigma.pack(side='left')

        self._frame_gmm = ttk.Frame(inner)
        ttk.Label(self._frame_gmm, text=t("gmm_params"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)
        f_init = ttk.Frame(self._frame_gmm)
        f_init.pack(fill='x', padx=px, pady=4)
        ttk.Label(f_init, text=t("n_init_gmm"), width=30).pack(side='left')
        ttk.Spinbox(f_init, from_=1, to=50, textvariable=self.n_init_gmm,
                    width=6).pack(side='left')
        f_guard_gmm = ttk.Frame(self._frame_gmm)
        f_guard_gmm.pack(fill='x', padx=px, pady=2)
        ttk.Checkbutton(
            f_guard_gmm,
            text=t("gmm_guard", thr=GMM_SILHOUETTE_GUARD),
            variable=self.gmm_guard).pack(anchor='w')

        self._frame_nrefs = ttk.Frame(inner)
        f_nr = ttk.Frame(self._frame_nrefs)
        f_nr.pack(fill='x', padx=px)
        ttk.Label(f_nr, text=t("n_refs"), width=30).pack(side='left')
        ttk.Spinbox(f_nr, from_=5, to=200, textvariable=self.n_refs,
                    width=6).pack(side='left')

        self._frame_kmeans_guard = ttk.Frame(inner)
        ttk.Checkbutton(
            self._frame_kmeans_guard,
            text=t("kmeans_guard"),
            variable=self.kmeans_guard).pack(anchor='w', padx=px + 4)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=10)
        ttk.Label(inner, text=t("spatial_options"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(0, 4))
        ttk.Checkbutton(inner,
                        text=t("hierarchical_opt"),
                        variable=self.hierarchical_enabled,
                        ).pack(anchor='w', padx=px + 4, pady=2)
        ttk.Checkbutton(inner,
                        text=t("mrf_opt") + f"  (λ={MRF_LAMBDA}, thr={MRF_CONFIDENCE_THR})",
                        variable=self.mrf_enabled,
                        ).pack(anchor='w', padx=px + 4, pady=2)

    def _build_tab_output(self, parent):
        pad = {"padx": THEME["pad_x"], "pady": THEME["pad_y"]}
        px  = THEME["pad_x"]

        ttk.Label(parent, text=t("output_folder"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Entry(parent, textvariable=self.output_name,
                  width=20).pack(anchor='w', padx=px)
        ttk.Label(parent, text=t("output_folder_hint"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(2, 0))

        ttk.Separator(parent).pack(fill='x', padx=px, pady=12)

        ttk.Label(parent, text=t("output_dir"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        f = ttk.Frame(parent)
        f.pack(fill='x', padx=px)
        ttk.Entry(f, textvariable=self.output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f, text=t("browse"),
                   style="Secondary.TButton",
                   command=self._select_output_dir).pack(side='right', padx=(6, 0))
        ttk.Label(parent, text=t("output_dir_hint"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(4, 0))

    # ------------------------------------------------------------------
    # File selectors
    # ------------------------------------------------------------------

    def _select_csv(self):
        files = filedialog.askopenfilenames(
            title=t("dlg_select_csv"),
            filetypes=[("CSV files", "*.csv")])
        if files:
            self.csv_files = list(dict.fromkeys(files))
            self.lst_csv.delete(0, 'end')
            for f in self.csv_files:
                self.lst_csv.insert('end', os.path.basename(f))
            self.lbl_csv.config(
                text=t("n_files_selected", n=len(self.csv_files)),
                foreground=THEME["text"])

    def _select_csv_folder(self):
        folder = filedialog.askdirectory(title=t("dlg_select_folder"))
        if not folder:
            return
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith('.csv')
        )
        if not files:
            messagebox.showwarning(t("warn_empty_folder"), t("warn_no_csv_in_folder"))
            return
        self.csv_files = files
        self.lst_csv.delete(0, 'end')
        for f in files:
            self.lst_csv.insert('end', os.path.basename(f))
        self.lbl_csv.config(
            text=t("n_files_selected", n=len(files)),
            foreground=THEME["text"])

    def _select_nifti(self):
        path = filedialog.askopenfilename(
            title=t("dlg_select_nifti"),
            filetypes=[("NIfTI files", "*.nii.gz *.nii")])
        if path:
            self.nifti_path.set(path)

    def _select_output_dir(self):
        path = filedialog.askdirectory(title=t("dlg_select_output"))
        if path:
            self.output_dir.set(path)

    # ------------------------------------------------------------------
    # Queue listener
    # ------------------------------------------------------------------

    def listen_to_queue(self):
        try:
            while True:
                task   = self.queue.get_nowait()
                action = task.get("type")

                if action == "log":
                    self._write_log(task["text"], tag=task.get("tag"))
                    if "sigma_feature =" in task["text"] and hasattr(self, 'lbl_sigma'):
                        try:
                            val = task["text"].split("sigma_feature =")[1].split()[0]
                            self.lbl_sigma.config(text=val)
                        except (IndexError, ValueError):
                            pass

                elif action == "open_dialog":
                    self._safe_open_label_dialog(
                        task["centroids"],    task["parameters"],
                        task["best_k"],       task["colors"],
                        task["event"],        task["result"],
                        task["out_dir"],
                        cl_stats            = task.get("cl_stats"),
                        raw_centroids       = task.get("raw_centroids"),
                        scaling_info        = task.get("scaling_info"),
                        dti_mode            = task.get("dti_mode", "MD"),
                        three_pass_results  = task.get("three_pass_results"),
                        hier_meta           = task.get("hier_meta"))

                elif action == "napari_ready":
                    self._napari_params = task
                    self._write_log(t("log_napari_ready"), tag="ok")
                    self.btn_napari.config(state='normal')
                    self.btn_run.config(state='normal', text=t("run_analysis"))

                elif action == "done":
                    self.btn_run.config(state='normal', text=t("run_analysis"))

                elif action == "ia_done":
                    if self.btn_ia_run:
                        self.btn_ia_run.config(state='normal', text=t("run_ia"))

                elif action == "im_done":
                    if self.btn_im_run:
                        self.btn_im_run.config(state='normal', text=t("run_im"))

                elif action == "disc_done":
                    if self.btn_disc_run:
                        self.btn_disc_run.config(state='normal', text=t("run_discovery"))

                elif action == "disc_results":
                    self._show_discovery_results(task["text"], task["pdf_path"])

                elif action == "jt_done":
                    if self.btn_jt_run:
                        self.btn_jt_run.config(state='normal', text=t("run_joint"))

                elif action == "jt_results":
                    self._show_joint_results(
                        task["summary"], task["output_dir"],
                        dtopo_report  = task.get("dtopo_report", {}),
                        dtopo_weight  = task.get("dtopo_weight", 1.0),
                        dtopo_min_frac= task.get("dtopo_min_frac", 0.05))

                elif action == "cmp_done":
                    if self.btn_cmp_run:
                        self.btn_cmp_run.config(state='normal', text=t("run_comparative"))

                elif action == "cmp_results":
                    self._show_comparative_results(task["summary"], task["pdf_path"])

                elif action == "atl_done":
                    if self.btn_atl_run:
                        self.btn_atl_run.config(state='normal', text=t("run_atlas"))

                elif action == "atl_results":
                    self._show_atlas_results(task["summary"], task["pdf_path"])

                elif action == "error":
                    self._write_log(task["text"], tag="err")
                    self.btn_run.config(state='normal', text=t("run_analysis"))

                self.queue.task_done()

        except queue.Empty:
            pass

        self.after(100, self.listen_to_queue)

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _write_log(self, msg, tag=None):
        if threading.current_thread() is not threading.main_thread():
            self.queue.put({"type": "log", "text": msg, "tag": tag})
            return
        self.log.config(state='normal')
        if tag is None and "[ERROR]" in msg:
            tag = "err"
        if tag:
            self.log.insert('end', msg + "\n", tag)
        else:
            self.log.insert('end', msg + "\n")
        self.log.see('end')
        self.log.config(state='disabled')

    # ------------------------------------------------------------------
    # Label dialog (main thread)
    # ------------------------------------------------------------------

    def _safe_open_label_dialog(self, centroids, parameters,
                                 best_k, colors, event, result, out_dir,
                                 cl_stats=None, raw_centroids=None,
                                 scaling_info=None, dti_mode='MD',
                                 three_pass_results=None, hier_meta=None):
        try:
            res = _label_dialog(centroids, parameters, best_k, colors,
                                theme=THEME,
                                cl_stats=cl_stats,
                                raw_centroids=raw_centroids,
                                scaling_info=scaling_info,
                                dti_mode=dti_mode,
                                three_pass_results=three_pass_results,
                                hier_meta=hier_meta)
            result.update(res)
            label_path = os.path.join(out_dir, 'habitat_labels.json')
            with open(label_path, 'w') as f:
                json.dump(res, f, indent=2)
            self._write_log(t("log_labels_saved", path=label_path), tag="ok")
        except Exception as e:
            self._write_log(f"[ERROR labeling] {e}", tag="err")
        finally:
            event.set()

    # ------------------------------------------------------------------
    # Comparison tab
    # ------------------------------------------------------------------

    def _build_tab_comparison(self, parent):
        sub = ttk.Notebook(parent)
        sub.pack(fill='both', expand=True, padx=6, pady=6)
        ia_tab = ttk.Frame(sub)
        im_tab = ttk.Frame(sub)
        sub.add(ia_tab, text=t("tab_ia"))
        sub.add(im_tab, text=t("tab_im"))
        self._build_ia_panel(ia_tab)
        self._build_im_panel(im_tab)
        self._comp_sub = sub

    # --- Inter-animal panel ---

    def _build_ia_panel(self, parent):
        px  = THEME["pad_x"]
        pad = {"padx": px, "pady": THEME["pad_y"]}

        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        ttk.Label(inner, text=t("ia_animals"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        list_frame = ttk.Frame(inner, style="Card.TFrame")
        list_frame.pack(fill='x', padx=px, pady=(0, 4))
        self._ia_list_inner = list_frame
        if not self._ia_animals:
            self._ia_add_animal()
            self._ia_add_animal()
        else:
            self._ia_refresh_list()

        ttk.Button(inner, text=t("add_mouse"), style="Secondary.TButton",
                   command=self._ia_add_animal).pack(anchor='w', padx=px, pady=(0, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("method_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        for val, desc_key in [
            ("gmm",    "gmm_desc"),
            ("srsc",   "srsc_desc"),
            ("kmeans", "kmeans_desc"),
        ]:
            ttk.Radiobutton(inner, text=t(desc_key),
                            variable=self._ia_method,
                            value=val).pack(anchor='w', padx=px+4, pady=1)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("parameters_section"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)

        def ia_row(label, widget_fn):
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=3)
            ttk.Label(f, text=label, width=28).pack(side='left')
            widget_fn(f)

        ia_row(t("k_min"),
               lambda f: ttk.Spinbox(f, from_=2, to=10,
                                      textvariable=self._ia_k_min,
                                      width=6).pack(side='left'))
        ia_row(t("k_max"),
               lambda f: ttk.Spinbox(f, from_=3, to=15,
                                      textvariable=self._ia_k_max,
                                      width=6).pack(side='left'))
        ia_row(t("k_override"),
               lambda f: ttk.Entry(f, textvariable=self._ia_k_override,
                                    width=6).pack(side='left'))

        f_sw = ttk.Frame(inner)
        f_sw.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_sw, text=t("srsc_spatial_weight"), width=28).pack(side='left')
        lbl_sw_ia = ttk.Label(f_sw, text=f"{SPATIAL_WEIGHT:.2f}", width=5)
        ttk.Scale(f_sw, from_=0.0, to=1.0, variable=self._ia_spatial_w,
                  orient='horizontal', length=140,
                  command=lambda v: lbl_sw_ia.config(text=f"{float(v):.2f}")
                  ).pack(side='left')
        lbl_sw_ia.pack(side='left')

        ia_row(t("gmm_n_init"),
               lambda f: ttk.Spinbox(f, from_=1, to=50,
                                      textvariable=self._ia_n_init_gmm,
                                      width=6).pack(side='left'))
        ia_row(t("n_refs"),
               lambda f: ttk.Spinbox(f, from_=5, to=200,
                                      textvariable=self._ia_n_refs,
                                      width=6).pack(side='left'))

        ttk.Checkbutton(inner, text=t("gmm_guard_short", thr=GMM_SILHOUETTE_GUARD),
                        variable=self._ia_gmm_guard).pack(anchor='w', padx=px+4, pady=2)
        ttk.Checkbutton(inner, text=t("kmeans_guard_short"),
                        variable=self._ia_kmeans_guard).pack(anchor='w', padx=px+4, pady=2)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("spatial_options"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(0, 2))
        ttk.Checkbutton(inner, text=t("hierarchical_short"),
                        variable=self._ia_hierarchical_enabled,
                        ).pack(anchor='w', padx=px+4, pady=1)
        ttk.Checkbutton(inner, text=t("mrf_short"),
                        variable=self._ia_mrf_enabled,
                        ).pack(anchor='w', padx=px+4, pady=1)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("output_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f_name = ttk.Frame(inner)
        f_name.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_name, text=t("folder_name"), width=28).pack(side='left')
        ttk.Entry(f_name, textvariable=self._ia_output_name,
                  width=22).pack(side='left')

        f_dir = ttk.Frame(inner)
        f_dir.pack(fill='x', padx=px, pady=3)
        ttk.Entry(f_dir, textvariable=self._ia_output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f_dir, text=t("browse"), style="Secondary.TButton",
                   command=self._ia_select_output).pack(side='right', padx=(6, 0))
        ttk.Label(inner, text=t("ia_output_hint"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(2, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)

    def _ia_add_animal(self):
        self._ia_animals.append({"files": []})
        self._ia_refresh_list()

    def _ia_remove_animal(self, idx):
        if len(self._ia_animals) <= 1:
            return
        self._ia_animals.pop(idx)
        self._ia_refresh_list()

    def _ia_refresh_list(self):
        if self._ia_list_inner is None:
            return
        for w in self._ia_list_inner.winfo_children():
            w.destroy()
        for i, animal in enumerate(self._ia_animals):
            self._ia_build_card(i, animal)

    def _ia_build_card(self, i, animal):
        card = ttk.Frame(self._ia_list_inner)
        card.pack(fill='x', padx=4, pady=2)

        ttk.Label(card, text=t("mouse_n", n=i+1),
                  font=THEME["font_bold"], width=9).pack(side='left', padx=(4, 2))

        n   = len(animal["files"])
        lbl = ttk.Label(card,
                        text=t("n_files", n=n) if n else t("no_files"),
                        style="Sub.TLabel", width=12)
        lbl.pack(side='left', padx=6)
        animal["_lbl"] = lbl

        ttk.Button(card, text=t("browse"), style="Secondary.TButton",
                   command=lambda a=animal: self._ia_browse(a)
                   ).pack(side='left', padx=2)
        ttk.Button(card, text="📁", style="Secondary.TButton", width=3,
                   command=lambda a=animal: self._browse_animal_folder(a)
                   ).pack(side='left', padx=2)

        ttk.Button(card, text="✕", style="Secondary.TButton", width=2,
                   state='normal' if len(self._ia_animals) > 1 else 'disabled',
                   command=lambda idx=i: self._ia_remove_animal(idx)
                   ).pack(side='right', padx=4)

    def _ia_browse(self, animal):
        files = filedialog.askopenfilenames(
            title=t("dlg_select_csv_mouse"),
            filetypes=[("CSV files", "*.csv")])
        if files:
            animal["files"] = list(files)
            if animal.get("_lbl"):
                animal["_lbl"].config(
                    text=t("n_files", n=len(files)),
                    foreground=THEME["text"])

    def _ia_select_output(self):
        path = filedialog.askdirectory(title=t("dlg_select_output"))
        if path:
            self._ia_output_dir.set(path)

    def _run_inter_animal(self):
        if len(self._ia_animals) < 2:
            messagebox.showerror(t("err_title"), t("err_min_mice"))
            return
        if any(not a["files"] for a in self._ia_animals):
            messagebox.showerror(t("err_title"), t("err_mouse_no_csv"))
            return
        if self._ia_k_min.get() >= self._ia_k_max.get():
            messagebox.showerror(t("err_title"), t("err_k_range"))
            return

        default_dir = os.path.dirname(self._ia_animals[0]["files"][0])
        if not self._confirm_output_dir(self._ia_output_dir, self._ia_output_name, default_dir):
            return

        if self._ia_output_dir.get():
            out_root = os.path.join(self._ia_output_dir.get(),
                                    self._ia_output_name.get())
        else:
            out_root = os.path.join(default_dir, self._ia_output_name.get())

        params = {
            "animals"             : [{"files": list(a["files"])} for a in self._ia_animals],
            "method"              : self._ia_method.get(),
            "k_min"               : self._ia_k_min.get(),
            "k_max"               : self._ia_k_max.get(),
            "k_override"          : self._ia_k_override.get().strip(),
            "spatial_w"           : self._ia_spatial_w.get(),
            "n_refs"              : self._ia_n_refs.get(),
            "n_init_gmm"          : self._ia_n_init_gmm.get(),
            "gmm_guard"           : self._ia_gmm_guard.get(),
            "kmeans_guard"        : self._ia_kmeans_guard.get(),
            "mrf_enabled"         : self._ia_mrf_enabled.get(),
            "hierarchical_enabled": self._ia_hierarchical_enabled.get(),
            "out_root"            : out_root,
        }

        self.btn_ia_run.config(state='disabled', text=t("running"))
        n = len(params["animals"])
        self._write_log(t("log_ia_start", n=n))
        threading.Thread(target=self._run_ia_pipeline,
                         args=(params,), daemon=True).start()

    def _run_ia_pipeline(self, p):
        def log(msg, tag=None):
            self.queue.put({"type": "log", "text": msg, "tag": tag})

        try:
            out_root     = p["out_root"]
            os.makedirs(out_root, exist_ok=True)
            k_range      = range(p["k_min"], p["k_max"] + 1)
            method       = p["method"]
            k_ov         = p["k_override"]
            k_override   = int(k_ov) if k_ov.isdigit() else None
            gmm_guard    = GMM_SILHOUETTE_GUARD if p["gmm_guard"] else None
            kmeans_guard = bool(p["kmeans_guard"])

            run_results = []
            for i, animal in enumerate(p["animals"]):
                name    = f"mouse_{i+1}"
                out_dir = os.path.join(out_root, name)
                os.makedirs(out_dir, exist_ok=True)
                log(t("log_mouse_loading", n=i+1))

                parameters, df_all = load_data(animal["files"])
                parameters, dti_mode = resolve_dti_parameters(parameters)
                df_scaled, features_scaled, scaling_info = robust_scale(df_all, parameters)
                plot_correlation(df_scaled, parameters, out_dir)

                use_hier_ia    = p.get("hierarchical_enabled", HIERARCHICAL_ENABLED)
                use_mrf_ia     = p.get("mrf_enabled", MRF_ENABLED)
                hier_meta      = None
                dtopo_meta_ia  = None   # set below if necrosis guard passes
                if use_hier_ia:
                    try:
                        labels_hier_ia, hier_meta = hierarchical_segment(
                            features_scaled, df_all, parameters,
                            n_init_gmm=p["n_init_gmm"],
                            method=method,
                            spatial_weight=p["spatial_w"],
                            n_refs=p["n_refs"],
                            n_jobs=N_JOBS_GAP,
                            gmm_silhouette_guard=gmm_guard,
                            kmeans_gap_guard=kmeans_guard)
                    except Exception:
                        _dbg_path = os.path.join(out_dir, 'hier_debug.txt')
                        _full_tb = traceback.format_exc()
                        with open(_dbg_path, 'w') as _dbg_f:
                            _dbg_f.write(_full_tb)
                        raise
                    if hier_meta.get('hierarchical_valid'):
                        labels      = labels_hier_ia
                        best_k      = hier_meta['best_k']
                        extra_info  = {}
                        d_topo_hier, nec_id_hier = compute_necrotic_dtopo(
                            df_all, labels_hier_ia, hier_meta)
                        dtopo_meta_ia = {
                            'has_necrosis'    : nec_id_hier is not None,
                            'd_topo'          : d_topo_hier,
                            'necrotic_cluster': nec_id_hier,
                        }
                    else:
                        hier_meta = None

                if not use_hier_ia or hier_meta is None:
                    features_clust, dtopo_meta_ia = prepare_features(
                        features_scaled, df_all, parameters, n_init_gmm=p["n_init_gmm"])
                    if k_override:
                        best_k = k_override
                        log(t("log_mouse_optimal_k", n=i+1, k=best_k))
                        labels, extra_info = run_clustering(
                            features_clust, df_all, best_k,
                            method=method, spatial_weight=p["spatial_w"],
                            n_init_gmm=p["n_init_gmm"])
                    else:
                        best_k, gmm_cache = select_optimal_k(
                            features_clust, method=method, df_all=df_all,
                            k_range=k_range, n_refs=p["n_refs"], n_jobs=N_JOBS_GAP,
                            spatial_weight=p["spatial_w"], n_init_gmm=p["n_init_gmm"],
                            gmm_silhouette_guard=gmm_guard, kmeans_gap_guard=kmeans_guard,
                            output_dir=out_dir)
                        log(t("log_mouse_optimal_k", n=i+1, k=best_k))
                        labels, extra_info, best_k = run_clustering_with_size_guard(
                            features_clust, features_scaled, df_all, best_k,
                            method=method, spatial_weight=p["spatial_w"],
                            n_init_gmm=p["n_init_gmm"],
                            parameters=parameters,
                            dtopo_meta=dtopo_meta_ia,
                            k_min=p["k_min"],
                            gmm_cache=gmm_cache)

                df_scaled['Habitat'] = labels
                proba = extra_info.get('proba')

                if method == 'gmm' and proba is not None:
                    plot_gmm_probabilities(labels, df_scaled, proba,
                                           HABITAT_COLORS_HEX[:best_k], best_k, out_dir)

                centroids = show_habitat_map(labels, df_scaled, features_scaled,
                                             parameters, best_k, out_dir)
                inject_dtopo_centroids(centroids, labels, dtopo_meta_ia)
                save_centroids_json(centroids, out_dir)
                raw_centroids = {
                    c: {param: float(df_all[param].values[labels == c].mean())
                        for param in parameters}
                    for c in range(best_k)
                }
                features_by_cluster = {c: features_scaled[labels == c]
                                       for c in range(best_k)}
                three_pass = label_clusters_ws(centroids, parameters)

                if use_mrf_ia and proba is not None:
                    labels = mrf_boundary_smooth(labels, proba, df_all, three_pass)
                    df_scaled['Habitat'] = labels

                confidence_arr, ambiguous_arr, _ = analyze_spatial_coherence(
                    labels, df_all, proba, best_k, out_dir)
                df_scaled['label_confidence'] = confidence_arr
                df_scaled['label_ambiguous']  = ambiguous_arr.astype(int)

                cl_stats = load_cl_stats(animal["files"], parameters)
                colors       = HABITAT_COLORS_HEX[:best_k]
                label_event  = threading.Event()
                label_result = {}

                self.queue.put({
                    "type"              : "open_dialog",
                    "centroids"         : centroids,
                    "parameters"        : parameters,
                    "best_k"            : best_k,
                    "colors"            : colors,
                    "event"             : label_event,
                    "result"            : label_result,
                    "out_dir"           : out_dir,
                    "cl_stats"          : cl_stats,
                    "raw_centroids"     : raw_centroids,
                    "scaling_info"      : scaling_info,
                    "dti_mode"          : dti_mode,
                    "three_pass_results": three_pass,
                    "hier_meta"         : hier_meta,
                })
                label_event.wait()
                habitat_labels = label_result

                plot_habitat_barchart(df_scaled, parameters, habitat_labels, out_dir)
                plot_heatmap_profiles(df_scaled, parameters, habitat_labels, best_k, out_dir)
                plot_radar_profiles(df_scaled, parameters, habitat_labels, out_dir)
                export_results(df_scaled, parameters, habitat_labels, out_dir)
                save_run_meta(scaling_info, cl_stats, out_dir)
                generate_report_pdf(out_dir, df_scaled, parameters,
                                    habitat_labels, best_k, method)
                log(t("log_mouse_done", n=i+1, path=out_dir), tag="ok")

                run_results.append({
                    "name"   : f"Mouse {i+1}",
                    "csv"    : os.path.join(out_dir, "habitats_result.csv"),
                    "method" : method,
                    "best_k" : best_k,
                    "out_dir": out_dir,
                    "files"  : animal["files"],
                })

            cmp_dir = os.path.join(out_root, "comparison")
            os.makedirs(cmp_dir, exist_ok=True)
            run_comparison_v4(run_results, cmp_dir)
            log(t("log_comparison_saved", path=cmp_dir), tag="ok")
            log(t("log_ia_complete"), tag="ok")

        except Exception as e:
            self.queue.put({"type": "error",
                            "text": f"\n[ERROR inter-animal] {e}\n{traceback.format_exc()}"})
        finally:
            self.queue.put({"type": "ia_done"})

    # --- Inter-method panel ---

    def _build_im_panel(self, parent):
        px  = THEME["pad_x"]
        pad = {"padx": px, "pady": THEME["pad_y"]}

        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        ttk.Label(inner, text=t("im_csv_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f_browse = ttk.Frame(inner)
        f_browse.pack(fill='x', padx=px)
        self._im_lbl_csv = ttk.Label(f_browse, text=t("no_file"),
                                      style="Sub.TLabel")
        self._im_lbl_csv.pack(side='left', fill='x', expand=True)
        ttk.Button(f_browse, text="📁", style="Secondary.TButton", width=3,
                   command=self._im_select_files_folder).pack(side='right', padx=(0, 2))
        ttk.Button(f_browse, text=t("browse"), style="Secondary.TButton",
                   command=self._im_select_files).pack(side='right')

        self._im_lst_csv = tk.Listbox(
            inner, height=4,
            font=THEME["font_log"],
            bg=THEME["bg_input"], fg=THEME["text"],
            selectbackground=THEME["accent"],
            selectforeground="#FFFFFF",
            relief="flat", borderwidth=0,
            highlightthickness=1,
            highlightbackground=THEME["separator"],
            activestyle="none")
        self._im_lst_csv.pack(fill='x', padx=px, pady=(4, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("im_methods_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        for key, desc_key in [
            ("gmm",    "gmm_desc"),
            ("srsc",   "srsc_desc"),
            ("kmeans", "kmeans_desc"),
        ]:
            var = {"gmm": self._im_use_gmm,
                   "srsc": self._im_use_srsc,
                   "kmeans": self._im_use_kmeans}[key]
            ttk.Checkbutton(inner, text=t(desc_key),
                            variable=var).pack(anchor='w', padx=px+4, pady=1)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("parameters_section"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)

        def im_row(label, widget_fn):
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=3)
            ttk.Label(f, text=label, width=28).pack(side='left')
            widget_fn(f)

        im_row(t("k_min"),
               lambda f: ttk.Spinbox(f, from_=2, to=10,
                                      textvariable=self._im_k_min,
                                      width=6).pack(side='left'))
        im_row(t("k_max"),
               lambda f: ttk.Spinbox(f, from_=3, to=15,
                                      textvariable=self._im_k_max,
                                      width=6).pack(side='left'))
        im_row(t("k_override"),
               lambda f: ttk.Entry(f, textvariable=self._im_k_override,
                                    width=6).pack(side='left'))

        f_sw = ttk.Frame(inner)
        f_sw.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_sw, text=t("srsc_spatial_weight"), width=28).pack(side='left')
        lbl_sw_im = ttk.Label(f_sw, text=f"{SPATIAL_WEIGHT:.2f}", width=5)
        ttk.Scale(f_sw, from_=0.0, to=1.0, variable=self._im_spatial_w,
                  orient='horizontal', length=140,
                  command=lambda v: lbl_sw_im.config(text=f"{float(v):.2f}")
                  ).pack(side='left')
        lbl_sw_im.pack(side='left')

        im_row(t("gmm_n_init"),
               lambda f: ttk.Spinbox(f, from_=1, to=50,
                                      textvariable=self._im_n_init_gmm,
                                      width=6).pack(side='left'))
        im_row(t("n_refs"),
               lambda f: ttk.Spinbox(f, from_=5, to=200,
                                      textvariable=self._im_n_refs,
                                      width=6).pack(side='left'))

        ttk.Checkbutton(inner, text=t("gmm_guard_short", thr=GMM_SILHOUETTE_GUARD),
                        variable=self._im_gmm_guard).pack(anchor='w', padx=px+4, pady=2)
        ttk.Checkbutton(inner, text=t("kmeans_guard_short"),
                        variable=self._im_kmeans_guard).pack(anchor='w', padx=px+4, pady=2)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("spatial_options"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(0, 2))
        ttk.Checkbutton(inner, text=t("hierarchical_short"),
                        variable=self._im_hierarchical_enabled,
                        ).pack(anchor='w', padx=px+4, pady=1)
        ttk.Checkbutton(inner, text=t("mrf_short"),
                        variable=self._im_mrf_enabled,
                        ).pack(anchor='w', padx=px+4, pady=1)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("output_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f_name = ttk.Frame(inner)
        f_name.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_name, text=t("folder_name"), width=28).pack(side='left')
        ttk.Entry(f_name, textvariable=self._im_output_name,
                  width=22).pack(side='left')

        f_dir = ttk.Frame(inner)
        f_dir.pack(fill='x', padx=px, pady=3)
        ttk.Entry(f_dir, textvariable=self._im_output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f_dir, text=t("browse"), style="Secondary.TButton",
                   command=self._im_select_output).pack(side='right', padx=(6, 0))
        ttk.Label(inner, text=t("im_output_hint"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(2, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)

    def _im_select_files(self):
        files = filedialog.askopenfilenames(
            title=t("dlg_select_csv"),
            filetypes=[("CSV files", "*.csv")])
        if files:
            self._im_csv_files = list(dict.fromkeys(files))
            self._im_lst_csv.delete(0, 'end')
            for f in self._im_csv_files:
                self._im_lst_csv.insert('end', os.path.basename(f))
            self._im_lbl_csv.config(
                text=t("n_files_selected", n=len(self._im_csv_files)),
                foreground=THEME["text"])

    def _im_select_files_folder(self):
        folder = filedialog.askdirectory(title=t("dlg_select_folder"))
        if not folder:
            return
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith('.csv')
        )
        if not files:
            messagebox.showwarning(t("warn_empty_folder"), t("warn_no_csv_in_folder"))
            return
        self._im_csv_files = files
        self._im_lst_csv.delete(0, 'end')
        for f in files:
            self._im_lst_csv.insert('end', os.path.basename(f))
        self._im_lbl_csv.config(
            text=t("n_files_selected", n=len(files)),
            foreground=THEME["text"])

    def _im_select_output(self):
        path = filedialog.askdirectory(title=t("dlg_select_output"))
        if path:
            self._im_output_dir.set(path)

    def _run_inter_method(self):
        if not self._im_csv_files:
            messagebox.showerror(t("err_title"), t("err_no_csv"))
            return
        methods = [m for m, v in [("gmm",    self._im_use_gmm),
                                   ("srsc",   self._im_use_srsc),
                                   ("kmeans", self._im_use_kmeans)]
                   if v.get()]
        if len(methods) < 2:
            messagebox.showerror(t("err_title"), t("err_min_methods"))
            return
        if self._im_k_min.get() >= self._im_k_max.get():
            messagebox.showerror(t("err_title"), t("err_k_range"))
            return

        default_dir = os.path.dirname(self._im_csv_files[0])
        if not self._confirm_output_dir(self._im_output_dir, self._im_output_name, default_dir):
            return

        if self._im_output_dir.get():
            out_root = os.path.join(self._im_output_dir.get(),
                                    self._im_output_name.get())
        else:
            out_root = os.path.join(default_dir, self._im_output_name.get())

        params = {
            "csv_files"           : list(self._im_csv_files),
            "methods"             : methods,
            "k_min"               : self._im_k_min.get(),
            "k_max"               : self._im_k_max.get(),
            "k_override"          : self._im_k_override.get().strip(),
            "spatial_w"           : self._im_spatial_w.get(),
            "n_refs"              : self._im_n_refs.get(),
            "n_init_gmm"          : self._im_n_init_gmm.get(),
            "gmm_guard"           : self._im_gmm_guard.get(),
            "kmeans_guard"        : self._im_kmeans_guard.get(),
            "mrf_enabled"         : self._im_mrf_enabled.get(),
            "hierarchical_enabled": self._im_hierarchical_enabled.get(),
            "out_root"            : out_root,
        }

        self.btn_im_run.config(state='disabled', text=t("running"))
        label = ', '.join(m.upper() for m in methods)
        self._write_log(t("log_im_start", methods=label))
        threading.Thread(target=self._run_im_pipeline,
                         args=(params,), daemon=True).start()

    def _run_im_pipeline(self, p):
        def log(msg, tag=None):
            self.queue.put({"type": "log", "text": msg, "tag": tag})

        try:
            out_root     = p["out_root"]
            os.makedirs(out_root, exist_ok=True)
            k_range      = range(p["k_min"], p["k_max"] + 1)
            k_ov         = p["k_override"]
            k_override   = int(k_ov) if k_ov.isdigit() else None
            gmm_guard    = GMM_SILHOUETTE_GUARD if p["gmm_guard"] else None
            kmeans_guard = bool(p["kmeans_guard"])

            log(t("log_loading_shared"))
            parameters, df_all = load_data(p["csv_files"])
            parameters, dti_mode = resolve_dti_parameters(parameters)
            cl_stats = load_cl_stats(p["csv_files"], parameters)

            run_results = []
            for method in p["methods"]:
                name    = method.upper()
                out_dir = os.path.join(out_root, f"method_{method}")
                os.makedirs(out_dir, exist_ok=True)
                log(t("log_method_start", name=name))

                df_scaled, features_scaled, scaling_info = robust_scale(df_all, parameters)
                plot_correlation(df_scaled, parameters, out_dir)

                features_clust, dtopo_meta_im = prepare_features(
                    features_scaled, df_all, parameters, n_init_gmm=p["n_init_gmm"])
                if k_override:
                    best_k = k_override
                    log(t("log_method_optimal_k", name=name, k=best_k))
                    labels, extra_info = run_clustering(
                        features_clust, df_all, best_k,
                        method=method, spatial_weight=p["spatial_w"],
                        n_init_gmm=p["n_init_gmm"])
                else:
                    best_k, gmm_cache = select_optimal_k(
                        features_clust, method=method, df_all=df_all,
                        k_range=k_range, n_refs=p["n_refs"], n_jobs=N_JOBS_GAP,
                        spatial_weight=p["spatial_w"], n_init_gmm=p["n_init_gmm"],
                        gmm_silhouette_guard=gmm_guard, kmeans_gap_guard=kmeans_guard,
                        output_dir=out_dir)
                    log(t("log_method_optimal_k", name=name, k=best_k))
                    labels, extra_info, best_k = run_clustering_with_size_guard(
                        features_clust, features_scaled,
                        df_all, best_k,
                        method=method, spatial_weight=p["spatial_w"],
                        n_init_gmm=p["n_init_gmm"],
                        parameters=parameters,
                        dtopo_meta=dtopo_meta_im,
                        k_min=p["k_min"],
                        gmm_cache=gmm_cache)
                df_scaled['Habitat'] = labels
                proba = extra_info.get('proba')

                if method == 'gmm' and proba is not None:
                    plot_gmm_probabilities(labels, df_scaled, proba,
                                           HABITAT_COLORS_HEX[:best_k], best_k, out_dir)

                centroids = show_habitat_map(labels, df_scaled, features_scaled,
                                             parameters, best_k, out_dir)
                inject_dtopo_centroids(centroids, labels, dtopo_meta_im)
                save_centroids_json(centroids, out_dir)
                raw_centroids = {
                    c: {param: float(df_all[param].values[labels == c].mean())
                        for param in parameters}
                    for c in range(best_k)
                }
                features_by_cluster = {c: features_scaled[labels == c]
                                       for c in range(best_k)}
                three_pass = label_clusters_ws(centroids, parameters)

                if p.get("mrf_enabled", MRF_ENABLED) and proba is not None:
                    labels = mrf_boundary_smooth(labels, proba, df_all, three_pass)
                    df_scaled['Habitat'] = labels

                confidence_arr, ambiguous_arr, _ = analyze_spatial_coherence(
                    labels, df_all, proba, best_k, out_dir)
                df_scaled['label_confidence'] = confidence_arr
                df_scaled['label_ambiguous']  = ambiguous_arr.astype(int)
                colors       = HABITAT_COLORS_HEX[:best_k]
                label_event  = threading.Event()
                label_result = {}

                self.queue.put({
                    "type"              : "open_dialog",
                    "centroids"         : centroids,
                    "parameters"        : parameters,
                    "best_k"            : best_k,
                    "colors"            : colors,
                    "event"             : label_event,
                    "result"            : label_result,
                    "out_dir"           : out_dir,
                    "cl_stats"          : cl_stats,
                    "raw_centroids"     : raw_centroids,
                    "scaling_info"      : scaling_info,
                    "dti_mode"          : dti_mode,
                    "three_pass_results": three_pass,
                })
                label_event.wait()
                habitat_labels = label_result

                plot_habitat_barchart(df_scaled, parameters, habitat_labels, out_dir)
                plot_heatmap_profiles(df_scaled, parameters, habitat_labels, best_k, out_dir)
                plot_radar_profiles(df_scaled, parameters, habitat_labels, out_dir)
                export_results(df_scaled, parameters, habitat_labels, out_dir)
                save_run_meta(scaling_info, cl_stats, out_dir)
                generate_report_pdf(out_dir, df_scaled, parameters,
                                    habitat_labels, best_k, method)
                log(t("log_method_done", name=name, path=out_dir), tag="ok")

                run_results.append({
                    "name"   : name,
                    "csv"    : os.path.join(out_dir, "habitats_result.csv"),
                    "method" : method,
                    "best_k" : best_k,
                    "out_dir": out_dir,
                    "files"  : p["csv_files"],
                })

            cmp_dir = os.path.join(out_root, "comparison")
            os.makedirs(cmp_dir, exist_ok=True)
            run_comparison_v4(run_results, cmp_dir)
            log(t("log_comparison_saved", path=cmp_dir), tag="ok")
            log(t("log_im_complete"), tag="ok")

        except Exception as e:
            self.queue.put({"type": "error",
                            "text": f"\n[ERROR inter-method] {e}\n{traceback.format_exc()}"})
        finally:
            self.queue.put({"type": "im_done"})

    # ------------------------------------------------------------------
    # Napari
    # ------------------------------------------------------------------

    def _open_napari(self):
        self.destroy()

    def _show_comparative_results(self, summary, pdf_path):
        win = tk.Toplevel(self)
        win.title(t("wintitle_cmp_done"))
        win.configure(bg=THEME["bg"])
        win.geometry("500x200")

        ttk.Label(win, text=t("cmp_done_header"),
                  style="Section.TLabel").pack(pady=(18, 4))
        ttk.Label(win, text=summary,
                  style="Sub.TLabel", wraplength=460).pack(pady=4)
        ttk.Label(win, text=pdf_path,
                  style="Sub.TLabel", wraplength=460).pack(pady=4)

        btn_row = ttk.Frame(win)
        btn_row.pack(pady=12)
        ttk.Button(btn_row, text=t("open_pdf"),
                   style="Primary.TButton",
                   command=lambda: os.startfile(pdf_path)
                   ).pack(side='left', padx=8)
        ttk.Button(btn_row, text=t("open_folder"),
                   style="Secondary.TButton",
                   command=lambda: os.startfile(os.path.dirname(pdf_path))
                   ).pack(side='left', padx=8)
        ttk.Button(btn_row, text=t("close"),
                   style="Secondary.TButton",
                   command=win.destroy
                   ).pack(side='left', padx=8)

    # ------------------------------------------------------------------
    # Atlas tab
    # ------------------------------------------------------------------

    def _build_tab_atlas(self, parent):
        px  = THEME["pad_x"]
        pad = {"padx": px, "pady": THEME["pad_y"]}

        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        ttk.Label(inner,
                  text=t("atl_title"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("atl_desc"),
                  style="Sub.TLabel",
                  wraplength=520).pack(anchor='w', padx=px, pady=(0, 10))

        # ── Group names ──────────────────────────────────────────────────────
        ttk.Label(inner, text=t("group_names_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("group_names_desc"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(0, 4))

        gnames_frame = ttk.Frame(inner, style="Card.TFrame")
        gnames_frame.pack(fill='x', padx=px, pady=(0, 4))
        self._atl_gnames_frame = gnames_frame
        self._atl_refresh_group_names(gnames_frame)

        btn_row_g = ttk.Frame(inner)
        btn_row_g.pack(anchor='w', padx=px, pady=(0, 8))
        ttk.Button(btn_row_g, text=t("add_group"), style="Secondary.TButton",
                   command=self._atl_add_group).pack(side='left', padx=(0, 4))
        ttk.Button(btn_row_g, text=t("remove_last_group"), style="Secondary.TButton",
                   command=self._atl_remove_group).pack(side='left')

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Covariables ──────────────────────────────────────────────────────
        ttk.Label(inner, text=t("covariables_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("covariables_desc"),
                  style="Sub.TLabel", wraplength=520).pack(anchor='w', padx=px, pady=(0, 4))

        covar_frame = ttk.Frame(inner, style="Card.TFrame")
        covar_frame.pack(fill='x', padx=px, pady=(0, 4))
        self._atl_covar_frame = covar_frame
        self._atl_refresh_covar_names(covar_frame)

        btn_row_cov = ttk.Frame(inner)
        btn_row_cov.pack(anchor='w', padx=px, pady=(0, 8))
        ttk.Button(btn_row_cov, text=t("add_covar"), style="Secondary.TButton",
                   command=self._atl_add_covar).pack(side='left', padx=(0, 4))
        ttk.Button(btn_row_cov, text=t("remove_last_covar"), style="Secondary.TButton",
                   command=self._atl_remove_covar).pack(side='left')

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Mouse list ───────────────────────────────────────────────────────
        ttk.Label(inner, text=t("mice_csv_group"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        list_frame = ttk.Frame(inner, style="Card.TFrame")
        list_frame.pack(fill='x', padx=px, pady=(0, 4))
        self._atl_list_inner = list_frame

        if not self._atl_animals:
            self._atl_add_mouse()
            self._atl_add_mouse()
        else:
            self._atl_refresh_list()

        ttk.Button(inner, text=t("add_mouse"), style="Secondary.TButton",
                   command=self._atl_add_mouse).pack(anchor='w', padx=px, pady=(0, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Normalization ────────────────────────────────────────────────────
        ttk.Label(inner, text=t("normalization_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Radiobutton(inner,
                        text=t("norm_global_robust"),
                        variable=self._atl_normalization,
                        value='robust_global').pack(anchor='w', padx=px + 4, pady=1)
        ttk.Radiobutton(inner,
                        text=t("norm_permouse_robust"),
                        variable=self._atl_normalization,
                        value='robust').pack(anchor='w', padx=px + 4, pady=1)
        ttk.Radiobutton(inner,
                        text=t("norm_cl"),
                        variable=self._atl_normalization,
                        value='cl').pack(anchor='w', padx=px + 4, pady=(1, 6))
        ttk.Label(inner,
                  text=t("norm_global_robust_hint"),
                  style="Sub.TLabel", wraplength=500).pack(anchor='w', padx=px + 4, pady=(0, 6))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Atlas / K parameters ─────────────────────────────────────────────
        ttk.Label(inner, text=t("atl_params_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        def arow(label, widget_fn):
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=3)
            ttk.Label(f, text=label, width=34).pack(side='left')
            widget_fn(f)

        arow(t("k_min_bic"),
             lambda f: ttk.Spinbox(f, from_=2, to=10,
                                    textvariable=self._atl_k_min,
                                    width=6).pack(side='left'))
        arow(t("k_max_bic"),
             lambda f: ttk.Spinbox(f, from_=3, to=15,
                                    textvariable=self._atl_k_max,
                                    width=6).pack(side='left'))
        arow(t("k_override_blank"),
             lambda f: ttk.Entry(f, textvariable=self._atl_k_override,
                                  width=6).pack(side='left'))
        arow(t("gmm_restarts"),
             lambda f: ttk.Spinbox(f, from_=1, to=100,
                                    textvariable=self._atl_n_init,
                                    width=6).pack(side='left'))
        arow(t("bootstrap_runs"),
             lambda f: ttk.Spinbox(f, from_=0, to=200,
                                    textvariable=self._atl_n_bootstrap,
                                    width=6).pack(side='left'))
        arow(t("permutations_label"),
             lambda f: ttk.Spinbox(f, from_=99, to=9999,
                                    textvariable=self._atl_n_perm,
                                    width=7).pack(side='left'))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Output ───────────────────────────────────────────────────────────
        ttk.Label(inner, text=t("output_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f_name = ttk.Frame(inner)
        f_name.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_name, text=t("output_folder"), width=34).pack(side='left')
        ttk.Entry(f_name, textvariable=self._atl_output_name,
                  width=22).pack(side='left')

        f_dir = ttk.Frame(inner)
        f_dir.pack(fill='x', padx=px, pady=3)
        ttk.Entry(f_dir, textvariable=self._atl_output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f_dir, text=t("browse"), style="Secondary.TButton",
                   command=self._atl_select_output).pack(side='right', padx=(6, 0))
        ttk.Label(inner,
                  text=t("defaults_to_first_mouse"),
                  style="Sub.TLabel").pack(anchor='w', padx=px, pady=(2, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)

    # ── Atlas group helpers ───────────────────────────────────────────────────

    def _atl_refresh_group_names(self, frame=None):
        if frame is None:
            frame = getattr(self, '_atl_gnames_frame', None)
        if frame is None:
            return
        for w in frame.winfo_children():
            w.destroy()
        for gi, gvar in enumerate(self._atl_group_names):
            row = ttk.Frame(frame)
            row.pack(fill='x', padx=4, pady=2)
            ttk.Label(row, text=t("group_n_label", n=gi), width=10).pack(side='left', padx=(4, 2))
            ttk.Entry(row, textvariable=gvar, width=18).pack(side='left', padx=4)

    def _atl_add_group(self):
        if len(self._atl_group_names) >= 8:
            return
        self._atl_group_names.append(tk.StringVar(value=f"Group {len(self._atl_group_names)}"))
        self._atl_refresh_group_names()

    def _atl_remove_group(self):
        if len(self._atl_group_names) <= 2:
            return
        removed = len(self._atl_group_names) - 1
        self._atl_group_names.pop()
        for a in self._atl_animals:
            if a["group_var"].get() >= removed:
                a["group_var"].set(removed - 1)
        self._atl_refresh_group_names()
        self._atl_refresh_list()

    # ── Atlas covariable helpers ──────────────────────────────────────────────

    def _atl_refresh_covar_names(self, frame=None):
        if frame is None:
            frame = getattr(self, '_atl_covar_frame', None)
        if frame is None:
            return
        for w in frame.winfo_children():
            w.destroy()
        if not self._atl_covar_names:
            ttk.Label(frame, text=t("no_covar_defined"),
                      style="Sub.TLabel").pack(anchor='w', padx=6, pady=4)
            return
        for j, cvar in enumerate(self._atl_covar_names):
            row = ttk.Frame(frame)
            row.pack(fill='x', padx=4, pady=2)
            ttk.Label(row, text=t("cov_n_label", n=j + 1), width=8).pack(side='left', padx=(4, 2))
            ttk.Entry(row, textvariable=cvar, width=20).pack(side='left', padx=4)

    def _atl_add_covar(self):
        if len(self._atl_covar_names) >= 8:
            return
        default_names = ["Sexe", "Date_injection", "Diet",
                         "Cov4", "Cov5", "Cov6", "Cov7", "Cov8"]
        self._atl_covar_names.append(
            tk.StringVar(value=default_names[len(self._atl_covar_names)]))
        # Add an empty entry to every existing mouse
        for a in self._atl_animals:
            a.setdefault("covar_vars", []).append(tk.StringVar(value=""))
        self._atl_refresh_covar_names()
        self._atl_refresh_list()

    def _atl_remove_covar(self):
        if not self._atl_covar_names:
            return
        self._atl_covar_names.pop()
        for a in self._atl_animals:
            cvars = a.get("covar_vars", [])
            if cvars:
                cvars.pop()
        self._atl_refresh_covar_names()
        self._atl_refresh_list()

    # ── Atlas mouse card helpers ──────────────────────────────────────────────

    def _atl_add_mouse(self):
        n = len(self._atl_covar_names)
        self._atl_animals.append({
            "files"     : [],
            "group_var" : tk.IntVar(value=0),
            "covar_vars": [tk.StringVar(value="") for _ in range(n)],
        })
        self._atl_refresh_list()

    def _atl_remove_mouse(self, idx):
        if len(self._atl_animals) <= 1:
            return
        self._atl_animals.pop(idx)
        self._atl_refresh_list()

    def _atl_refresh_list(self):
        if self._atl_list_inner is None:
            return
        for w in self._atl_list_inner.winfo_children():
            w.destroy()
        for i, animal in enumerate(self._atl_animals):
            self._atl_build_card(i, animal)

    def _atl_build_card(self, i, animal):
        card = ttk.Frame(self._atl_list_inner, style="Card.TFrame")
        card.pack(fill='x', padx=4, pady=3)

        # ── Ligne 1 : identité, fichiers, groupe, suppression ────────────
        row1 = ttk.Frame(card)
        row1.pack(fill='x', padx=4, pady=(4, 1))

        ttk.Label(row1, text=t("mouse_n", n=i + 1),
                  font=THEME["font_bold"], width=9).pack(side='left', padx=(4, 2))

        n   = len(animal["files"])
        lbl = ttk.Label(row1,
                        text=t("n_files", n=n) if n else t("no_files"),
                        style="Sub.TLabel", width=12)
        lbl.pack(side='left', padx=6)
        animal["_lbl"] = lbl

        ttk.Button(row1, text=t("browse"), style="Secondary.TButton",
                   command=lambda a=animal: self._atl_browse(a)
                   ).pack(side='left', padx=2)
        ttk.Button(row1, text="📁", style="Secondary.TButton", width=3,
                   command=lambda a=animal: self._browse_animal_folder(a)
                   ).pack(side='left', padx=2)

        ttk.Label(row1, text=t("group_label"), style="Sub.TLabel").pack(side='left', padx=(10, 2))
        n_groups = len(self._atl_group_names)
        ttk.Spinbox(row1, from_=0, to=max(n_groups - 1, 1),
                    textvariable=animal["group_var"],
                    width=4).pack(side='left', padx=2)

        ttk.Button(row1, text="✕", style="Secondary.TButton", width=2,
                   state='normal' if len(self._atl_animals) > 1 else 'disabled',
                   command=lambda idx=i: self._atl_remove_mouse(idx)
                   ).pack(side='right', padx=4)

        # ── Ligne 2 : covariables (si définies) ──────────────────────────
        cvars = animal.setdefault("covar_vars", [])
        while len(cvars) < len(self._atl_covar_names):
            cvars.append(tk.StringVar(value=""))

        if self._atl_covar_names:
            row2 = ttk.Frame(card)
            row2.pack(fill='x', padx=4, pady=(0, 4))
            ttk.Label(row2, text=" ", width=9).pack(side='left')  # indentation
            for j, cname_var in enumerate(self._atl_covar_names):
                lbl_txt = cname_var.get() or f"Cov{j + 1}"
                ttk.Label(row2, text=f"{lbl_txt}:",
                          style="Sub.TLabel").pack(side='left', padx=(10, 1))
                ttk.Entry(row2, textvariable=cvars[j],
                          width=10).pack(side='left', padx=(0, 4))

    def _atl_browse(self, animal):
        files = filedialog.askopenfilenames(
            title=t("dlg_select_csv_mouse"),
            filetypes=[("CSV files", "*.csv")])
        if files:
            animal["files"] = list(files)
            if animal.get("_lbl"):
                animal["_lbl"].config(
                    text=t("n_files", n=len(files)),
                    foreground=THEME["text"])

    def _atl_select_output(self):
        path = filedialog.askdirectory(title=t("dlg_select_output"))
        if path:
            self._atl_output_dir.set(path)

    # ── Atlas run ─────────────────────────────────────────────────────────────

    def _run_atlas(self):
        mice_data, group_labels, covar_rows = [], [], []
        for animal in self._atl_animals:
            if not animal["files"]:
                continue
            name = os.path.basename(os.path.dirname(animal["files"][0]))
            mice_data.append({"name": name, "files": list(animal["files"])})
            group_labels.append(animal["group_var"].get())
            covar_rows.append([v.get().strip()
                               for v in animal.get("covar_vars", [])])

        if len(mice_data) < 2:
            messagebox.showerror(t("err_title"),
                                 "Select CSV files for at least 2 mice.")
            return

        if len(set(group_labels)) < 2:
            messagebox.showerror(t("err_title"),
                                 "Assign mice to at least 2 different groups.")
            return

        default_dir = os.path.dirname(os.path.dirname(mice_data[0]["files"][0]))
        if not self._confirm_output_dir(self._atl_output_dir, self._atl_output_name, default_dir):
            return

        out_dir = self._atl_output_dir.get() or default_dir
        out_dir = os.path.join(out_dir, self._atl_output_name.get() or "atlas_comparison")

        k_ov_str = self._atl_k_override.get().strip()
        k_ov     = int(k_ov_str) if k_ov_str.isdigit() else None

        group_names  = {gi: gv.get() for gi, gv in enumerate(self._atl_group_names)}
        covar_names  = [v.get().strip() for v in self._atl_covar_names]

        params = {
            "mice_data"    : mice_data,
            "group_labels" : group_labels,
            "group_names"  : group_names,
            "covar_names"  : covar_names,
            "covar_rows"   : covar_rows,
            "k_range"      : range(self._atl_k_min.get(), self._atl_k_max.get() + 1),
            "k_override"   : k_ov,
            "normalization": self._atl_normalization.get(),
            "n_init"       : self._atl_n_init.get(),
            "n_bootstrap"  : self._atl_n_bootstrap.get(),
            "n_perm"       : self._atl_n_perm.get(),
            "out_dir"      : out_dir,
        }

        self.btn_atl_run.config(state='disabled', text=t("running"))
        self._write_log(f"[Atlas] Starting — {len(mice_data)} mice, "
                        f"{len(set(group_labels))} groups")
        threading.Thread(target=self._run_atlas_pipeline,
                         args=(params,), daemon=True).start()

    def _run_atlas_pipeline(self, p):
        from atlas_pipeline import run_atlas_pipeline

        def log(msg, tag=None):
            self.queue.put({"type": "log", "text": f"[Atlas] {msg}", "tag": tag})

        try:
            covariates = _parse_atlas_covariates(
                p.get("covar_names", []),
                p.get("covar_rows",  []),
                log)
            result = run_atlas_pipeline(
                mice_data      = p["mice_data"],
                group_labels   = p["group_labels"],
                group_names    = p["group_names"],
                k_range        = p["k_range"],
                k_override     = p["k_override"],
                normalization  = p["normalization"],
                n_init         = p["n_init"],
                n_bootstrap    = p["n_bootstrap"],
                n_perm         = p["n_perm"],
                out_dir        = p["out_dir"],
                covariates     = covariates,
                covar_names    = p.get("covar_names", []),
                log            = log,
            )
            k     = result["atlas"].k
            pa    = result["permanova_aitch"]
            pm    = result["permanova_mmd"]
            dr    = result["dirichlet"]
            summary = (
                f"{result['n_mice'] if 'n_mice' in result else len(result['mouse_names'])} mice  |  "
                f"K={k}  |  "
                f"PERMANOVA-π  F={pa['F']:.2f} p={pa['p_value']:.3f}  |  "
                f"MMD  F={pm['F']:.2f} p={pm['p_value']:.3f}  |  "
                f"Dirichlet p={dr['p_value']:.3f}"
            )
            self.queue.put({
                "type"    : "atl_results",
                "pdf_path": result["pdf_path"],
                "summary" : summary,
            })
        except Exception as e:
            log(f"Error: {e}", "err")
            self.queue.put({"type": "log",
                            "text": f"[Atlas] {traceback.format_exc()}", "tag": "err"})
        finally:
            self.queue.put({"type": "atl_done"})

    def _show_atlas_results(self, summary, pdf_path):
        win = tk.Toplevel(self)
        win.title(t("wintitle_atl_done"))
        win.configure(bg=THEME["bg"])
        win.geometry("580x220")

        ttk.Label(win, text=t("atl_done_header"),
                  style="Section.TLabel").pack(pady=(18, 4))
        ttk.Label(win, text=summary,
                  style="Sub.TLabel", wraplength=540).pack(pady=4)
        ttk.Label(win, text=pdf_path,
                  style="Sub.TLabel", wraplength=540).pack(pady=4)

        btn_row = ttk.Frame(win)
        btn_row.pack(pady=12)
        ttk.Button(btn_row, text=t("open_pdf"),
                   style="Primary.TButton",
                   command=lambda: os.startfile(pdf_path)
                   ).pack(side='left', padx=8)
        ttk.Button(btn_row, text=t("open_folder"),
                   style="Secondary.TButton",
                   command=lambda: os.startfile(os.path.dirname(pdf_path))
                   ).pack(side='left', padx=8)
        ttk.Button(btn_row, text=t("close"),
                   style="Secondary.TButton",
                   command=win.destroy
                   ).pack(side='left', padx=8)

    # ------------------------------------------------------------------
    # Validation and launch
    # ------------------------------------------------------------------

    def _validate(self):
        if not self.csv_files:
            messagebox.showerror(t("err_title"), t("err_no_csv"))
            return False
        if self.k_min.get() >= self.k_max.get():
            messagebox.showerror(t("err_title"), t("err_k_range"))
            return False
        return True

    def _confirm_output_dir(self, dir_var, name_var, default_dir):
        """
        If output location or folder name is not set, show a warning dialog
        with the default path that will be used.
        Returns True to proceed, False to cancel so the user can set it.
        """
        dir_empty  = not dir_var.get().strip()
        name_empty = name_var is not None and not name_var.get().strip()

        if not dir_empty and not name_empty:
            return True

        name         = (name_var.get().strip() if name_var else "") or "output"
        default_path = os.path.join(default_dir, name)

        lines = []
        if dir_empty:
            lines.append("• No output location selected")
        if name_empty:
            lines.append("• No folder name specified")
        msg = "\n".join(lines)
        msg += f"\n\nResults will be saved to:\n{default_path}\n\nProceed with this default?"

        return messagebox.askyesno("Output not configured", msg, icon='warning')

    def _run(self):
        """
        Read ALL tk.Variable values here (main thread) before launching
        the secondary thread. Never read tk.Variable from the thread.
        """
        if not self._validate():
            return

        default_dir = os.path.dirname(self.csv_files[0])
        if not self._confirm_output_dir(self.output_dir, self.output_name, default_dir):
            return

        self.btn_run.config(state='disabled', text=t("running"))
        self.btn_napari.config(state='disabled')

        params = {
            "csv_files"          : self.csv_files.copy(),
            "nifti_path"         : self.nifti_path.get(),
            "method"             : self.method.get(),
            "k_min"              : self.k_min.get(),
            "k_max"              : self.k_max.get(),
            "k_override"         : self.k_override.get().strip(),
            "spatial_w"          : self.spatial_w.get(),
            "n_refs"             : self.n_refs.get(),
            "n_init_gmm"         : self.n_init_gmm.get(),
            "output_name"        : self.output_name.get(),
            "output_dir"         : self.output_dir.get(),
            "gmm_guard"          : self.gmm_guard.get(),
            "kmeans_guard"       : self.kmeans_guard.get(),
            "mrf_enabled"        : self.mrf_enabled.get(),
            "hierarchical_enabled": self.hierarchical_enabled.get(),
        }

        thread = threading.Thread(target=self._run_pipeline,
                                  args=(params,), daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Pipeline (secondary thread)
    # ------------------------------------------------------------------

    def _run_pipeline(self, p):
        def log(msg, tag=None):
            self.queue.put({"type": "log", "text": msg, "tag": tag})

        try:
            k_range    = range(p["k_min"], p["k_max"] + 1)
            method     = p["method"]
            k_ov_str   = p["k_override"]
            k_override = int(k_ov_str) if k_ov_str.isdigit() else None

            gmm_guard    = GMM_SILHOUETTE_GUARD if p["gmm_guard"] else None
            kmeans_guard = bool(p["kmeans_guard"])

            if p["output_dir"]:
                out_dir = os.path.join(p["output_dir"], p["output_name"])
            else:
                out_dir = os.path.join(
                    os.path.dirname(p["csv_files"][0]), p["output_name"])
            os.makedirs(out_dir, exist_ok=True)
            log(t("log_output_dir", path=out_dir))

            parameters, df_all = load_data(p["csv_files"])
            parameters, dti_mode = resolve_dti_parameters(parameters)
            log(t("log_dti_mode", mode=dti_mode, params=parameters))
            df_scaled, features_scaled, scaling_info = robust_scale(df_all, parameters)
            plot_correlation(df_scaled, parameters, out_dir)

            # ── Step 6 (option): Hierarchical k=2 segmentation ───────
            use_hier   = p.get("hierarchical_enabled", HIERARCHICAL_ENABLED)
            use_mrf    = p.get("mrf_enabled", MRF_ENABLED)
            hier_meta  = None
            dtopo_meta = None   # set below if necrosis guard passes
            if use_hier:
                log("  [Hierarchical] Running k=2 guard + sub-segmentation...")
                labels_hier, hier_meta = hierarchical_segment(
                    features_scaled, df_all, parameters,
                    n_init_gmm=p["n_init_gmm"],
                    method=method,
                    spatial_weight=p["spatial_w"],
                    n_refs=p["n_refs"],
                    n_jobs=N_JOBS_GAP,
                    gmm_silhouette_guard=gmm_guard,
                    kmeans_gap_guard=kmeans_guard)
                if hier_meta.get('hierarchical_valid'):
                    labels     = labels_hier
                    best_k     = hier_meta['best_k']
                    extra_info = {}
                    d_topo_hier, nec_id_hier = compute_necrotic_dtopo(
                        df_all, labels_hier, hier_meta)
                    dtopo_meta = {
                        'has_necrosis'    : nec_id_hier is not None,
                        'd_topo'          : d_topo_hier,
                        'necrotic_cluster': nec_id_hier,
                    }
                    log(f"  [Hierarchical] Valid — k={best_k} "
                        f"(core: {hier_meta['core_cluster_ids']}, "
                        f"periph: {hier_meta['periphery_cluster_ids']}, "
                        f"criterion: {hier_meta.get('l1_criterion','?')})", tag="ok")
                else:
                    hier_meta = None
                    log("  [Hierarchical] Guard failed — normal pipeline")

            if not use_hier or hier_meta is None:
                # ── Step 3: d_topo feature enrichment ────────────────
                features_clust, dtopo_meta = prepare_features(
                    features_scaled, df_all, parameters, n_init_gmm=p["n_init_gmm"])
                if dtopo_meta['has_necrosis']:
                    log(f"  [d_topo] Feature added "
                        f"(necrotic cluster {dtopo_meta['necrotic_cluster']})", tag="ok")

                # ── k selection and clustering ────────────────────────
                if k_override:
                    best_k = k_override
                    log(t("log_optimal_k", k=best_k))
                    labels, extra_info = run_clustering(
                        features_clust, df_all, best_k,
                        method=method,
                        spatial_weight=p["spatial_w"],
                        n_init_gmm=p["n_init_gmm"])
                else:
                    best_k, gmm_cache = select_optimal_k(
                        features_clust,
                        method=method,
                        df_all=df_all,
                        k_range=k_range,
                        n_refs=p["n_refs"],
                        n_jobs=N_JOBS_GAP,
                        spatial_weight=p["spatial_w"],
                        n_init_gmm=p["n_init_gmm"],
                        gmm_silhouette_guard=gmm_guard,
                        kmeans_gap_guard=kmeans_guard,
                        output_dir=out_dir,
                    )
                    log(t("log_optimal_k", k=best_k))
                    labels, extra_info, best_k = run_clustering_with_size_guard(
                        features_clust, features_scaled, df_all, best_k,
                        method=method,
                        spatial_weight=p["spatial_w"],
                        n_init_gmm=p["n_init_gmm"],
                        parameters=parameters,
                        dtopo_meta=dtopo_meta,
                        k_min=p["k_min"],
                        gmm_cache=gmm_cache)

            df_scaled['Habitat'] = labels

            if method == 'gmm' and 'proba' in extra_info:
                plot_gmm_probabilities(
                    labels, df_scaled, extra_info['proba'],
                    HABITAT_COLORS_HEX[:best_k], best_k, out_dir)

            # Centroids from ORIGINAL features (not d_topo-augmented)
            centroids = show_habitat_map(
                labels, df_scaled, features_scaled,
                parameters, best_k, out_dir)
            inject_dtopo_centroids(centroids, labels, dtopo_meta)
            save_centroids_json(centroids, out_dir)

            raw_centroids = {
                c: {param: float(df_all[param].values[labels == c].mean())
                    for param in parameters}
                for c in range(best_k)
            }
            features_by_cluster = {c: features_scaled[labels == c]
                                   for c in range(best_k)}
            # Zone-constrained labeling when hierarchical split was used
            zone_map = None
            if hier_meta and hier_meta.get('hierarchical_valid'):
                zone_map = {}
                for _c in hier_meta['core_cluster_ids']:
                    zone_map[_c] = 'core'
                for _c in hier_meta['periphery_cluster_ids']:
                    zone_map[_c] = 'periphery'
            three_pass = label_clusters_ws(centroids, parameters,
                                            zone_map=zone_map)

            # ── Step 5 (option): MRF light boundary correction ───────
            proba = extra_info.get('proba')
            if use_mrf and proba is not None:
                log("  [MRF] Light boundary correction...")
                labels = mrf_boundary_smooth(labels, proba, df_all, three_pass)
                df_scaled['Habitat'] = labels

            # ── Step 4: Spatial coherence analysis + reporting ───────
            confidence_arr, ambiguous_arr, coh_report = analyze_spatial_coherence(
                labels, df_all, proba, best_k, out_dir)
            df_scaled['label_confidence'] = confidence_arr
            df_scaled['label_ambiguous']  = ambiguous_arr.astype(int)

            n_micro = sum(r.get('micro_components', 0) for r in coh_report.values())
            if n_micro > 0:
                log(f"  [Spatial] {n_micro} micro-component(s) flagged "
                    "(see spatial_coherence.csv)")

            cl_stats = load_cl_stats(p["csv_files"], parameters)
            if cl_stats:
                log(t("log_cl_loaded", n=len(cl_stats), total=len(parameters)), tag="ok")
            else:
                log(t("log_cl_missing"))

            colors = HABITAT_COLORS_HEX[:best_k]

            label_event  = threading.Event()
            label_result = {}

            self.queue.put({
                "type"              : "open_dialog",
                "centroids"         : centroids,
                "parameters"        : parameters,
                "best_k"            : best_k,
                "colors"            : colors,
                "event"             : label_event,
                "result"            : label_result,
                "out_dir"           : out_dir,
                "cl_stats"          : cl_stats,
                "raw_centroids"     : raw_centroids,
                "scaling_info"      : scaling_info,
                "dti_mode"          : dti_mode,
                "three_pass_results": three_pass,
                "hier_meta"         : hier_meta,
            })

            label_event.wait()
            habitat_labels = label_result
            log(t("log_labels_received"))

            plot_habitat_barchart(df_scaled, parameters, habitat_labels, out_dir)
            plot_heatmap_profiles(df_scaled, parameters, habitat_labels, best_k, out_dir)
            plot_radar_profiles(df_scaled, parameters, habitat_labels, out_dir)
            export_results(df_scaled, parameters, habitat_labels, out_dir)
            save_run_meta(scaling_info, cl_stats, out_dir)
            generate_report_pdf(out_dir, df_scaled, parameters,
                                habitat_labels, best_k, method)

            log(t("log_analysis_complete"), tag="ok")

            nifti = p["nifti_path"]
            if nifti:
                mask_path   = save_habitat_mask(nifti, df_scaled, df_all, out_dir)
                labels_path = os.path.join(out_dir, 'habitat_labels.json')
                self.queue.put({
                    "type"       : "napari_ready",
                    "nifti"      : nifti,
                    "mask_path"  : mask_path,
                    "labels_path": labels_path,
                })
            else:
                self.queue.put({"type": "done"})

        except Exception as e:
            self.queue.put({"type": "error",
                            "text": f"\n[ERROR] {e}\n{traceback.format_exc()}"})


    # ------------------------------------------------------------------
    # Discovery tab — UI
    # ------------------------------------------------------------------

    def _build_tab_discovery(self, parent):
        px  = THEME["pad_x"]
        pad = {"padx": px, "pady": THEME["pad_y"]}

        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        ttk.Label(inner,
                  text=t("disc_title"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("disc_desc"),
                  style="Sub.TLabel",
                  wraplength=520).pack(anchor='w', padx=px, pady=(0, 8))

        # ── Animal list ───────────────────────────────────────────────
        ttk.Label(inner, text=t("mice_per_animal"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        list_frame = ttk.Frame(inner, style="Card.TFrame")
        list_frame.pack(fill='x', padx=px, pady=(0, 4))
        self._disc_list_inner = list_frame

        if not self._disc_animals:
            self._disc_add_mouse()
            self._disc_add_mouse()
        else:
            self._disc_refresh_list()

        ttk.Button(inner, text=t("add_mouse"), style="Secondary.TButton",
                   command=self._disc_add_mouse).pack(anchor='w', padx=px, pady=(0, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Segmentation settings ─────────────────────────────────────
        ttk.Label(inner, text=t("segmentation_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        for val, key in [("gmm", "gmm_desc"), ("kmeans", "kmeans_desc")]:
            ttk.Radiobutton(inner, text=t(key),
                            variable=self._disc_method,
                            value=val).pack(anchor='w', padx=px+4, pady=1)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)
        ttk.Label(inner, text=t("parameters_section"), style="Sub.TLabel").pack(anchor='w', padx=px)

        def drow(label, widget_fn):
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=3)
            ttk.Label(f, text=label, width=30).pack(side='left')
            widget_fn(f)

        drow(t("k_max_per_mouse"),
             lambda f: ttk.Spinbox(f, from_=2, to=10,
                                   textvariable=self._disc_k_max,
                                   width=6).pack(side='left'))
        drow(t("gmm_n_init"),
             lambda f: ttk.Spinbox(f, from_=1, to=50,
                                   textvariable=self._disc_n_init,
                                   width=6).pack(side='left'))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Meta-clustering settings ───────────────────────────────────
        ttk.Label(inner, text=t("meta_clustering_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        drow(t("force_meta_k"),
             lambda f: ttk.Spinbox(f, from_=0, to=15,
                                   textvariable=self._disc_meta_k,
                                   width=6).pack(side='left'))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Output ────────────────────────────────────────────────────
        ttk.Label(inner, text=t("output_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f_name = ttk.Frame(inner)
        f_name.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_name, text=t("folder_name"), width=30).pack(side='left')
        ttk.Entry(f_name, textvariable=self._disc_output_name,
                  width=22).pack(side='left')

        f_dir = ttk.Frame(inner)
        f_dir.pack(fill='x', padx=px, pady=3)
        ttk.Entry(f_dir, textvariable=self._disc_output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f_dir, text=t("browse"), style="Secondary.TButton",
                   command=self._disc_select_output).pack(side='right', padx=(6, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Section : Comparaison de groupes ──────────────────────────
        ttk.Label(inner, text=t("group_compare_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("group_compare_desc"),
                  style="Sub.TLabel", wraplength=510).pack(anchor='w', padx=px, pady=(0, 4))
        ttk.Checkbutton(inner,
                        text=t("enable_group_compare"),
                        variable=self._disc_compare_enabled,
                        command=self._disc_toggle_compare
                        ).pack(anchor='w', padx=px + 4, pady=(0, 4))

        disc_gf = ttk.Frame(inner, style="Card.TFrame")
        disc_gf.pack(fill='x', padx=px, pady=(0, 8))
        self._disc_gnames_frame = disc_gf
        self._disc_refresh_gnames()

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)

    # ── Group compare helpers (Discovery) ─────────────────────────────

    def _disc_toggle_compare(self):
        self._disc_refresh_gnames()
        self._disc_refresh_list()

    def _disc_refresh_gnames(self):
        f = self._disc_gnames_frame
        if f is None:
            return
        for w in f.winfo_children():
            w.destroy()
        if not self._disc_compare_enabled.get():
            ttk.Label(f, text=t("compare_disabled"),
                      style="Sub.TLabel").pack(anchor='w', padx=6, pady=4)
            return
        for gi, gvar in enumerate(self._disc_group_names):
            row = ttk.Frame(f)
            row.pack(fill='x', padx=4, pady=2)
            ttk.Label(row, text=t("group_n_label", n=gi), width=10).pack(side='left', padx=(4, 2))
            ttk.Entry(row, textvariable=gvar, width=18).pack(side='left', padx=4)

    # ── Animal card helpers ───────────────────────────────────────────

    def _disc_add_mouse(self):
        self._disc_animals.append({"files": [], "group_var": tk.IntVar(value=0)})
        self._disc_refresh_list()

    def _disc_remove_mouse(self, idx):
        if len(self._disc_animals) <= 1:
            return
        self._disc_animals.pop(idx)
        self._disc_refresh_list()

    def _disc_refresh_list(self):
        if self._disc_list_inner is None:
            return
        for w in self._disc_list_inner.winfo_children():
            w.destroy()
        for i, animal in enumerate(self._disc_animals):
            self._disc_build_card(i, animal)

    def _disc_build_card(self, i, animal):
        card = ttk.Frame(self._disc_list_inner)
        card.pack(fill='x', padx=4, pady=2)

        ttk.Label(card, text=t("mouse_n", n=i+1),
                  font=THEME["font_bold"], width=9).pack(side='left', padx=(4, 2))

        n   = len(animal["files"])
        lbl = ttk.Label(card,
                        text=t("n_files", n=n) if n else t("no_files"),
                        style="Sub.TLabel", width=12)
        lbl.pack(side='left', padx=6)
        animal["_lbl"] = lbl

        ttk.Button(card, text=t("browse"), style="Secondary.TButton",
                   command=lambda a=animal: self._disc_browse(a)
                   ).pack(side='left', padx=2)
        ttk.Button(card, text="📁", style="Secondary.TButton", width=3,
                   command=lambda a=animal: self._disc_browse_folder(a)
                   ).pack(side='left', padx=2)

        if self._disc_compare_enabled.get():
            ttk.Label(card, text=t("group_label"), style="Sub.TLabel").pack(side='left', padx=(10, 2))
            gvar = animal.setdefault("group_var", tk.IntVar(value=0))
            ttk.Spinbox(card, from_=0, to=max(len(self._disc_group_names) - 1, 1),
                        textvariable=gvar, width=4).pack(side='left', padx=2)

        ttk.Button(card, text="✕", style="Secondary.TButton", width=2,
                   state='normal' if len(self._disc_animals) > 1 else 'disabled',
                   command=lambda idx=i: self._disc_remove_mouse(idx)
                   ).pack(side='right', padx=4)

    def _disc_browse(self, animal):
        files = filedialog.askopenfilenames(
            title=t("dlg_select_csv_mouse"),
            filetypes=[("CSV files", "*.csv")])
        if files:
            animal["files"] = list(files)
            if animal.get("_lbl"):
                animal["_lbl"].config(
                    text=t("n_files", n=len(files)),
                    foreground=THEME["text"])

    def _browse_animal_folder(self, animal):
        """Shared folder-browse for all per-animal cards: picks all .csv files directly inside the chosen folder."""
        folder = filedialog.askdirectory(title=t("dlg_select_folder"))
        if not folder:
            return
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith('.csv')
        )
        if not files:
            messagebox.showwarning(t("warn_empty_folder"), t("warn_no_csv_in_folder"))
            return
        animal["files"] = files
        if animal.get("_lbl"):
            animal["_lbl"].config(
                text=t("n_files", n=len(files)),
                foreground=THEME["text"])

    # kept as alias so existing references in _disc_build_card still work
    _disc_browse_folder = _browse_animal_folder

    def _disc_select_output(self):
        path = filedialog.askdirectory(title=t("dlg_select_output"))
        if path:
            self._disc_output_dir.set(path)

    # ------------------------------------------------------------------
    # Discovery tab — pipeline
    # ------------------------------------------------------------------

    def _run_discovery(self):
        if len(self._disc_animals) < 2:
            messagebox.showerror(t("err_title"), t("err_min_mice"))
            return
        if any(not a["files"] for a in self._disc_animals):
            messagebox.showerror(t("err_title"), t("err_mouse_no_csv"))
            return

        default_dir = os.path.dirname(self._disc_animals[0]["files"][0])
        if not self._confirm_output_dir(self._disc_output_dir, self._disc_output_name, default_dir):
            return

        if self._disc_output_dir.get():
            out_root = os.path.join(self._disc_output_dir.get(),
                                    self._disc_output_name.get())
        else:
            out_root = os.path.join(default_dir, self._disc_output_name.get())

        params = {
            "animals"   : [{"files": list(a["files"])} for a in self._disc_animals],
            "method"    : self._disc_method.get(),
            "k_max"     : self._disc_k_max.get(),
            "n_init"    : self._disc_n_init.get(),
            "meta_k"    : int(self._disc_meta_k.get()) or None,
            "out_root"  : out_root,
        }

        self.btn_disc_run.config(state='disabled', text=t("running"))
        self._write_log(f"[Discovery] Starting — {len(params['animals'])} mice")
        threading.Thread(target=self._run_discovery_pipeline,
                         args=(params,), daemon=True).start()

    def _run_discovery_pipeline(self, p):
        import numpy as np

        def log(msg, tag=None):
            self.queue.put({"type": "log", "text": msg, "tag": tag})

        try:
            out_root = p["out_root"]
            os.makedirs(out_root, exist_ok=True)
            method   = p["method"]
            k_range  = range(2, p["k_max"] + 1)
            n_init   = p["n_init"]

            mice_data  = []
            _used_disc_names: dict = {}

            def _disc_animal_name(files, fallback):
                """Derive animal name from parent folder of first input file."""
                _GENERIC = {"v2","v3","v4","v5","gmm","kmeans","srsc",
                            "results","output","out","run","csv","nifti","data","raw"}
                if not files:
                    return fallback
                parts = os.path.abspath(files[0]).replace("\\", "/").split("/")
                parent = parts[-2] if len(parts) >= 2 else fallback
                if parent.lower() in _GENERIC and len(parts) >= 3:
                    return parts[-3]
                return parent if parent else fallback

            for i, animal in enumerate(p["animals"]):
                raw_name = _disc_animal_name(animal["files"], f"Mouse_{i+1}")
                # Deduplicate names so folder creation never collides
                if raw_name in _used_disc_names:
                    _used_disc_names[raw_name] += 1
                    name = f"{raw_name}_{_used_disc_names[raw_name]}"
                else:
                    _used_disc_names[raw_name] = 1
                    name = raw_name
                out_dir = os.path.join(out_root, name)
                os.makedirs(out_dir, exist_ok=True)
                log(f"[{name}] Loading data…")

                parameters, df_all = load_data(animal["files"])
                parameters, dti_mode = resolve_dti_parameters(parameters)
                df_scaled, features_scaled, scaling_info = robust_scale(df_all, parameters)

                features_clust, dtopo_meta = prepare_features(
                    features_scaled, df_all, parameters, n_init_gmm=n_init)

                best_k, gmm_cache = select_optimal_k(
                    features_clust, method=method, df_all=df_all,
                    k_range=k_range, n_refs=N_REFS_GAP, n_jobs=N_JOBS_GAP,
                    spatial_weight=SPATIAL_WEIGHT, n_init_gmm=n_init,
                    gmm_silhouette_guard=None, kmeans_gap_guard=False,
                    output_dir=out_dir)
                log(f"[{name}] Optimal K = {best_k}")

                labels, extra_info, best_k = run_clustering_with_size_guard(
                    features_clust, features_scaled, df_all, best_k,
                    method=method, spatial_weight=SPATIAL_WEIGHT,
                    n_init_gmm=n_init, parameters=parameters,
                    dtopo_meta=dtopo_meta, k_min=2,
                    gmm_cache=gmm_cache)

                df_scaled['Habitat'] = labels

                # Compute centroids (robust-scaled space)
                centroids_scaled = {
                    int(c): {param: float(features_scaled[labels == c, pi].mean())
                             for pi, param in enumerate(parameters)}
                    for c in range(best_k)
                }

                # Inject d_topo so Hypoxic/vascular and Infiltrative rules fire correctly
                inject_dtopo_centroids(centroids_scaled, labels, dtopo_meta)

                # Auto-suggest labels for display (not used in grouping)
                three_pass = label_clusters_ws(centroids_scaled, parameters)

                # Export minimal per-mouse result
                hab_labels_auto = {
                    c: (three_pass.get(c, {}).get("label") or f"Cluster {c+1}")
                    for c in range(best_k)
                }
                export_results(df_scaled, parameters, hab_labels_auto, out_dir)
                save_run_meta(scaling_info, None, out_dir)
                log(f"[{name}] Done → {best_k} clusters", tag="ok")

                # Build mouse_data dict for meta-clustering
                total_vox = max(len(labels), 1)
                clusters  = []
                for c in range(best_k):
                    mask  = labels == c
                    n_vox = int(mask.sum())
                    if n_vox < _DISC_SIZE_FLOOR:
                        continue
                    clusters.append({
                        "cluster_id": c,
                        "user_label": hab_labels_auto.get(c, f"Cluster {c+1}"),
                        "n_voxels"  : n_vox,
                        "proportion": n_vox / total_vox,
                        "centroid"  : centroids_scaled[c],
                    })

                if clusters:
                    mice_data.append({
                        "name"    : name,
                        "csv_path": os.path.join(out_dir, "habitats_result.csv"),
                        "clusters": clusters,
                    })

            if len(mice_data) < 2:
                raise ValueError("Too few valid mice after segmentation.")

            # ── Meta-clustering ─────────────────────────────────────
            log("[Discovery] Aligning parameters across mice…")
            common_params = find_common_parameters(mice_data)
            log(f"[Discovery] Common parameters: {', '.join(common_params)}")

            X, row_meta = build_centroid_matrix(mice_data, common_params)
            log(f"[Discovery] Centroid pool: {len(X)} clusters")

            meta_k_override = p.get("meta_k")
            if meta_k_override and meta_k_override >= 2:
                chosen_k    = min(meta_k_override, len(X) - 1)
                meta_labels = run_meta_clustering(X, chosen_k, 30, 35, method)
                try:
                    from sklearn.metrics import silhouette_score as _sil
                    sil = float(_sil(X, meta_labels))
                except Exception:
                    sil = -1.0
                k_scores = {chosen_k: sil}
                log(f"[Discovery] Meta-K forced to {chosen_k}")
            else:
                log("[Discovery] Searching optimal meta-K…")
                safe_kmax  = min(10, len(X) - 1)
                chosen_k, k_scores = find_optimal_meta_k(
                    X, range(2, safe_kmax + 1), 30, 35, method,
                    lambda msg: log(f"[Discovery]   {msg}"))
                meta_labels = run_meta_clustering(X, chosen_k, 30, 35, method)
                log(f"[Discovery] Optimal meta-K = {chosen_k}  "
                    f"(silhouette={k_scores[chosen_k]:.3f})")

            meta_stats = compute_meta_stats(
                X, meta_labels, row_meta, common_params, chosen_k, len(mice_data))

            # ── Figures ─────────────────────────────────────────────
            log("[Discovery] Generating figures…")
            n_mice_disc = len(mice_data)
            fig_mouse_index(mice_data, out_root)
            fig_summary_table(meta_stats, common_params, n_mice_disc, out_root)
            fig_k_selection(k_scores, chosen_k, out_root)
            fig_pca_scatter(X, meta_labels, meta_stats, out_root)
            fig_profiles_heatmap(meta_stats, common_params, out_root)
            fig_meta_profiles_bar(meta_stats, common_params, n_mice_disc, out_root)
            fig_radar_per_mh(meta_stats, common_params, n_mice_disc,
                             X, meta_labels, row_meta, out_root)
            fig_radar_combined(meta_stats, common_params, out_root)
            fig_crosstab(row_meta, meta_labels, meta_stats, out_root)
            seg_figs = fig_segmentation_maps(
                mice_data, row_meta, meta_labels, meta_stats, out_root)
            log(f"[Discovery] Segmentation maps: {len(seg_figs)} mouse map(s)")

            pdf_path = _disc_generate_pdf(
                meta_stats, common_params, len(mice_data),
                chosen_k, method, k_scores, out_root,
                seg_figs=seg_figs)
            log(f"[Discovery] PDF → {pdf_path}", tag="ok")

            # ── Explanation text ─────────────────────────────────────
            text = self._generate_discovery_text(
                X, meta_labels, meta_stats, common_params,
                len(mice_data), chosen_k, k_scores, method,
                mice_data=mice_data)

            text_path = os.path.join(out_root, "discovery_explanation.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            log(f"[Discovery] Explanation → {text_path}", tag="ok")
            log("[Discovery] Complete.", tag="ok")

            self.queue.put({"type": "disc_results",
                            "text": text, "pdf_path": pdf_path})

        except Exception as e:
            self.queue.put({"type": "error",
                            "text": f"\n[ERROR Discovery] {e}\n{traceback.format_exc()}"})
        finally:
            self.queue.put({"type": "disc_done"})

    # ------------------------------------------------------------------
    # Discovery — explanation text generator
    # ------------------------------------------------------------------

    def _generate_discovery_text(self, X, meta_labels, meta_stats,
                                  common_params, n_mice, chosen_k,
                                  k_scores, method, mice_data=None):
        import numpy as np

        def _magnitude_label(v):
            av = abs(v)
            if av >= 1.5:
                return "strongly elevated" if v > 0 else "strongly decreased"
            if av >= 0.7:
                return "mildly elevated"   if v > 0 else "mildly decreased"
            return "near-normal"

        def _stars(d):
            if d >= 1.5: return "★★★ (strong)"
            if d >= 0.8: return "★★  (moderate)"
            if d >= 0.4: return "★   (weak)"
            return    "—   (not discriminant)"

        def _cohen_d(a, b):
            if len(a) < 2 or len(b) < 2:
                return 0.0
            pa = np.std(a, ddof=1) ** 2
            pb = np.std(b, ddof=1) ** 2
            pooled = np.sqrt((pa + pb) / 2.0 + 1e-10)
            return float(abs(np.mean(a) - np.mean(b)) / pooled)

        # Build mouse-number map (insertion order = selection order in UI)
        mouse_num: dict = {}
        if mice_data:
            mouse_num = {m["name"]: i + 1 for i, m in enumerate(mice_data)}
        else:
            # Fallback: derive order from meta_stats members
            for s in meta_stats:
                for mb in s.get("members", []):
                    if mb["mouse"] not in mouse_num:
                        mouse_num[mb["mouse"]] = len(mouse_num) + 1

        def _mlabel(name: str) -> str:
            num = mouse_num.get(name)
            return f"M{num}" if num else name

        chosen_sil = k_scores.get(chosen_k, 0.0) if k_scores else 0.0
        max_sil_k  = max(k_scores, key=k_scores.get) if k_scores else chosen_k
        max_sil    = k_scores.get(max_sil_k, 0.0)   if k_scores else 0.0
        elbow_note = (
            f"  Max sil. K={max_sil_k}: {max_sil:.3f} (not chosen — outlier trap)"
            if max_sil_k != chosen_k else ""
        )

        # Mouse index section
        mouse_index_lines = [
            "═" * 62,
            "  MOUSE INDEX",
            "═" * 62,
        ]
        for name, num in sorted(mouse_num.items(), key=lambda kv: kv[1]):
            mouse_index_lines.append(f"  M{num:<3} : {name}")
        mouse_index_lines.append("")

        lines = mouse_index_lines + [
            "═" * 62,
            "  MULTI-MOUSE EMPIRICAL HABITAT DISCOVERY",
            "═" * 62,
            f"  Mice          : {n_mice}",
            f"  Method        : {method.upper()}",
            f"  Meta-habitats : {chosen_k}",
            f"  K selection   : silhouette elbow criterion",
            f"  Silhouette at elbow K={chosen_k}: {chosen_sil:.3f}",
        ] + ([elbow_note] if elbow_note else []) + [
            f"  Common parameters: {', '.join(common_params)}",
            "",
            "  No prior biological label was used during grouping.",
            "  User labels appear in [COMPOSITION] for validation only.",
            "",
        ]

        for s in meta_stats:
            lbl  = s["display_label"]
            mask = meta_labels == s["meta_label"]
            X_in = X[mask]
            X_out= X[~mask]
            members = s["members"]

            # Intra / inter distances
            if len(X_in) >= 2:
                pairs_in  = [(X_in[a] - X_in[b])
                             for a in range(len(X_in))
                             for b in range(a+1, len(X_in))]
                intra_d   = float(np.mean([np.linalg.norm(d) for d in pairs_in]))
            else:
                intra_d = 0.0

            if len(X_out) and len(X_in):
                pairs_cross = [(a - b) for a in X_in for b in X_out]
                inter_d     = float(np.mean([np.linalg.norm(d) for d in pairs_cross]))
            else:
                inter_d = 1.0

            ratio = intra_d / inter_d if inter_d > 1e-8 else 0.0
            sep_verdict = "✓ well-separated" if ratio < 0.5 else "⚠ overlapping"

            # Cohen's d per parameter
            cohen = []
            for pi, param in enumerate(common_params):
                d = _cohen_d(X_in[:, pi], X_out[:, pi]) if len(X_out) else 0.0
                cohen.append((param, d))
            cohen.sort(key=lambda x: -x[1])

            lines += [
                "═" * 62,
                f"  META-HABITAT  {lbl}",
                "═" * 62,
                f"  Present in {s['n_mice']}/{n_mice} mice  •  "
                f"{s['n_clusters']} clusters  •  "
                f"prevalence {s['presence_rate']*100:.0f}%",
                "",
                "  MRI SIGNATURE  (mean ± std, robust-scaled):",
            ]
            for pi, param in enumerate(common_params):
                mean_v = float(s["mean_centroid"][pi])
                std_v  = float(s["std_centroid"][pi])
                tag    = _magnitude_label(mean_v)
                char   = "◄" if abs(mean_v) >= 0.7 else " "
                lines.append(
                    f"    {param:<12} {mean_v:+.2f} ± {std_v:.2f}"
                    f"   {char} {tag}")
            lines += [
                "",
                "  WHY THESE CLUSTERS WERE GROUPED TOGETHER:",
                f"    Intra-{lbl} mean distance  : {intra_d:.3f} IQR units",
                f"    vs. all other clusters    : {inter_d:.3f} IQR units",
                f"    Separation ratio          : {ratio:.2f}  {sep_verdict}",
                "    → Clusters share the same multi-parameter direction",
                "      in robust-scaled feature space.",
                "",
                "  DISCRIMINANT FEATURES vs. other meta-habitats:",
            ]
            for param, d in cohen:
                lines.append(f"    {param:<12}  Cohen's d = {d:.2f}   {_stars(d)}")

            lines += ["", "  CLUSTER COMPOSITION  (auto-suggest label | volume %):", ]
            for m in sorted(members, key=lambda x: mouse_num.get(x["mouse"], 999)):
                pct   = m["proportion"] * 100
                mlbl  = _mlabel(m["mouse"])
                lines.append(
                    f"    {mlbl:<5}  cluster {m['cluster_id']+1}"
                    f"  {pct:4.1f}%  [{m['user_label']}]")
            lines.append("")

        lines += [
            "═" * 62,
            "  NOTE ON INTERPRETATION",
            "═" * 62,
            "  Values in IQR units (0 = tumour median for each mouse).",
            "  'Strongly elevated/decreased' means |value| > 1.5 IQR.",
            "  Cohen's d > 1.5 = strong separator between groups.",
            "  Separation ratio < 0.5 = compact, well-separated group.",
            "  [user_label] is auto-suggested by the pipeline rules",
            "  and shown here only for biological sanity-check.",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Joint clustering tab — UI
    # ------------------------------------------------------------------

    def _build_tab_joint(self, parent):
        px  = THEME["pad_x"]
        pad = {"padx": px, "pady": THEME["pad_y"]}

        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        ttk.Label(inner,
                  text=t("jt_title"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("jt_desc"),
                  style="Sub.TLabel",
                  wraplength=520).pack(anchor='w', padx=px, pady=(0, 10))

        # ── Mouse list ───────────────────────────────────────────────
        ttk.Label(inner, text=t("mice_raw_csv"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        list_frame = ttk.Frame(inner, style="Card.TFrame")
        list_frame.pack(fill='x', padx=px, pady=(0, 4))
        self._jt_list_inner = list_frame

        if not self._jt_animals:
            self._jt_add_mouse()
            self._jt_add_mouse()
        else:
            self._jt_refresh_list()

        ttk.Button(inner, text=t("add_mouse"), style="Secondary.TButton",
                   command=self._jt_add_mouse).pack(anchor='w', padx=px, pady=(0, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Normalization ─────────────────────────────────────────────
        ttk.Label(inner, text=t("normalization_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        ttk.Radiobutton(
            inner,
            text=t("norm_robust_global_jt_radio"),
            variable=self._jt_normalization,
            value='robust_global'
        ).pack(anchor='w', padx=px + 4, pady=1)
        ttk.Label(inner,
                  text=t("norm_robust_global_jt_hint"),
                  style="Sub.TLabel", wraplength=500
                  ).pack(anchor='w', padx=px + 22, pady=(0, 6))

        ttk.Radiobutton(
            inner,
            text=t("norm_robust_jt_radio"),
            variable=self._jt_normalization,
            value='robust'
        ).pack(anchor='w', padx=px + 4, pady=1)
        ttk.Label(inner,
                  text=t("norm_robust_jt_hint"),
                  style="Sub.TLabel", wraplength=500
                  ).pack(anchor='w', padx=px + 22, pady=(0, 6))

        ttk.Radiobutton(
            inner,
            text=t("norm_cl_hybrid_jt_radio"),
            variable=self._jt_normalization,
            value='cl_hybrid'
        ).pack(anchor='w', padx=px + 4, pady=1)
        ttk.Label(inner,
                  text=t("norm_cl_hybrid_jt_hint"),
                  style="Sub.TLabel", wraplength=500
                  ).pack(anchor='w', padx=px + 22, pady=(0, 6))

        ttk.Radiobutton(
            inner,
            text=t("norm_cl_jt_radio"),
            variable=self._jt_normalization,
            value='cl'
        ).pack(anchor='w', padx=px + 4, pady=1)
        ttk.Label(inner,
                  text=t("norm_cl_jt_hint"),
                  style="Sub.TLabel", wraplength=500
                  ).pack(anchor='w', padx=px + 22, pady=(0, 6))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Clustering method ────────────────────────────────────────
        ttk.Label(inner, text=t("method_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("srsc_unavailable"),
                  style="Sub.TLabel",
                  wraplength=520).pack(anchor='w', padx=px, pady=(0, 4))

        for val, key in [("gmm", "gmm_desc"), ("kmeans", "kmeans_desc")]:
            ttk.Radiobutton(inner, text=t(key),
                            variable=self._jt_method,
                            value=val).pack(anchor='w', padx=px + 4, pady=1)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Parameter frames (toggled by method) ─────────────────────
        params_area = ttk.Frame(inner)
        params_area.pack(fill='x')

        # --- K-based params frame ---
        k_frame = ttk.Frame(params_area)
        self._jt_k_params_frame = k_frame

        ttk.Label(k_frame, text=t("k_params_section"),
                  style="Sub.TLabel").pack(anchor='w', padx=px)

        def jrow(label, widget_fn, parent=None):
            p = parent if parent is not None else k_frame
            f = ttk.Frame(p)
            f.pack(fill='x', padx=px, pady=3)
            ttk.Label(f, text=label, width=30).pack(side='left')
            widget_fn(f)

        jrow(t("k_min_label"),
             lambda f: ttk.Spinbox(f, from_=2, to=10,
                                    textvariable=self._jt_k_min,
                                    width=6).pack(side='left'))
        jrow(t("k_max_label"),
             lambda f: ttk.Spinbox(f, from_=3, to=15,
                                    textvariable=self._jt_k_max,
                                    width=6).pack(side='left'))
        jrow(t("k_override"),
             lambda f: ttk.Entry(f, textvariable=self._jt_k_override,
                                  width=6).pack(side='left'))
        jrow(t("gmm_n_init"),
             lambda f: ttk.Spinbox(f, from_=1, to=50,
                                    textvariable=self._jt_n_init_gmm,
                                    width=6).pack(side='left'))
        jrow(t("n_refs"),
             lambda f: ttk.Spinbox(f, from_=5, to=200,
                                    textvariable=self._jt_n_refs,
                                    width=6).pack(side='left'))
        ttk.Checkbutton(k_frame,
                        text=t("gmm_guard_short", thr=GMM_SILHOUETTE_GUARD),
                        variable=self._jt_gmm_guard
                        ).pack(anchor='w', padx=px + 4, pady=2)
        ttk.Checkbutton(k_frame,
                        text=t("kmeans_guard_short"),
                        variable=self._jt_kmeans_guard
                        ).pack(anchor='w', padx=px + 4, pady=2)

        self._jt_k_params_frame.pack(fill='x')

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── d_topo spatial feature ───────────────────────────────────
        hdr_f = ttk.Frame(inner)
        hdr_f.pack(fill='x', padx=px, pady=(4, 0))
        ttk.Label(hdr_f, text=t("dtopo_section"),
                  style="Section.TLabel").pack(side='left')
        ttk.Button(hdr_f, text="?", width=2,
                   style="Secondary.TButton",
                   command=self._show_dtopo_help).pack(side='left', padx=(8, 0))

        def dtopo_row(label, widget_fn):
            f = ttk.Frame(inner)
            f.pack(fill='x', padx=px, pady=3)
            ttk.Label(f, text=label, width=30).pack(side='left')
            widget_fn(f)

        dtopo_row(
            t("dtopo_weight_label"),
            lambda f: ttk.Spinbox(
                f, from_=0.0, to=1.0, increment=0.05, format="%.2f",
                textvariable=self._jt_dtopo_weight, width=7
            ).pack(side='left'))

        dtopo_row(
            t("dtopo_min_frac_label"),
            lambda f: ttk.Spinbox(
                f, from_=1, to=30, increment=1,
                textvariable=self._jt_dtopo_min_frac_pct, width=7
            ).pack(side='left'))

        ttk.Label(inner,
                  text=t("dtopo_hint"),
                  style="Sub.TLabel", wraplength=500
                  ).pack(anchor='w', padx=px, pady=(0, 4))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Output ───────────────────────────────────────────────────
        ttk.Label(inner, text=t("output_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)

        f_name = ttk.Frame(inner)
        f_name.pack(fill='x', padx=px, pady=3)
        ttk.Label(f_name, text=t("folder_name"), width=30).pack(side='left')
        ttk.Entry(f_name, textvariable=self._jt_output_name,
                  width=22).pack(side='left')

        f_dir = ttk.Frame(inner)
        f_dir.pack(fill='x', padx=px, pady=3)
        ttk.Entry(f_dir, textvariable=self._jt_output_dir,
                  state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(f_dir, text=t("browse"), style="Secondary.TButton",
                   command=self._jt_select_output).pack(side='right', padx=(6, 0))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # ── Section : Comparaison de groupes ──────────────────────────
        ttk.Label(inner, text=t("group_compare_section"),
                  style="Section.TLabel").pack(anchor='w', **pad)
        ttk.Label(inner,
                  text=t("group_compare_desc_jt"),
                  style="Sub.TLabel", wraplength=510).pack(anchor='w', padx=px, pady=(0, 4))
        ttk.Checkbutton(inner,
                        text=t("enable_group_compare"),
                        variable=self._jt_compare_enabled,
                        command=self._jt_toggle_compare
                        ).pack(anchor='w', padx=px + 4, pady=(0, 4))

        jt_gf = ttk.Frame(inner, style="Card.TFrame")
        jt_gf.pack(fill='x', padx=px, pady=(0, 8))
        self._jt_gnames_frame = jt_gf
        self._jt_refresh_gnames()

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)

    # ── Group compare helpers (Joint) ─────────────────────────────────

    def _jt_toggle_compare(self):
        self._jt_refresh_gnames()
        self._jt_refresh_list()

    def _jt_refresh_gnames(self):
        f = self._jt_gnames_frame
        if f is None:
            return
        for w in f.winfo_children():
            w.destroy()
        if not self._jt_compare_enabled.get():
            ttk.Label(f, text=t("compare_disabled"),
                      style="Sub.TLabel").pack(anchor='w', padx=6, pady=4)
            return
        for gi, gvar in enumerate(self._jt_group_names):
            row = ttk.Frame(f)
            row.pack(fill='x', padx=4, pady=2)
            ttk.Label(row, text=t("group_n_label", n=gi), width=10).pack(side='left', padx=(4, 2))
            ttk.Entry(row, textvariable=gvar, width=18).pack(side='left', padx=4)

    # ── Animal card helpers ──────────────────────────────────────────

    def _jt_add_mouse(self):
        self._jt_animals.append({"files": [], "group_var": tk.IntVar(value=0)})
        self._jt_refresh_list()

    def _jt_remove_mouse(self, idx):
        if len(self._jt_animals) <= 1:
            return
        self._jt_animals.pop(idx)
        self._jt_refresh_list()

    def _jt_refresh_list(self):
        if self._jt_list_inner is None:
            return
        for w in self._jt_list_inner.winfo_children():
            w.destroy()
        for i, animal in enumerate(self._jt_animals):
            self._jt_build_card(i, animal)

    def _jt_build_card(self, i, animal):
        card = ttk.Frame(self._jt_list_inner)
        card.pack(fill='x', padx=4, pady=2)

        ttk.Label(card, text=t("mouse_n", n=i + 1),
                  font=THEME["font_bold"], width=9).pack(side='left', padx=(4, 2))

        n   = len(animal["files"])
        lbl = ttk.Label(card,
                        text=t("n_files", n=n) if n else t("no_files"),
                        style="Sub.TLabel", width=12)
        lbl.pack(side='left', padx=6)
        animal["_lbl"] = lbl

        ttk.Button(card, text=t("browse"), style="Secondary.TButton",
                   command=lambda a=animal: self._jt_browse(a)
                   ).pack(side='left', padx=2)
        ttk.Button(card, text="📁", style="Secondary.TButton", width=3,
                   command=lambda a=animal: self._browse_animal_folder(a)
                   ).pack(side='left', padx=2)

        if self._jt_compare_enabled.get():
            ttk.Label(card, text=t("group_label"), style="Sub.TLabel").pack(side='left', padx=(10, 2))
            gvar = animal.setdefault("group_var", tk.IntVar(value=0))
            ttk.Spinbox(card, from_=0, to=max(len(self._jt_group_names) - 1, 1),
                        textvariable=gvar, width=4).pack(side='left', padx=2)

        ttk.Button(card, text="✕", style="Secondary.TButton", width=2,
                   state='normal' if len(self._jt_animals) > 1 else 'disabled',
                   command=lambda idx=i: self._jt_remove_mouse(idx)
                   ).pack(side='right', padx=4)

    def _jt_browse(self, animal):
        files = filedialog.askopenfilenames(
            title=t("dlg_select_csv_mouse"),
            filetypes=[("CSV files", "*.csv")])
        if files:
            animal["files"] = list(files)
            if animal.get("_lbl"):
                animal["_lbl"].config(
                    text=t("n_files", n=len(files)),
                    foreground=THEME["text"])

    def _jt_select_output(self):
        path = filedialog.askdirectory(title=t("dlg_select_output"))
        if path:
            self._jt_output_dir.set(path)

    # ------------------------------------------------------------------
    # Joint clustering tab — pipeline
    # ------------------------------------------------------------------

    def _run_joint(self):
        if len(self._jt_animals) < 2:
            messagebox.showerror(t("err_title"), t("err_min_mice"))
            return
        if any(not a["files"] for a in self._jt_animals):
            messagebox.showerror(t("err_title"), t("err_mouse_no_csv"))
            return

        default_dir = os.path.dirname(self._jt_animals[0]["files"][0])
        if not self._confirm_output_dir(self._jt_output_dir, self._jt_output_name, default_dir):
            return

        if self._jt_output_dir.get():
            out_root = os.path.join(self._jt_output_dir.get(),
                                    self._jt_output_name.get())
        else:
            out_root = os.path.join(default_dir, self._jt_output_name.get())

        _GENERIC = {"v2", "v3", "v4", "v5", "gmm", "kmeans", "srsc",
                    "results", "output", "out", "run", "csv", "nifti",
                    "data", "raw", "joint_clustering", "joint"}

        def _animal_name(files, fallback):
            if not files:
                return fallback
            parts  = os.path.abspath(files[0]).replace("\\", "/").split("/")
            parent = parts[-2] if len(parts) >= 2 else fallback
            if parent.lower() in _GENERIC and len(parts) >= 3:
                return parts[-3]
            return parent if parent else fallback

        mice_list   = []
        used_names: dict = {}
        for i, animal in enumerate(self._jt_animals):
            raw_name = _animal_name(animal["files"], f"Mouse_{i + 1}")
            if raw_name in used_names:
                used_names[raw_name] += 1
                name = f"{raw_name}_{used_names[raw_name]}"
            else:
                used_names[raw_name] = 1
                name = raw_name
            mice_list.append({"name": name, "files": list(animal["files"])})

        k_ov_str = self._jt_k_override.get().strip()
        k_ov     = (int(k_ov_str)
                    if k_ov_str.isdigit() and int(k_ov_str) >= 2
                    else None)

        gmm_guard = GMM_SILHOUETTE_GUARD if self._jt_gmm_guard.get() else None

        params = {
            "mice_list"              : mice_list,
            "method"                 : self._jt_method.get(),
            "k_range"                : range(self._jt_k_min.get(),
                                             self._jt_k_max.get() + 1),
            "k_override"             : k_ov,
            "n_init_gmm"             : self._jt_n_init_gmm.get(),
            "n_refs"                 : self._jt_n_refs.get(),
            "normalization"          : self._jt_normalization.get(),
            "gmm_silhouette_guard"   : gmm_guard,
            "kmeans_gap_guard"       : self._jt_kmeans_guard.get(),
            "out_root"               : out_root,
            "dtopo_weight"           : round(self._jt_dtopo_weight.get(), 3),
            "dtopo_min_frac"         : self._jt_dtopo_min_frac_pct.get() / 100.0,
        }

        self.btn_jt_run.config(state='disabled', text=t("running"))
        self._write_log(f"[Joint] Starting — {len(mice_list)} mice, "
                        f"method={params['method'].upper()}")
        threading.Thread(target=self._run_joint_pipeline,
                         args=(params,), daemon=True).start()

    def _run_joint_pipeline(self, p):
        def log(msg, tag=None):
            self.queue.put({"type": "log", "text": msg, "tag": tag})

        try:
            out_root = p["out_root"]
            os.makedirs(out_root, exist_ok=True)

            normalization = p.get("normalization", "cl")

            result = run_joint_pipeline(
                mice_list                = p["mice_list"],
                method                   = p["method"],
                k_range                  = p["k_range"],
                n_init_gmm               = p["n_init_gmm"],
                n_refs                   = p["n_refs"],
                output_dir               = out_root,
                log_fn                   = log,
                k_override               = p.get("k_override"),
                normalization            = normalization,
                gmm_silhouette_guard     = p.get("gmm_silhouette_guard"),
                kmeans_gap_guard         = p.get("kmeans_gap_guard", False),
                dtopo_weight             = p.get("dtopo_weight", DTOPO_JOINT_WEIGHT),
                dtopo_min_frac           = p.get("dtopo_min_frac", DTOPO_MIN_NECROSIS_FRACTION),
            )

            best_k        = result['best_k']
            centroids     = result['centroids']
            common_params = result['parameters']
            df_pooled     = result['df_pooled']
            dtopo_report  = result.get('dtopo_report', {})
            dtopo_weight  = result.get('dtopo_weight', 1.0)
            dtopo_min_frac= result.get('dtopo_min_frac', 0.05)
            log(f"[Joint] Clustering complete — {best_k} universal habitats (H0…H{best_k-1})")

            habitat_labels = {c: f"H{c}" for c in range(best_k)}

            log("[Joint] Exporting per-mouse results…")
            _jt_export_per_mouse(
                df_pooled, common_params,
                p["mice_list"], out_root, best_k,
                normalization=normalization, log_fn=log)

            log("[Joint] Generating boundary analysis PDF…")
            try:
                _generate_boundary_analysis(out_root)
                log("[Joint] joint_boundaries_analysis.pdf saved.", tag="ok")
            except Exception as _e:
                log(f"[Joint] Boundary analysis skipped: {_e}")

            # Build summary text
            _norm_desc_map = {
                'cl'           : "CL-based  (0 = contralateral tissue)",
                'cl_hybrid'    : "CL-hybrid  (CL centering + tumor-IQR scale)",
                'robust'       : "per-mouse robust  (0 = each mouse's tumor median)",
                'robust_global': "global robust  (0 = global tumor median)",
            }
            norm_desc = _norm_desc_map.get(normalization, normalization)
            n_total = len(df_pooled)
            lines = [
                "═" * 62,
                "  JOINT MULTI-MOUSE CLUSTERING — RESULTS",
                "═" * 62,
                f"  Mice              : {len(p['mice_list'])}",
                f"  Method            : {p['method'].upper()}",
                f"  Universal habitats: {best_k}",
                f"  Parameters        : {', '.join(common_params)}",
                f"  Total voxels      : {n_total}",
                f"  Normalization     : {norm_desc}",
                "",
                "  UNIVERSAL HABITAT PROFILES  (CL-normalized):",
                "",
            ]
            for c in range(best_k):
                lbl = habitat_labels.get(c, f"H{c}")
                n_c = int((df_pooled['Habitat'] == c).sum())
                pct = 100 * n_c / n_total
                lines.append(f"  H{c}  {lbl:<25}  {n_c:>6} voxels  ({pct:.1f}% of all)")
                for p_name in common_params:
                    val = centroids[c].get(p_name, 0.0)
                    lines.append(f"    {p_name:<12} {val:+.3f}")
                lines.append("")

            lines += [
                "═" * 62,
                "  HABITAT PRESENCE PER MOUSE",
                "═" * 62,
            ]
            for i, mouse in enumerate(p['mice_list']):
                name = mouse['name']
                mask = df_pooled['mouse_id'].values == i
                df_m = df_pooled[mask]
                n_m  = int(mask.sum())
                lines.append(f"\n  {name}  ({n_m} voxels)")
                for c in range(best_k):
                    lbl = habitat_labels.get(c, f"H{c}")
                    n_c = int((df_m['Habitat'] == c).sum())
                    pct = 100 * n_c / n_m if n_m > 0 else 0.0
                    lines.append(
                        f"    H{c}  {lbl:<22}  {n_c:>5} vox  {pct:4.1f}%")

            if normalization == 'cl':
                note_lines = [
                    "  Values in CL-IQR units.  0 = contralateral level.",
                    "  +1 = one CL-IQR above healthy tissue.",
                    "  Directly comparable across mice (same biological zero).",
                ]
            else:
                note_lines = [
                    "  Values in per-mouse tumor IQR units.",
                    "  0 = each mouse's own tumor median (not the same biological state).",
                    "  Intra-mouse heterogeneity is preserved; cross-mouse magnitude",
                    "  comparisons should be made with caution.",
                ]
            lines += [
                "",
                "═" * 62,
                "  NOTE",
                "═" * 62,
            ] + note_lines + [
                "  Same H-index = same universal habitat across all mice.",
                "  No biological labels applied — raw data only.",
                "  PDF report (joint_report.pdf) + per-mouse segmentation maps in output folder.",
                "",
            ]

            summary = "\n".join(lines)
            self.queue.put({"type"         : "jt_results",
                            "summary"      : summary,
                            "output_dir"   : out_root,
                            "dtopo_report" : dtopo_report,
                            "dtopo_weight" : dtopo_weight,
                            "dtopo_min_frac": dtopo_min_frac})
            log("[Joint] Complete.", tag="ok")

        except Exception as e:
            self.queue.put({
                "type": "error",
                "text": f"\n[ERROR Joint] {e}\n{traceback.format_exc()}"
            })
        finally:
            self.queue.put({"type": "jt_done"})

    # ------------------------------------------------------------------
    # Joint clustering — results window
    # ------------------------------------------------------------------

    def _show_dtopo_help(self):
        _show_dtopo_help_window()

    def _show_joint_results(self, summary: str, output_dir: str,
                             dtopo_report=None, dtopo_weight=1.0,
                             dtopo_min_frac=0.05):
        win = tk.Toplevel(self)
        win.title(t("wintitle_jt_results"))
        win.geometry("820x640")
        win.configure(bg=THEME["bg"])
        win.lift()

        nb = ttk.Notebook(win)
        nb.pack(fill='both', expand=True, padx=8, pady=(8, 4))

        # ── Tab 1 — summary text ────────────────────────────────────────
        tab_text = ttk.Frame(nb)
        nb.add(tab_text, text=t("tab_summary"))

        txt = scrolledtext.ScrolledText(
            tab_text,
            font=("Courier", 9),
            bg=THEME["log_bg"], fg=THEME["log_fg"],
            insertbackground=THEME["log_fg"],
            relief="flat", borderwidth=0,
            padx=10, pady=6, wrap=tk.NONE)
        txt.pack(fill='both', expand=True)
        txt.insert('1.0', summary)
        txt.config(state='disabled')

        hbar = ttk.Scrollbar(tab_text, orient='horizontal', command=txt.xview)
        txt.configure(xscrollcommand=hbar.set)
        hbar.pack(fill='x')

        # ── Tab 2 — output folder ───────────────────────────────────────
        tab_out = ttk.Frame(nb)
        nb.add(tab_out, text=t("tab_output_folder"))
        ttk.Label(tab_out, text=t("results_saved_to"),
                  style="Sub.TLabel").pack(anchor='w', padx=16, pady=(16, 4))
        ttk.Label(tab_out, text=output_dir,
                  style="Sub.TLabel", wraplength=700).pack(anchor='w', padx=16)
        ttk.Button(tab_out, text=t("open_folder"),
                   style="Primary.TButton",
                   command=lambda: os.startfile(output_dir)
                   ).pack(padx=16, pady=12)
        ttk.Label(tab_out,
                  text=t("jt_output_contents"),
                  style="Sub.TLabel",
                  wraplength=680).pack(anchor='w', padx=16)

        # ── Tab 3 — d_topo impact ───────────────────────────────────────
        tab_dtopo = ttk.Frame(nb)
        nb.add(tab_dtopo, text=t("tab_dtopo_impact"))

        canvas = tk.Canvas(tab_dtopo, bg=THEME["bg"], highlightthickness=0)
        sb     = ttk.Scrollbar(tab_dtopo, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_ = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        px = 16

        # — What is d_topo? —
        tk.Label(inner, text=t("dtopo_what_title"),
                 font=(THEME["font_family"], 11, "bold"),
                 bg=THEME["bg"], fg=THEME["text"]
                 ).pack(anchor='w', padx=px, pady=(14, 2))
        tk.Label(inner, wraplength=740, justify='left',
                 font=THEME["font_small"], bg=THEME["bg"], fg=THEME["text_sub"],
                 text=(
                     "d_topo_norm is a spatial feature added in Pass 2. It measures how far "
                     "each voxel is from the necrotic core (identified in Pass 1), then "
                     "normalises that distance per mouse using the median and IQR.\n\n"
                     "Low d_topo_norm (≈ −0.85 for H0) → spatially close to necrosis.\n"
                     "High d_topo_norm (≈ +0.64 for H2) → spatially far from necrosis (periphery).\n\n"
                     "Risk: in mice with little or no necrosis, this feature can pull voxels into "
                     "the 'necrosis' habitat based on their map position rather than their MRI values."
                 )).pack(anchor='w', padx=px, pady=(0, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # — Current settings —
        tk.Label(inner, text=t("dtopo_settings_title"),
                 font=(THEME["font_family"], 11, "bold"),
                 bg=THEME["bg"], fg=THEME["text"]
                 ).pack(anchor='w', padx=px, pady=(4, 2))

        settings_text = (
            f"  • Weight applied to d_topo_norm  :  {dtopo_weight}  "
            f"({'equal to MRI features' if dtopo_weight == 1.0 else 'reduced — MRI features dominate'})\n"
            f"  • Min necrosis fraction to include:  {100*dtopo_min_frac:.0f}%  "
            f"(mice below this threshold → d_topo set to 0)"
        )
        tk.Label(inner, text=settings_text, justify='left',
                 font=("Courier", 9), bg=THEME["bg"], fg=THEME["text_sub"]
                 ).pack(anchor='w', padx=px, pady=(0, 4))

        tk.Label(inner, text=t("dtopo_adjust_hint"),
                 font=THEME["font_small"], bg=THEME["bg"], fg=THEME["accent"]
                 ).pack(anchor='w', padx=px, pady=(0, 8))

        ttk.Separator(inner).pack(fill='x', padx=px, pady=6)

        # — Per-mouse decision table —
        tk.Label(inner, text=t("dtopo_per_mouse_title"),
                 font=(THEME["font_family"], 11, "bold"),
                 bg=THEME["bg"], fg=THEME["text"]
                 ).pack(anchor='w', padx=px, pady=(4, 6))

        if not dtopo_report:
            tk.Label(inner, text=t("dtopo_not_activated"),
                     font=THEME["font_small"], bg=THEME["bg"], fg=THEME["text_sub"]
                     ).pack(anchor='w', padx=px)
        else:
            # Header row
            hdr = ttk.Frame(inner)
            hdr.pack(fill='x', padx=px, pady=(0, 2))
            for txt_h, w in [(t("dtopo_table_mouse"), 28), (t("dtopo_table_necro"), 10),
                              ("d_topo", 8), ("Reason", 42)]:
                tk.Label(hdr, text=txt_h, width=w, anchor='w',
                         font=(THEME["font_family"], 9, "bold"),
                         bg=THEME["bg"], fg=THEME["accent"]
                         ).pack(side='left')

            ttk.Separator(inner).pack(fill='x', padx=px, pady=2)

            for mouse, info in sorted(dtopo_report.items()):
                row = ttk.Frame(inner)
                row.pack(fill='x', padx=px, pady=1)

                included = info['included']
                pct      = f"{100*info['fraction']:.1f}%"
                status   = t("dtopo_included") if included else t("dtopo_skipped")
                s_color  = THEME["log_ok"] if included else THEME["log_err"]

                short_mouse = mouse[:27] + "…" if len(mouse) > 28 else mouse
                short_reason = info['reason'][:90] + "…" if len(info['reason']) > 90 else info['reason']

                tk.Label(row, text=short_mouse, width=28, anchor='w',
                         font=THEME["font_small"], bg=THEME["bg"],
                         fg=THEME["text"]).pack(side='left')
                tk.Label(row, text=pct, width=10, anchor='w',
                         font=THEME["font_small"], bg=THEME["bg"],
                         fg=THEME["text"]).pack(side='left')
                tk.Label(row, text=status, width=10, anchor='w',
                         font=(THEME["font_family"], 9, "bold"),
                         bg=THEME["bg"], fg=s_color).pack(side='left')
                tk.Label(row, text=short_reason, anchor='w', wraplength=380,
                         font=THEME["font_small"], bg=THEME["bg"],
                         fg=THEME["text_sub"]).pack(side='left', fill='x', expand=True)

        ttk.Separator(inner).pack(fill='x', padx=px, pady=8)

        # — Interpretation guide —
        tk.Label(inner, text=t("dtopo_interp_title"),
                 font=(THEME["font_family"], 11, "bold"),
                 bg=THEME["bg"], fg=THEME["text"]
                 ).pack(anchor='w', padx=px, pady=(0, 4))
        tk.Label(inner, wraplength=740, justify='left',
                 font=THEME["font_small"], bg=THEME["bg"], fg=THEME["text_sub"],
                 text=(
                     "• Mice marked ✔ included: d_topo contributed to their voxel assignment. "
                     "H0 in these mice reflects both MRI tissue values AND spatial proximity to necrosis.\n\n"
                     "• Mice marked ✘ skipped: voxel assignment is based on MRI features only. "
                     "H0 in these mice is defined purely by DTI/T2/MTR values — not by tumour anatomy.\n\n"
                     "• If many mice are skipped, your cohort has heterogeneous necrosis loads. "
                     "The global H0 centroid may not be biologically meaningful for the skipped mice — "
                     "treat their H0 label with caution.\n\n"
                     "• Open joint_boundaries_analysis.pdf in the output folder to see the full "
                     "per-mouse parallel coordinates and deviation heatmap."
                 )).pack(anchor='w', padx=px, pady=(0, 14))

        ttk.Button(win, text=t("close"), style="Secondary.TButton",
                   command=win.destroy).pack(pady=(0, 8))

    # ------------------------------------------------------------------
    # Discovery — results window
    # ------------------------------------------------------------------

    def _show_discovery_results(self, text: str, pdf_path: str):
        win = tk.Toplevel(self)
        win.title(t("wintitle_disc_results"))
        win.geometry("780x620")
        win.configure(bg=THEME["bg"])
        win.lift()

        nb = ttk.Notebook(win)
        nb.pack(fill='both', expand=True, padx=8, pady=(8, 4))

        # Tab 1: explanation text
        tab_text = ttk.Frame(nb)
        nb.add(tab_text, text=t("tab_explanation"))

        txt = scrolledtext.ScrolledText(
            tab_text,
            font=("Courier", 9),
            bg=THEME["log_bg"], fg=THEME["log_fg"],
            insertbackground=THEME["log_fg"],
            relief="flat", borderwidth=0,
            padx=10, pady=6, wrap=tk.NONE)
        txt.pack(fill='both', expand=True)
        txt.insert('1.0', text)
        txt.config(state='disabled')

        # Horizontal scrollbar for the text
        hbar = ttk.Scrollbar(tab_text, orient='horizontal', command=txt.xview)
        txt.configure(xscrollcommand=hbar.set)
        hbar.pack(fill='x')

        # Tab 2: figures (open PDF)
        tab_fig = ttk.Frame(nb)
        nb.add(tab_fig, text=t("tab_figures_pdf"))
        ttk.Label(tab_fig, text=t("pdf_report_with_figs"),
                  style="Sub.TLabel").pack(anchor='w', padx=16, pady=(16, 4))
        ttk.Label(tab_fig, text=pdf_path,
                  style="Sub.TLabel", wraplength=700).pack(anchor='w', padx=16)
        ttk.Button(tab_fig, text=t("open_pdf"),
                   style="Primary.TButton",
                   command=lambda: os.startfile(pdf_path)
                   ).pack(padx=16, pady=12)

        # Bottom: close
        ttk.Button(win, text=t("close"), style="Secondary.TButton",
                   command=win.destroy).pack(pady=(0, 8))

# =================================================================
# stdout -> queue redirect
# =================================================================

class _LogRedirect:
    def __init__(self, q):
        self._q   = q
        self._buf = ""

    def write(self, msg):
        if "\n" in msg:
            lines = (self._buf + msg).split("\n")
            for line in lines[:-1]:
                self._q.put({"type": "log", "text": line})
            self._buf = lines[-1]
        else:
            self._buf += msg

    def flush(self):
        if self._buf:
            self._q.put({"type": "log", "text": self._buf})
            self._buf = ""


# =================================================================
# Entry point (called from main.ipynb or directly)
# =================================================================

def launch():
    app = HabitatApp()
    sys.stdout = _LogRedirect(app.queue)

    def _bring_to_front():
        app.attributes('-topmost', True)
        app.lift()
        app.focus_force()
        app.after(200, lambda: app.attributes('-topmost', False))

    app.after(100, _bring_to_front)

    try:
        while app.winfo_exists():
            app.update()
            app.update_idletasks()
    except tk.TclError:
        pass
    finally:
        sys.stdout = sys.__stdout__

    if app._napari_params is not None:
        p           = app._napari_params
        script_path = os.path.join(os.path.dirname(__file__),
                                   "launch_napari_standalone.py")
        subprocess.Popen([
            sys.executable, script_path,
            p["nifti"], p["mask_path"], p["labels_path"]
        ])


if __name__ == "__main__":
    launch()
