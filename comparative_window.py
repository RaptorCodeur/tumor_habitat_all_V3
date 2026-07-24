"""
comparative_window.py — Comparative Analysis window.

Opens a standalone Toplevel that lets the user:
  1. Select N mice (each as a folder of per-parameter CSV files)
  2. Run Classic, Discovery and Joint analyses on the same cohort
  3. See a side-by-side PCA + proportion + centroid view per mouse
"""
import os
import threading
import queue
import math

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

# ── Pipeline imports ────────────────────────────────────────────────────────
from pipeline.loading    import load_data, robust_scale, resolve_dti_parameters
from pipeline.spatial    import prepare_features
from pipeline.clustering import run_clustering
from pipeline.joint_clustering import run_joint_pipeline
from multi_mouse_discovery import (
    load_mouse_data, find_common_parameters,
    build_centroid_matrix, run_meta_clustering,
    find_optimal_meta_k,
)
from config import (
    HABITAT_COLORS, HABITAT_COLORS_HEX,
    GMM_SILHOUETTE_GUARD, KMEANS_GAP_GUARD,
    JOINT_MIN_CLUSTER_SIZE_RATIO,
    DTOPO_JOINT_WEIGHT, DTOPO_MIN_NECROSIS_FRACTION,
    N_JOBS_GAP, RANDOM_SEED, APP_THEME,
)
from i18n import t

# Build a local THEME dict matching the one in app.py
_THEMES = {
    "dark": {
        "bg": "#1A1D23", "bg_card": "#21262F", "bg_input": "#2A3140",
        "accent": "#4A90D9", "text": "#DDE4EE", "text_sub": "#7A8899",
        "separator": "#2E3849",
        "log_bg": "#10131A", "log_fg": "#B8C8DC",
        "log_ok": "#4CAF7D", "log_err": "#E05252",
        "font_family": "Helvetica",
        "font_log": ("Courier", 9),
        "pad_x": 16, "pad_y": 8,
    },
    "light": {
        "bg": "#F7F8FA", "bg_card": "#FFFFFF", "bg_input": "#FFFFFF",
        "accent": "#3A6EA5", "text": "#1C1C1E", "text_sub": "#6B7280",
        "separator": "#DDE1E7",
        "log_bg": "#1E2228", "log_fg": "#C8D3E0",
        "log_ok": "#5CB85C", "log_err": "#E05252",
        "font_family": "Helvetica",
        "font_log": ("Courier", 9),
        "pad_x": 16, "pad_y": 8,
    },
}
THEME = _THEMES.get(APP_THEME, _THEMES["dark"])

# ── constants ───────────────────────────────────────────────────────────────
_PAD     = 8
_METHODS = ["gmm", "kmeans"]
_NORMS   = ["cl", "robust"]

_CLASSIC_COL  = "#4A90D9"
_DISC_COL     = "#E67E22"
_JOINT_COL    = "#27AE60"
_METHOD_NAMES = {"classic": "Classic", "discovery": "Discovery", "joint": "Joint"}

_N_REFS_DEFAULT    = 30
_N_INIT_GMM_DEFAULT = 10


# ════════════════════════════════════════════════════════════════════════════
# Helper: run classic pipeline for one mouse (no hierarchical / MRF)
# ════════════════════════════════════════════════════════════════════════════

def _run_classic_for_mouse(csv_files, method, k_range, k_override,
                            n_init_gmm, n_refs, out_dir, log):
    """
    Light-weight single-mouse classic run.
    Returns dict with keys: labels, features, parameters, df, centroids, k
    or None on failure.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        parameters, df_all = load_data(csv_files)
        parameters, _      = resolve_dti_parameters(parameters)
        df_scaled, feat_sc, _ = robust_scale(df_all, parameters)

        features_clust, dtopo_meta = prepare_features(
            feat_sc, df_all, parameters, n_init_gmm=n_init_gmm)

        k_ov = int(k_override) if str(k_override).isdigit() else None

        labels, best_k, extra = run_clustering(
            features_clust,
            method            = method,
            k_range           = k_range,
            k_override        = k_ov,
            spatial_weight    = 0.0,
            n_refs            = n_refs,
            n_init_gmm        = n_init_gmm,
            n_jobs            = N_JOBS_GAP,
            gmm_silhouette_guard = GMM_SILHOUETTE_GUARD if method == "gmm" else None,
            kmeans_gap_guard  = (method == "kmeans"),
        )

        df_scaled["Habitat"] = labels
        df_scaled.to_csv(os.path.join(out_dir, "habitats_result.csv"), index=False)

        # Global centroids (mean of normalized features per habitat)
        feat_cols = [c for c in df_scaled.columns
                     if c not in ("X", "Y", "Slice", "Habitat", "d_topo_norm")]
        centroids = (df_scaled.groupby("Habitat")[feat_cols]
                     .mean().values)

        log(f"  Classic k={best_k}  ({len(np.unique(labels))} habitats)")
        return {
            "labels"    : labels,
            "features"  : feat_sc,
            "parameters": feat_cols,
            "df"        : df_scaled,
            "centroids" : centroids,
            "k"         : best_k,
        }
    except Exception as e:
        log(f"  [!] Classic failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Helper: derive per-voxel discovery labels from classic results
# ════════════════════════════════════════════════════════════════════════════

def _run_discovery_meta(classic_results, result_csv_paths, method,
                         k_range, k_override, n_init, log):
    """
    Meta-cluster the per-mouse classic centroids and map back to voxel level.
    Returns dict  mouse_name -> {'labels', 'centroids', 'k', 'common_params'}
    or None on failure.
    """
    try:
        # Build mice_list expected by discovery helpers
        mice_list = []
        for name, csv_path in result_csv_paths.items():
            m = load_mouse_data(csv_path)
            if m is not None:
                m["name"] = name          # override with our canonical name
                mice_list.append(m)
            else:
                log(f"  [Discovery] Skipped {name} (no valid clusters)")

        if len(mice_list) < 2:
            log("  [Discovery] Need ≥2 valid mice — skipped")
            return None

        common_params = find_common_parameters(mice_list)
        if not common_params:
            log("  [Discovery] No common parameters — skipped")
            return None

        X, row_meta = build_centroid_matrix(mice_list, common_params)

        if k_override and str(k_override).isdigit():
            chosen_k = min(int(k_override), len(X) - 1)
        else:
            safe_kmax  = min(max(k_range), len(X) - 1)
            safe_range = range(min(k_range), safe_kmax + 1)
            chosen_k, _ = find_optimal_meta_k(
                X, safe_range, n_init, RANDOM_SEED, method, log)

        chosen_k = max(2, chosen_k)
        log(f"  [Discovery] Meta-clustering K={chosen_k}")
        meta_labels = run_meta_clustering(X, chosen_k, n_init, RANDOM_SEED, method)

        # Build centroid positions in common_params space
        meta_centroids = np.array([
            X[meta_labels == i].mean(axis=0)
            for i in range(chosen_k)
        ])

        # Build cluster→meta mapping per mouse
        cluster_to_meta = {}
        for i, rm in enumerate(row_meta):
            cluster_to_meta[(rm["mouse"], rm["cluster_id"])] = int(meta_labels[i])

        # Map back to voxel level using the classic labels
        out = {}
        for name, cr in classic_results.items():
            df         = cr["df"].copy()
            hab_col    = df["Habitat"].values
            disc_lbls  = np.array([
                cluster_to_meta.get((name, int(h)), 0)
                for h in hab_col
            ])
            out[name] = {
                "labels"       : disc_lbls,
                "features"     : cr["features"],
                "parameters"   : cr["parameters"],
                "centroids"    : meta_centroids,
                "common_params": common_params,
                "k"            : chosen_k,
            }
        return out
    except Exception as e:
        log(f"  [Discovery] Error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ════════════════════════════════════════════════════════════════════════════

def _hab_color(hab_id, alpha=1.0):
    r, g, b, _ = HABITAT_COLORS[hab_id % len(HABITAT_COLORS)]
    return (r / 255, g / 255, b / 255, alpha)


def _pca_scatter(ax, features, labels, centroids, title, header_color,
                 parameters=None):
    """PCA scatter of voxels coloured by habitat, with centroid stars."""
    if features.shape[1] > 2:
        pca     = PCA(n_components=2, random_state=RANDOM_SEED)
        pts     = pca.fit_transform(features)
        cents   = pca.transform(centroids) if centroids is not None else None
        var1, var2 = pca.explained_variance_ratio_ * 100
        xlabel  = f"PC1 ({var1:.0f}%)"
        ylabel  = f"PC2 ({var2:.0f}%)"
    else:
        pts   = features
        cents = centroids
        cols  = (parameters or []) + ["", ""]
        xlabel, ylabel = cols[0], cols[1]

    unique_ids = sorted(np.unique(labels))
    for hid in unique_ids:
        mask = labels == hid
        ax.scatter(pts[mask, 0], pts[mask, 1],
                   s=4, alpha=0.35, linewidths=0,
                   color=_hab_color(int(hid), 0.5),
                   label=f"H{hid}")

    if cents is not None:
        for hid, c in enumerate(cents):
            ax.scatter(c[0], c[1], marker="*", s=200,
                       color=_hab_color(hid), edgecolors="white",
                       linewidths=0.8, zorder=5)

    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, fontweight="bold", color=header_color)
    ax.legend(fontsize=6, markerscale=2, framealpha=0.6,
              loc="upper right", handlelength=1)


def _proportion_bar(ax, labels, k, colors_hex, method_name):
    """Stacked horizontal proportion bar per habitat."""
    ids    = np.arange(k)
    counts = np.array([np.sum(labels == i) for i in ids])
    total  = max(counts.sum(), 1)
    fracs  = counts / total
    left   = 0.0
    for hid, frac in zip(ids, fracs):
        ax.barh(0, frac, left=left, color=_hab_color(hid),
                height=0.6, linewidth=0)
        if frac > 0.04:
            ax.text(left + frac / 2, 0, f"H{hid}\n{frac*100:.0f}%",
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold")
        left += frac
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6)
    ax.tick_params(axis="x", length=2)


def _centroid_heatmap(ax, centroids, parameters, k, title, header_color):
    """Compact heatmap of centroid values: rows = habitats, cols = features."""
    if centroids is None or len(centroids) == 0:
        ax.set_visible(False)
        return
    mat = centroids[:k]
    n_feats = min(mat.shape[1], len(parameters))
    mat = mat[:, :n_feats]
    params_short = [p.replace("DTI-", "").replace("T2star", "T2*")
                    for p in parameters[:n_feats]]
    vabs = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(params_short, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([f"H{i}" for i in range(len(mat))], fontsize=6)
    ax.set_title(title, fontsize=8, fontweight="bold", color=header_color)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    # Annotate cells
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center",
                    fontsize=5.5, color="black" if abs(mat[r, c]) < vabs * 0.6 else "white")


def _build_mouse_figure(mouse_name, classic, discovery, joint):
    """
    Build a matplotlib figure for one mouse with 3 column panels.
    Returns (fig, canvas_frame_builder).
    """
    n_cols  = 3
    n_rows  = 3    # PCA | proportions | centroids
    fig_w   = 13.5
    fig_h   = 9.0
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={"height_ratios": [5, 1, 3]})
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes.flat:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555577")

    cols_info = [
        ("Classic",   classic,   _CLASSIC_COL),
        ("Discovery", discovery, _DISC_COL),
        ("Joint",     joint,     _JOINT_COL),
    ]

    for col, (label, res, hdr_col) in enumerate(cols_info):
        ax_pca  = axes[0][col]
        ax_bar  = axes[1][col]
        ax_heat = axes[2][col]

        if res is None:
            for ax in (ax_pca, ax_bar, ax_heat):
                ax.set_visible(False)
            axes[0][col].set_visible(True)
            axes[0][col].text(0.5, 0.5, f"{label}\n(not available)",
                              transform=axes[0][col].transAxes,
                              ha="center", va="center",
                              fontsize=10, color="#888888")
            axes[0][col].set_title(label, fontsize=9, color="#888888",
                                   fontweight="bold")
            continue

        lbls  = res["labels"]
        feats = res["features"]
        cents = res["centroids"]
        params = res.get("parameters") or res.get("common_params", [])
        k     = res["k"]

        k_tag = f"K={k}  {res.get('method', '').upper() or ''}"
        title = f"{label}\n{k_tag}"

        _pca_scatter(ax_pca, feats, lbls, cents, title, hdr_col,
                     parameters=params)
        ax_pca.set_facecolor("#2a2a3e")
        for spine in ax_pca.spines.values():
            spine.set_edgecolor("#555577")
        ax_pca.tick_params(colors="#cccccc")
        ax_pca.xaxis.label.set_color("#cccccc")
        ax_pca.yaxis.label.set_color("#cccccc")
        ax_pca.legend(labelcolor="#cccccc", facecolor="#2a2a3e",
                      edgecolor="#555577", fontsize=6, markerscale=2)

        _proportion_bar(ax_bar, lbls, k, HABITAT_COLORS_HEX, label)
        ax_bar.set_facecolor("#2a2a3e")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#555577")
        ax_bar.tick_params(colors="#cccccc")

        heat_params = params if label != "Discovery" else res.get("common_params", params)
        _centroid_heatmap(ax_heat, cents, heat_params, k,
                          "Feature centroids (normalised)", hdr_col)
        ax_heat.set_facecolor("#2a2a3e")
        for spine in ax_heat.spines.values():
            spine.set_edgecolor("#555577")
        ax_heat.tick_params(colors="#cccccc")
        ax_heat.xaxis.label.set_color("#cccccc")
        ax_heat.yaxis.label.set_color("#cccccc")

    fig.suptitle(f"Mouse: {mouse_name}", fontsize=11,
                 color="white", fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Main window class
# ════════════════════════════════════════════════════════════════════════════

class ComparativeAnalysisWindow(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Comparative Analysis — Classic / Discovery / Joint")
        self.geometry("1400x860")
        self.configure(bg=THEME["bg"])
        self.minsize(1000, 600)

        self._queue   = queue.Queue()
        self._mice    = []       # list of (mouse_name, [csv_paths])
        self._running = False
        self._result_frames = []

        self._build_ui()
        self._poll_queue()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        hdr = tk.Frame(self, bg=THEME["bg_card"], height=42)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr,
                 text="Comparative Analysis",
                 font=(THEME["font_family"], 12, "bold"),
                 bg=THEME["bg_card"], fg=THEME["text"]
                 ).pack(side="left", padx=16, pady=8)
        tk.Label(hdr,
                 text="Classic  |  Discovery  |  Joint",
                 font=(THEME["font_family"], 10),
                 bg=THEME["bg_card"], fg=THEME["accent"]
                 ).pack(side="left")

        # Main paned window
        pane = tk.PanedWindow(self, orient="horizontal",
                              sashwidth=5, sashrelief="raised",
                              bg=THEME["separator"])
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: settings
        left = ttk.Frame(pane, width=300)
        pane.add(left, minsize=240)
        self._build_settings(left)

        # Right: results
        right = ttk.Frame(pane)
        pane.add(right, minsize=600)
        self._build_results_panel(right)

    def _build_settings(self, parent):
        pad = {"padx": _PAD, "pady": 3}

        # ── Mice section ──
        ttk.Label(parent, text="Mice", style="Section.TLabel"
                  ).pack(anchor="w", **pad)

        lst_frame = ttk.Frame(parent)
        lst_frame.pack(fill="x", padx=_PAD, pady=2)

        self._mice_listbox = tk.Listbox(
            lst_frame, height=7,
            font=THEME["font_log"],
            bg=THEME["bg_input"], fg=THEME["text"],
            selectbackground=THEME["accent"],
            selectforeground="#FFFFFF",
            relief="flat", borderwidth=0,
            highlightthickness=1,
            highlightbackground=THEME["separator"])
        self._mice_listbox.pack(fill="x")

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", padx=_PAD, pady=(0, 4))
        ttk.Button(btn_row, text="+ Add mouse",
                   style="Secondary.TButton",
                   command=self._add_mouse).pack(side="left", fill="x", expand=True)
        ttk.Button(btn_row, text="✕ Remove",
                   style="Secondary.TButton",
                   command=self._remove_mouse).pack(side="left", fill="x",
                                                    expand=True, padx=(4, 0))

        ttk.Separator(parent).pack(fill="x", padx=_PAD, pady=6)

        # ── Analysis settings ──
        ttk.Label(parent, text="Analysis settings",
                  style="Section.TLabel").pack(anchor="w", **pad)

        def row(lbl, widget_fn):
            f = ttk.Frame(parent)
            f.pack(fill="x", padx=_PAD, pady=2)
            ttk.Label(f, text=lbl, width=18).pack(side="left")
            widget_fn(f)

        self._method = tk.StringVar(value="gmm")
        row("Method",
            lambda f: ttk.OptionMenu(f, self._method, "gmm",
                                     *_METHODS).pack(side="left"))

        self._k_min = tk.IntVar(value=2)
        self._k_max = tk.IntVar(value=5)
        row("K range",
            lambda f: (
                ttk.Spinbox(f, from_=2, to=10, textvariable=self._k_min,
                            width=4).pack(side="left"),
                ttk.Label(f, text=" – ").pack(side="left"),
                ttk.Spinbox(f, from_=2, to=10, textvariable=self._k_max,
                            width=4).pack(side="left"),
            ))

        self._k_override = tk.StringVar(value="")
        row("Force K (opt.)",
            lambda f: ttk.Entry(f, textvariable=self._k_override,
                                width=5).pack(side="left"))

        self._norm = tk.StringVar(value="robust")
        row("Joint norm.",
            lambda f: ttk.OptionMenu(f, self._norm, "robust",
                                     *_NORMS).pack(side="left"))

        self._n_init = tk.IntVar(value=_N_INIT_GMM_DEFAULT)
        row("N inits GMM",
            lambda f: ttk.Spinbox(f, from_=1, to=50,
                                  textvariable=self._n_init,
                                  width=5).pack(side="left"))

        self._n_refs = tk.IntVar(value=_N_REFS_DEFAULT)
        row("N refs (gap)",
            lambda f: ttk.Spinbox(f, from_=5, to=200,
                                  textvariable=self._n_refs,
                                  width=5).pack(side="left"))

        ttk.Separator(parent).pack(fill="x", padx=_PAD, pady=6)

        # ── Output ──
        ttk.Label(parent, text="Output", style="Section.TLabel"
                  ).pack(anchor="w", **pad)

        self._out_dir = tk.StringVar(value="")
        f_out = ttk.Frame(parent)
        f_out.pack(fill="x", padx=_PAD, pady=2)
        ttk.Entry(f_out, textvariable=self._out_dir,
                  state="readonly").pack(side="left", fill="x", expand=True)
        ttk.Button(f_out, text="Browse", style="Secondary.TButton",
                   command=self._pick_output).pack(side="right", padx=(4, 0))
        ttk.Label(parent,
                  text="(defaults to first mouse's parent folder)",
                  style="Sub.TLabel").pack(anchor="w", padx=_PAD)

        ttk.Separator(parent).pack(fill="x", padx=_PAD, pady=8)

        # ── Run button ──
        self._btn_run = ttk.Button(parent,
                                   text="▶  Run Comparative Analysis",
                                   style="Primary.TButton",
                                   command=self._run)
        self._btn_run.pack(fill="x", padx=_PAD, pady=(0, 6))

        # ── Log ──
        ttk.Label(parent, text="Progress",
                  font=(THEME["font_family"], 9),
                  background=THEME["bg"], foreground=THEME["text_sub"]
                  ).pack(anchor="w", padx=_PAD)

        self._log_widget = scrolledtext.ScrolledText(
            parent, height=10,
            font=THEME["font_log"],
            state="disabled",
            bg=THEME["log_bg"], fg=THEME["log_fg"],
            relief="flat", borderwidth=0,
            padx=8, pady=4)
        self._log_widget.pack(fill="both", expand=True, padx=_PAD, pady=(0, 4))
        self._log_widget.tag_config("ok",  foreground=THEME["log_ok"])
        self._log_widget.tag_config("err", foreground=THEME["log_err"])

    def _build_results_panel(self, parent):
        ttk.Label(parent, text="Results",
                  style="Section.TLabel").pack(anchor="w", padx=_PAD, pady=4)

        # Scrollable canvas
        canvas_frame = tk.Frame(parent, bg=THEME["bg"])
        canvas_frame.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(canvas_frame, bg=THEME["bg"],
                                  highlightthickness=0)
        vscroll = ttk.Scrollbar(canvas_frame, orient="vertical",
                                 command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._results_inner = tk.Frame(self._canvas, bg=THEME["bg"])
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._results_inner, anchor="nw")

        self._results_inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind_all("<MouseWheel>",
                               lambda e: self._canvas.yview_scroll(
                                   int(-e.delta / 120), "units"))

        self._placeholder = ttk.Label(
            self._results_inner,
            text="Run the analysis to see results here.",
            style="Sub.TLabel")
        self._placeholder.pack(pady=40)

    # ── Geometry ─────────────────────────────────────────────────────────────

    def _on_inner_configure(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _add_mouse(self):
        folder = filedialog.askdirectory(title="Select mouse data folder")
        if not folder:
            return
        csv_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".csv") and "_roi_data" in f.lower()
        ]
        if not csv_files:
            self._log("[!] No *_roi_data.csv files found in selected folder.", "err")
            return
        mouse_name = os.path.basename(folder)
        self._mice.append((mouse_name, sorted(csv_files)))
        self._mice_listbox.insert("end", mouse_name)
        self._log(f"Added: {mouse_name}  ({len(csv_files)} CSV files)")

    def _remove_mouse(self):
        sel = self._mice_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._mice.pop(idx)
        self._mice_listbox.delete(idx)

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select output directory")
        if folder:
            self._out_dir.set(folder)

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return
        if not self._mice:
            self._log("[!] Add at least one mouse first.", "err")
            return

        method     = self._method.get()
        k_min      = self._k_min.get()
        k_max      = self._k_max.get()
        k_override = self._k_override.get().strip()
        n_init     = self._n_init.get()
        n_refs     = self._n_refs.get()
        norm       = self._norm.get()

        out_root = self._out_dir.get()
        if not out_root:
            out_root = os.path.join(os.path.dirname(self._mice[0][1][0]),
                                    "comparative_analysis")

        params = {
            "mice"      : list(self._mice),
            "method"    : method,
            "k_range"   : range(k_min, k_max + 1),
            "k_override": k_override,
            "n_init"    : n_init,
            "n_refs"    : n_refs,
            "norm"      : norm,
            "out_root"  : out_root,
        }

        self._running = True
        self._btn_run.config(state="disabled", text="Running…")

        # Clear previous results
        for widget in self._results_inner.winfo_children():
            widget.destroy()
        self._result_frames.clear()
        if hasattr(self, "_placeholder"):
            self._placeholder = ttk.Label(
                self._results_inner,
                text="Analysis running…",
                style="Sub.TLabel")
            self._placeholder.pack(pady=40)

        threading.Thread(target=self._analysis_thread,
                         args=(params,), daemon=True).start()

    def _analysis_thread(self, p):
        def log(msg, tag=None):
            self._queue.put({"type": "log", "text": msg, "tag": tag})

        try:
            mice   = p["mice"]
            method = p["method"]
            k_r    = p["k_range"]
            k_ov   = p["k_override"]
            n_init = p["n_init"]
            n_refs = p["n_refs"]
            norm   = p["norm"]
            root   = p["out_root"]
            os.makedirs(root, exist_ok=True)

            # ── Step 1: Classic per mouse ────────────────────────────────
            log("═══ Step 1/3 — Classic analysis ═══")
            classic_results   = {}
            result_csv_paths  = {}

            for mouse_name, csv_files in mice:
                log(f"[Classic] {mouse_name}")
                c_out = os.path.join(root, "classic", mouse_name)
                res   = _run_classic_for_mouse(
                    csv_files, method, k_r, k_ov,
                    n_init, n_refs, c_out, log)
                if res:
                    res["method"] = method
                    classic_results[mouse_name]  = res
                    result_csv_paths[mouse_name] = os.path.join(
                        c_out, "habitats_result.csv")

            # ── Step 2: Discovery (meta-clustering) ──────────────────────
            log("═══ Step 2/3 — Discovery meta-clustering ═══")
            disc_per_mouse = None
            if len(classic_results) >= 2:
                disc_per_mouse = _run_discovery_meta(
                    classic_results, result_csv_paths,
                    method, k_r, k_ov, n_init, log)
                if disc_per_mouse:
                    for v in disc_per_mouse.values():
                        v["method"] = method
            else:
                log("  [Discovery] Skipped — need ≥2 mice")

            # ── Step 3: Joint ────────────────────────────────────────────
            log("═══ Step 3/3 — Joint clustering ═══")
            joint_per_mouse = None
            try:
                mice_list_jt = [{"name": name, "files": files}
                                 for name, files in mice]
                jt_out = os.path.join(root, "joint")
                res_jt = run_joint_pipeline(
                    mice_list_jt, method, k_r, n_init, n_refs,
                    output_dir    = jt_out,
                    log_fn        = log,
                    k_override    = (int(k_ov) if k_ov.isdigit() else None),
                    normalization = norm,
                    dtopo_weight  = DTOPO_JOINT_WEIGHT,
                    dtopo_min_frac= DTOPO_MIN_NECROSIS_FRACTION)

                if res_jt and "df_pooled" in res_jt:
                    df_jt      = res_jt["df_pooled"]
                    jt_params  = res_jt["parameters"]
                    jt_cents   = res_jt.get("centroids")
                    jt_k       = res_jt.get("best_k", len(np.unique(df_jt["Habitat"])))
                    joint_per_mouse = {}
                    for mouse_name, _ in mice:
                        mask = df_jt["Mouse"] == mouse_name
                        if not mask.any():
                            continue
                        sub    = df_jt[mask]
                        lbls   = sub["Habitat"].values
                        feats  = sub[jt_params].values if jt_params else np.empty((0,))
                        joint_per_mouse[mouse_name] = {
                            "labels"    : lbls,
                            "features"  : feats,
                            "parameters": jt_params,
                            "centroids" : jt_cents,
                            "k"         : jt_k,
                            "method"    : method,
                        }
            except Exception as e:
                log(f"  [Joint] Error: {e}", "err")

            # ── Post: send results to GUI ─────────────────────────────────
            self._queue.put({
                "type"            : "results_ready",
                "mice_order"      : [name for name, _ in mice],
                "classic"         : classic_results,
                "discovery"       : disc_per_mouse or {},
                "joint"           : joint_per_mouse or {},
            })
            log("All analyses complete.", "ok")

        except Exception as e:
            log(f"[!] Fatal error: {e}", "err")
        finally:
            self._queue.put({"type": "done"})

    # ── Queue handler ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                task = self._queue.get_nowait()
                kind = task.get("type")

                if kind == "log":
                    self._log(task["text"], task.get("tag"))

                elif kind == "results_ready":
                    self._show_results(task)

                elif kind == "done":
                    self._running = False
                    self._btn_run.config(state="normal",
                                         text="▶  Run Comparative Analysis")
        except Exception:
            pass
        self.after(80, self._poll_queue)

    def _log(self, msg, tag=None):
        self._log_widget.config(state="normal")
        self._log_widget.insert("end", msg + "\n", tag or "")
        self._log_widget.see("end")
        self._log_widget.config(state="disabled")

    # ── Results display ───────────────────────────────────────────────────────

    def _show_results(self, task):
        # Clear placeholder
        for w in self._results_inner.winfo_children():
            w.destroy()
        self._result_frames.clear()

        mice_order = task["mice_order"]
        classic    = task["classic"]
        discovery  = task["discovery"]
        joint      = task["joint"]

        for mouse_name in mice_order:
            cl = classic.get(mouse_name)
            dc = discovery.get(mouse_name)
            jt = joint.get(mouse_name)

            # Section header
            hdr = tk.Frame(self._results_inner, bg=THEME["bg_card"], height=32)
            hdr.pack(fill="x", padx=4, pady=(10, 0))
            hdr.pack_propagate(False)
            tk.Label(hdr,
                     text=f"  {mouse_name}",
                     font=(THEME["font_family"], 10, "bold"),
                     bg=THEME["bg_card"], fg=THEME["text"]
                     ).pack(side="left", pady=4)
            # k-labels
            for label, res, col in [
                ("Classic", cl, _CLASSIC_COL),
                ("Discovery", dc, _DISC_COL),
                ("Joint", jt, _JOINT_COL),
            ]:
                txt = f"{label} K={res['k']}" if res else f"{label} N/A"
                tk.Label(hdr, text=f"  {txt}",
                         font=(THEME["font_family"], 9),
                         bg=THEME["bg_card"], fg=col
                         ).pack(side="left", pady=4)

            # Build and embed the matplotlib figure
            try:
                fig = _build_mouse_figure(mouse_name, cl, dc, jt)
                fig_frame = tk.Frame(self._results_inner, bg=THEME["bg"])
                fig_frame.pack(fill="x", padx=4, pady=2)
                canvas = FigureCanvasTkAgg(fig, master=fig_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="x")
                plt.close(fig)
                self._result_frames.append(fig_frame)
            except Exception as e:
                ttk.Label(self._results_inner,
                          text=f"  [!] Could not render figure for {mouse_name}: {e}",
                          style="Sub.TLabel").pack(anchor="w", padx=_PAD)

            # Separator
            ttk.Separator(self._results_inner).pack(
                fill="x", padx=4, pady=4)

        self._on_inner_configure()
