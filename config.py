# =================================================================
# config.py — Global constants for the tumor habitat pipeline
# Edit this file before launching the app to change defaults.
# =================================================================

# --- Reproducibility ---
RANDOM_SEED = 35   # Fixed seed for all stochastic steps. Set to None to vary between runs.

# --- App theme ---
APP_THEME = 'light'   # 'dark' | 'light'

# --- Clustering range ---
K_RANGE      = range(2, 11)  # Range of K values to test
K_OVERRIDE   = None           # Force a specific K (None = automatic)

# --- Method defaults ---
CLUSTERING_METHOD  = 'gmm'   # 'gmm' | 'srsc' | 'kmeans'
SPATIAL_WEIGHT     = 1.0     # SRSC alpha ∈ [0,1]: W = W_feat * (W_spat ** alpha); 0 = feature only, 1 = classic Hadamard
SIGMA_FEATURE      = None    # SRSC RBF bandwidth (None = median heuristic)
N_INIT_GMM         = 20      # Number of GMM restarts
N_REFS_GAP         = 20      # Number of reference datasets for Gap Statistic
N_JOBS_GAP         = -1      # Parallel jobs for Gap Statistic (-1 = all cores)

# --- Safety guards ---
# GMM: override BIC with Silhouette when Silhouette at BIC-k < threshold
# Set to None to disable
GMM_SILHOUETTE_GUARD = None

# K-means / SRSC: prefer Silhouette elbow over Gap Statistic when they disagree
# Set to False to disable
KMEANS_GAP_GUARD = False

# --- Labeling ---
UNDETERMINED_THRESHOLD = 0.3  # Max |robust-scaled value| to flag as Bulk tumor
# WS labeler abstention: clusters with max label probability below this are
# assigned 'Unknown' instead of being forced into the best-matching label.
# 0.0 = disabled (every cluster gets a label).  Suggested value: 0.35.
WS_ABSTENTION_THR = 0.0

# Percentile used for hybrid normalization offset (P10 of all-ROI voxels).
BRAIN_PERCENTILE = 10

# --- Napari ---
NAPARI_OPACITY = 0.8

# --- Habitat colors (RGBA 0-255) ---
HABITAT_COLORS = [
    (255,   0,   0, 200),   # red
    (  0, 200, 255, 200),   # cyan
    (  0, 255,   0, 200),   # green
    (255, 255,   0, 200),   # yellow
    (255,   0, 255, 200),   # magenta
    (255, 140,   0, 200),   # orange
    (100,   0, 255, 200),   # violet
    (  0, 200, 130, 200),   # spring green
    ( 30,  80, 255, 200),   # blue
    (210,  80,   0, 200),   # burnt sienna
]

# --- Habitat colors (hex, for matplotlib) ---
# Used only for pre-labeling cluster maps (cluster index → color).
HABITAT_COLORS_HEX = [
    '#E85D24',   # burnt orange
    '#3B8BD4',   # blue
    '#2EAA72',   # green
    '#9B59B6',   # purple
    '#F39C12',   # amber
    '#1ABC9C',   # turquoise
    '#E74C3C',   # red
    '#E91E8C',   # hot pink
    '#8BC34A',   # lime green
    '#607D8B',   # blue-grey
]

# --- Family-based semantic colors ---
# One base color per biological family.  Subtypes within the same family
# are shown as tones (dark → light) generated automatically by
# assign_habitat_colors().  Keep this list ordered longest-prefix first
# so habitat_family() matches the most specific name.
_FAMILY_PREFIXES = [
    ("hypoxic/vascular",        "Hypoxic/vascular"),
    ("cellular/proliferative",  "Cellular/proliferative"),
    ("astrocyte barrier",       "Astrocyte barrier"),
    ("infiltrative",            "Infiltrative"),
    ("bulk tumor",              "Bulk tumor"),
    ("near-normal",             "Near-normal"),
    ("necrosis",                "Necrosis"),
    ("unknown",                 "Unknown"),
]

HABITAT_FAMILY_COLORS = {
    "Necrosis"               : "#C0392B",  # red
    "Hypoxic/vascular"       : "#F4D03F",  # yellow
    "Cellular/proliferative" : "#8E44AD",  # purple
    "Astrocyte barrier"      : "#E67E22",  # orange
    "Infiltrative"           : "#27AE60",  # green
    "Bulk tumor"             : "#2471A3",  # blue
    "Near-normal"            : "#AAB7B8",  # light gray
    "Unknown"                : "#7F8C8D",  # dark gray
}
_HABITAT_COLOR_FALLBACK = "#95A5A6"


def habitat_family(label: str) -> str:
    """Return the biological family name for any habitat label string."""
    low = label.strip().lower()
    for prefix, name in _FAMILY_PREFIXES:
        if low.startswith(prefix):
            return name
    return label.strip()


def _make_tones(base_hex: str, n: int) -> list:
    """Return n visually distinct tones of base_hex (lightest to darkest)."""
    import colorsys
    s = base_hex.lstrip('#')
    r, g, b = int(s[0:2], 16)/255, int(s[2:4], 16)/255, int(s[4:6], 16)/255
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    l_lo = max(0.15, l * 0.55)
    l_hi = min(0.85, l * 1.65)
    tones = []
    for i in range(n):
        li = l_lo + (l_hi - l_lo) * (i / max(n - 1, 1))
        rr, gg, bb = colorsys.hls_to_rgb(h, li, sat)
        tones.append('#{:02X}{:02X}{:02X}'.format(
            round(rr * 255), round(gg * 255), round(bb * 255)))
    return tones[::-1]


def assign_habitat_colors(labels) -> dict:
    """Return a {label: hex_color} dict for an iterable of label strings.

    Labels of the same biological family share the same hue.
    When 2+ labels belong to the same family, distinct dark-to-light
    tones are generated automatically so they remain distinguishable.
    """
    from collections import defaultdict as _dd
    seen = list(dict.fromkeys(labels))          # unique, stable order
    groups = _dd(list)
    for lbl in seen:
        fam = habitat_family(lbl)
        if lbl not in groups[fam]:
            groups[fam].append(lbl)
    result = {}
    for fam, lbls in groups.items():
        base = HABITAT_FAMILY_COLORS.get(fam, _HABITAT_COLOR_FALLBACK)
        if len(lbls) == 1:
            result[lbls[0]] = base
        else:
            for lbl, tone in zip(lbls, _make_tones(base, len(lbls))):
                result[lbl] = tone
    return result


def habitat_color(label: str) -> str:
    """Base family color for a single label (no tone generation)."""
    return HABITAT_FAMILY_COLORS.get(habitat_family(label), _HABITAT_COLOR_FALLBACK)

# --- Biological rules for automatic habitat suggestion (Pass 1) ---
#
# Each rule has three sections:
#   required : ALL available conditions must be met, else rule is rejected
#   support  : matched conditions boost the score (tiebreaker)
#   exclude  : if ANY condition is met, rule is rejected outright
#
# Values are robust-scaled (0 = tumor median, unit = IQR).
# Thresholds are (low, high) where None means unbounded.
#
# DTI-MD is used as the primary diffusivity metric. When only DTI-AD and
# DTI-RD are available (DTI-MD dropped from clustering), a virtual MD is
# computed as (AD + 2×RD) / 3 and injected before rules are applied, so
# these thresholds always operate on the same biological quantity.

HABITAT_RULES = [
    {
        "name"       : "Necrosis",
        "description": "T2 and MD very high — free diffusion, tissue destruction (FA near zero)",
        "must_have"  : ["T2", "DTI-MD"],  # cannot confirm necrosis without both relaxometry and diffusion
        "required"   : {
            "T2"    : (0.5,  None),   # T2 very elevated — long relaxation
            "DTI-MD": (0.5,  None),   # MD very elevated — free water diffusion
        },
        "support"    : {
            "DTI-FA": (None, -0.1),   # FA clearly below tumor median — near isotropic
            "T2star": (None,  0.0),   # T2* reduced — possible hemorrhage
        },
        "exclude"    : {},
    },
    {
        "name"       : "Hypoxic/vascular",
        "description": (
            "T2* below tumour median (deoxyHb / iron — CAIX+ hypoxic zones), T2 non-negative. "
            "Threshold relaxed from −0.3 to −0.15 IQR: preclinical T2* is often insufficiently "
            "sensitive to reach −0.3 even in confirmed hypoxic zones. "
            "T2 requirement lowered to 0.0: CAIX+ zones are not necessarily oedematous."
        ),
        "must_have"  : ["T2star"],    # T2* is the defining feature — cannot call hypoxic without it
        "required"   : {
            "T2star": (None, -0.15),  # T2* below tumour median (relaxed: −0.3 missed many CAIX+ zones)
            "T2"    : (0.0,  None),   # T2 at or above tumour median (relaxed: hypoxia ≠ oedema)
        },
        "support"    : {
            "DTI-MD"    : (None, 0.3),     # MD not excessively high — distinguishes from necrosis
            "MTR"       : (None, 0.0),     # MTR at or below tumour median (iron-rich zones tend to be MTR-low)
            "d_topo_norm": (0.10, 0.70),   # penumbra zone: adjacent to core, not far periphery
        },
        "exclude"    : {
            "DTI-MD": (0.5,  None),   # MD very high → necrosis, not hypoxic
        },
        "compound_exclude": [
            # Far from core AND T2* not strongly low → not the hypoxic penumbra
            {"d_topo_norm": (0.80, None), "T2star": (-0.20, None)},
        ],
    },
    {
        "name"       : "Cellular/proliferative",
        "description": (
            "MD below tumour median (restricted diffusion — high cell density), "
            "MTR at or above tumour median (macromolecules / proteins). "
            "MTR threshold lowered from 0.2 to 0.1 IQR: many glioma models show only "
            "modest MTR elevation in Ki-67-high zones. "
            "T2 below tumour median added as support: dense cellularity shortens T2."
        ),
        "must_have"  : ["DTI-MD"],   # restricted diffusion is the primary discriminant
        "required"   : {
            "DTI-MD": (None, 0.0),   # MD below tumour median — restricted diffusion
            "MTR"   : (0.1,  None),  # MTR elevated (relaxed: 0.2 was too strict for some models)
        },
        "support"    : {
            "DTI-FA": (None, 0.0),   # FA reduced — tumour fibre disorganisation
            "T2"    : (None, 0.0),   # T2 below tumour median — dense cellularity
        },
        "exclude"    : {
            "T2"    : (0.5,  None),  # T2 very high → more likely necrosis / infiltrative
        },
    },
    {
        "name"       : "Infiltrative",
        "description": (
            "FA ABOVE tumour median (WM-border fibres preserved relative to disorganised "
            "tumour core), T2 at or above tumour median. "
            "High MD in IQR space alone is NOT excluded — this can reflect normal WM "
            "whose absolute DTI values far exceed the narrow tumour IQR. "
            "Exclusion only fires when MD high AND T2 high simultaneously (true necrosis). "
            "Jellison 2-3 sub-class done within the FA+ zone."
        ),
        "must_have"  : ["DTI-FA"],   # FA is the defining feature of WM infiltration
        "required"   : {
            "T2"    : (0.0,  None),  # T2 at or above tumour median
            "DTI-FA": (0.1,  None),  # FA meaningfully above tumour median
        },
        "support"    : {
            "DTI-MD": (0.0,  0.8),   # MD moderately elevated — not free diffusion
            "DTI-AD": (0.0,  None),  # AD elevated — axonal damage
            "DTI-RD": (0.0,  None),  # RD elevated — demyelination
            "MTR"   : (0.3,  None),  # MTR elevated — myelin-rich WM (strong support)
        },
        "exclude"    : {
            # d_topo_norm < 0.3 → cluster too close to core (necrotic or proxy) → not peripheral
            # d_topo_norm absent → proxy not computed (edge case) → MRI-only fallback
            "d_topo_norm": (None, 0.3),
            # MD below tumour median → restricted diffusion → cellular mass, NOT infiltrating WM
            # Prevents proliferative clusters (low MD) from being mislabelled Infiltrative.
            # Safe when MD absent: condition skipped (p in cd check fails silently).
            "DTI-MD"     : (None, -0.1),
        },
        "compound_exclude": [
            # Exclude if BOTH MD very high AND T2 elevated → true necrosis, not WM
            {"DTI-MD": (0.8, None), "T2": (0.3, None)},
        ],
    },
]

# --- Relative comparison thresholds ---
#
# TWO SPACES:
#   Pass 2 (relative reclassification) operates on per-parameter min-max normed
#   values in [0, 1] so that MD's large IQR does not dominate distance checks.
#   These same constants are reused there — in [0,1] space they are forgiving
#   (SIMILAR = 25% of full inter-cluster range), which is intentional: Pass 2
#   should only reclassify when the signal is unambiguous.
#
#   Pass 3 (within-category sub-classification) operates on raw robust-scaled
#   (IQR) values.  The constants below are calibrated in those IQR units.
#
# If Pass 2 starts over-firing, tighten these values and/or add separate
# RELATIVE_NORM_* constants for the normed space.
RELATIVE_SIMILAR        = 0.25   # |A − B| < SIMILAR     → "similar to"
RELATIVE_SMALLER        = 0.35   # A < B − SMALLER        → "smaller than"
RELATIVE_MUCH_SMALLER   = 0.75   # A < B − MUCH_SMALLER   → "much smaller than"
RELATIVE_BETWEEN_MARGIN = 0.25   # margin from each anchor to define "between the two"

# Minimum voxel coverage fraction for the median anchor to be called "Bulk tumor"
# (vs "Bulk tumor (atypique)" when coverage is below this threshold).
BULK_TUMOR_COVERAGE_MIN = 0.30

# --- V5 : Labeling — continuous scoring ---
# A rule wins only when its score exceeds MIN_RULE_CONFIDENCE.
# Score = 0.60 + 0.40*sup_ratio when ALL required conditions pass;
# lower otherwise. Clusters matching no rule above this threshold default to
# "Bulk tumor" by exclusion, which is more faithful than routing through Unknown.
MIN_RULE_CONFIDENCE = 0.55   # must be < 0.60 so "all-required, no support" still wins
AMBIGUITY_MARGIN    = 0.15   # top-2 rule scores within this → flag ambiguous

# --- V5 : d_topo spatial feature (Step 3) ---
NECROSIS_T2_MIN      = 0.5   # T2 centroid must exceed this (IQR) for guard to pass
NECROSIS_MD_MIN      = 0.5   # MD centroid must exceed this (IQR) for guard to pass
DTOPO_CORE_SEP_MIN   = 0.5   # min (core_score − 2nd_score) for guard to pass
DTOPO_FEATURE_WEIGHT = 1.0   # weight applied to d_topo column before concatenation (per-mouse pipeline)

# --- Joint clustering : d_topo controls ---
# Weight applied to d_topo_norm when it is added to the joint feature matrix.
# 1.0 = equal weight to any MRI feature.
# 0.3 = d_topo acts as a tiebreaker rather than a primary driver.
DTOPO_JOINT_WEIGHT = 0.3

# Per-mouse necrosis fraction threshold for joint clustering.
# If a mouse has fewer than this fraction of its voxels in the necrosis cluster (Pass 1),
# d_topo_norm is set to 0 for that mouse — it does not influence cluster assignment.
# This prevents spatially-driven misclassification in mice with little/no necrosis.
DTOPO_MIN_NECROSIS_FRACTION = 0.05   # 5 % of that mouse's voxels

# --- Hypoxic penumbra spatial prior (Pass-2a) ---
# A cluster labelled Hypoxic/vascular with d_topo_norm above this threshold
# is in the tumour periphery, outside the expected penumbra zone → Bulk tumor.
HYPOX_DTOPO_PRIOR_THR  = 0.75  # Pass-2a reclassification threshold
HYPOX_DTOPO_PERIPH_MAX = 0.80  # compound_exclude in HABITAT_RULES (Pass-1)

# --- V5 : Spatial reporting (Step 4) ---
MIN_COMPONENT_SIZE   = 20    # voxels below this → flagged micro-component
AMBIGUITY_PROBA_THR  = 0.30  # second-best P(label) above this → ambiguous voxel

# --- V5 : MRF light boundary correction (Step 5 — option, off by default) ---
MRF_ENABLED        = False
MRF_LAMBDA         = 0.3    # pairwise weight (keep low to preserve feature signal)
MRF_CONFIDENCE_THR = 0.65   # only voxels with P(assigned) < this are eligible
MRF_N_ITER         = 5

TRANSITION_PENALTY = {
    "Necrosis"              : {"Necrosis": 0.0, "Hypoxic/vascular": 0.0,
                               "Bulk tumor": 2.0, "Cellular/proliferative": 5.0,
                               "Infiltrative": 10.0},
    "Hypoxic/vascular"      : {"Necrosis": 0.0, "Hypoxic/vascular": 0.0,
                               "Bulk tumor": 0.0, "Cellular/proliferative": 2.0,
                               "Infiltrative": 8.0},
    "Bulk tumor"            : {"Necrosis": 2.0, "Hypoxic/vascular": 0.0,
                               "Bulk tumor": 0.0, "Cellular/proliferative": 0.0,
                               "Infiltrative": 2.0},
    "Cellular/proliferative": {"Necrosis": 5.0, "Hypoxic/vascular": 2.0,
                               "Bulk tumor": 0.0, "Cellular/proliferative": 0.0,
                               "Infiltrative": 0.0},
    "Infiltrative"          : {"Necrosis": 10.0, "Hypoxic/vascular": 8.0,
                               "Bulk tumor": 2.0, "Cellular/proliferative": 0.0,
                               "Infiltrative": 0.0},
}

# --- V5 : Hierarchical k=2 segmentation (Step 6 — option, off by default) ---
HIERARCHICAL_ENABLED = False

# Feature selection for hierarchical sub-zones:
# parameters whose intra-zone variance < this ratio × max_variance are dropped.
# They are near-constant within the zone and add noise to sub-clustering.
HIER_FEATURE_VARIANCE_THR = 0.10   # 10 % of max intra-zone variance

# Maximum k for core sub-segmentation (selected by GMM BIC, range 1..HIER_CORE_K_MAX).
# k=1 → homogeneous core (no split); k=4 → up to 4 core sub-populations.
HIER_CORE_K_MAX  = 4
# Maximum k for periphery sub-segmentation (selected by the main pipeline criterion).
HIER_PERIPH_K_MAX = 5

# --- Bio extent score thresholds ---
# score = (raw_value − CL_median) / (tumor_median − CL_median)
#   0 = identical to CL (healthy)
#   1 = identical to tumor median
#  >1 = more altered than tumor median
BIO_EXTENT_THRESHOLDS = {
    'healthy'    : 0.3,   # mean score < 0.3  → near healthy tissue
    'infiltrative': 0.7,  # mean score < 0.7  → halfway CL / tumor
    'viable'     : 1.2,   # mean score < 1.2  → at tumor median level
    'aggressive' : 3.0,   # mean score < 3.0  → beyond tumor median
                           # mean score >= 3.0 → markedly necrotic
}
# If |tumor_median − CL_median| / |CL_median| < this ratio, the extent is
# too small to be meaningful: score is forced to 1.0 (treated as tumor median).
BIO_EXTENT_SMALL_EXTENT_RATIO = 0.05

# --- Method info for the "?" help buttons in the GUI ---
METHOD_INFO = {
    "gmm": {
        "title": "GMM — Gaussian Mixture Model",
        "description": (
         "GMM models the data as a mixture of K Gaussian distributions. "
            "Each voxel receives a probability of belonging to each cluster "
            "(soft assignment), then updates Gaussian parameters (μ, Σ, π ) "
            "to best fit the data and repeat until convergence.\n\n"
            "K selection: BIC (Bayesian information criterion) maximizes likelihood while penalizing model complexity \n "
            "but when there are so few voxels, the “curse of dimensionality” "
            "impacts the relevant aspect of the euclidean distance (over-segmentation)"
            "to maximize likelihood, that’s why we can use the silhouette elbow (spectral space) as a safety guard."
        ),
        "advantages": (
            "- Probabilistic: captures uncertainty at cluster boundaries.\n"
            "- Handles ellipsoidal clusters (not just spherical like K-means).\n"
            "- Fast on CPU for typical ROI sizes."
        ),
        "limits": (
            "- Assumes Gaussian distributions so may fail on skewed MRI data.\n"
            "- BIC can overestimate K when distributions overlap "
            "(especially with high dimension and few data, but you can compare "
            "with the k from silhouette and use  a safety guard if you see a big difference of numbers of cluster.\n"
            "- Sensitive to initialization (you can increase n_init restarts but the comput time will also increase).\n"
            "- No spatial regularization so spatially fragmented results possible."
        ),
    },
   "srsc": {
        "title": "SRSC — Spatially Regularized Spectral Clustering",
        "description": (
            "SRSC builds an affinity matrix combining MRI feature similarity "
            "(RBF kernel) and spatial proximity (Gaussian kernel on X,Y) via a "
            "weighted Hadamard product: W = W_feat × (W_spat ^ alpha).\n\n"
            "alpha=1 → classic gating (full spatial regularization).\n"
            "alpha=0 → pure feature clustering (W_spat disappears).\n"
            "Intermediate values modulate spatial influence continuously.\n\n"
            "K selection: Silhouette elbow + Gap Statistic in spectral space "
            "(first it regroups data like k-means, for k in range, then "
            "generates a reference dataset, computes the loss function "
            "with WSS and the gap statistic, and finally chooses the optimal "
            "k: Gap(k)≥Gap(k+1)−SE(k+1))."
        ),
        "advantages": (
            "- Spatial regularization produces coherent, compact regions.\n"
            "- No Gaussian assumption, so handles arbitrary (even non convex) cluster shapes.\n"
            "- alpha controls spatial influence continuously: 0 = pure feature, 1 = full spatial.\n"
            "- Well suited to MRI habitat mapping (validated in literature)."
        ),
        "limits": (
            "- O(n²) affinity matrix: slow and memory-intensive for large ROIs.\n"
            "- Two parameters to tune: sigma_feature (small σ → only very close voxels are similar) "
            "and alpha (high α → may merge biologically distinct but spatially adjacent zones).\n"
            "- With high-dimensional data with few samples, the Euclidean distances "
            "become less discriminative (curse of dimensionality)."
        ),
    },
    "kmeans": {
        "title": "K-means",
        "description": (
            "Partitions the dataset into k distinct groups based on similarity, then assigns to each point the nearest cluster centroid ( Euclidean distance) and then minimize within-cluster inertia.\n\n"
            "K selection: Silhouette elbow + Gap Statistic "
            "(first it regroups data like k-means, for k in range, then"
           " generate a  reference dataset, then compute the loss fonction"
            "with WSS, and the gap statistique and finally choose the optimal"
            "k Gap(k)≥Gap(k+1)−SE(k+1), in spectral space. "
        ),
        "advantages": (
            "- Fast and simple: good baseline for comparison.\n"
            "- No hyperparameters beyond K.\n"
            "- Silhouette in feature space is directly interpretable."
        ),
        "limits": (
            "- Assumes spherical, equally sized clusters.\n"
            "- Sensitive to outliers (extreme voxels can distort centroids).\n"
            "- No probabilistic assignment (unlike the others): hard boundaries only.\n"
            "- Gap Statistic may overestimate K on monotone score curves (so you can use silhouette as safety guard)"
        ),
    },
}

# --- Small-cluster size guard ---
# Clusters whose voxel fraction is below this threshold are considered "too
# small" and trigger a k-reduction — UNLESS the WS labeler predicts Necrosis
# or Infiltrative (both are biologically relevant even at low volume).
MIN_CLUSTER_SIZE_RATIO       = 0.05   # 5 % of total tumour voxels (per-mouse pipeline)
JOINT_MIN_CLUSTER_SIZE_RATIO = 0.03   # 3 % of pooled voxels (joint clustering — lower because N mice pooled)

# --- Inter-animal comparison ---
COMPARISON_PARAMETERS    = ['DTI-AD', 'DTI-FA', 'DTI-MD', 'DTI-RD', 'T2star', 'MTR', 'T2']
DIRECTION_THRESHOLD_RATIO = 0.15   # adaptive threshold = ratio × max centroid norm per habitat
COSINE_THRESHOLD          = 0.80   # min cosine similarity to consider habitats identical
EUCLIDEAN_THRESHOLD       = 0.5    # max |centroid_A[p] - centroid_B[p]| (in IQR) to consider parameters close
