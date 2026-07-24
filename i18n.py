# =================================================================
# i18n.py — Internationalisation (English / French / Spanish)
# Usage:
#   from i18n import t, display_label, set_lang, get_lang
#   t("browse")              → "Browse" | "Parcourir" | "Explorar"
#   display_label("Necrosis (cystic)") → "Necrosis (cystic)" | "Nécrose (kystique)" | "Necrosis (quística)"
# =================================================================

_LANG = "en"   # active language; changed at runtime by the GUI

# ---------------------------------------------------------------------------
# UI / report strings
# ---------------------------------------------------------------------------
_TR = {
    # ── App title ───────────────────────────────────────────────────────────
    "app_title"            : {"en": "Tumor Habitat Segmentation  ·  V2",
                               "fr": "Segmentation des Habitats Tumoraux  ·  V2",
                               "es": "Segmentación de Hábitats Tumorales  ·  V2"},
    "header_title"         : {"en": "Tumor Habitat Segmentation",
                               "fr": "Segmentation des Habitats Tumoraux",
                               "es": "Segmentación de Hábitats Tumorales"},

    # ── Language button (shows next language in cycle en→fr→es→en) ──────────
    "lang_btn"             : {"en": "FR", "fr": "ES", "es": "EN"},

    # ── Buttons ─────────────────────────────────────────────────────────────
    "browse"               : {"en": "Browse",            "fr": "Parcourir",          "es": "Explorar"},
    "run_analysis"         : {"en": "▶  Run analysis",   "fr": "▶  Lancer l'analyse", "es": "▶  Ejecutar análisis"},
    "open_napari"          : {"en": "Open in Napari",    "fr": "Ouvrir dans Napari", "es": "Abrir en Napari"},
    "close"                : {"en": "Close",             "fr": "Fermer",             "es": "Cerrar"},
    "add_mouse"            : {"en": "+ Add mouse",       "fr": "+ Ajouter souris",   "es": "+ Añadir ratón"},
    "run_ia"               : {"en": "▶  Run inter-animal comparison",
                               "fr": "▶  Lancer comparaison inter-animal",
                               "es": "▶  Ejecutar comparación inter-animal"},
    "run_im"               : {"en": "▶  Run inter-method comparison",
                               "fr": "▶  Lancer comparaison inter-méthode",
                               "es": "▶  Ejecutar comparación inter-método"},
    "running"              : {"en": "⏳  Running…",      "fr": "⏳  En cours…",      "es": "⏳  Ejecutando…"},
    "confirm"              : {"en": "Confirm",           "fr": "Confirmer",          "es": "Confirmar"},

    # ── Tabs ────────────────────────────────────────────────────────────────
    "tab_files"            : {"en": "  Files  ",         "fr": "  Fichiers  ",       "es": "  Archivos  "},
    "tab_method"           : {"en": "  Method  ",        "fr": "  Méthode  ",        "es": "  Método  "},
    "tab_output"           : {"en": "  Output  ",        "fr": "  Sortie  ",         "es": "  Salida  "},
    "tab_comparison"       : {"en": "  Comparison  ",    "fr": "  Comparaison  ",    "es": "  Comparación  "},
    "tab_ia"               : {"en": "  Inter-animal  ",  "fr": "  Inter-animal  ",   "es": "  Inter-animal  "},
    "tab_im"               : {"en": "  Inter-method  ",  "fr": "  Inter-méthode  ",  "es": "  Inter-método  "},

    # ── Files tab ───────────────────────────────────────────────────────────
    "csv_section"          : {"en": "CSV files — qMRI parameters",
                               "fr": "Fichiers CSV — paramètres qMRI",
                               "es": "Archivos CSV — parámetros qMRI"},
    "no_file"              : {"en": "No file selected",
                               "fr": "Aucun fichier sélectionné",
                               "es": "Ningún archivo seleccionado"},
    "n_files_selected"     : {"en": "{n} file(s) selected",
                               "fr": "{n} fichier(s) sélectionné(s)",
                               "es": "{n} archivo(s) seleccionado(s)"},
    "nifti_section"        : {"en": "NIfTI reference image (optional)",
                               "fr": "Image NIfTI de référence (optionnel)",
                               "es": "Imagen NIfTI de referencia (opcional)"},

    # ── Method tab ──────────────────────────────────────────────────────────
    "method_section"       : {"en": "Clustering method",
                               "fr": "Méthode de clustering",
                               "es": "Método de agrupamiento"},
    "gmm_desc"             : {"en": "GMM  —  Gaussian Mixture Model (recommended)",
                               "fr": "GMM  —  Modèle de Mélange Gaussien (recommandé)",
                               "es": "GMM  —  Modelo de mezcla gaussiana (recomendado)"},
    "srsc_desc"            : {"en": "SRSC  —  Spectral + spatial regularization",
                               "fr": "SRSC  —  Spectral + régularisation spatiale",
                               "es": "SRSC  —  Espectral + regularización espacial"},
    "kmeans_desc"          : {"en": "K-means  —  K-means",
                               "fr": "K-means  —  K-means",
                               "es": "K-means  —  K-means"},
    "common_params"        : {"en": "Common parameters",
                               "fr": "Paramètres communs",
                               "es": "Parámetros comunes"},
    "k_min"                : {"en": "K range (min)",  "fr": "Plage K (min)",  "es": "Rango K (mín.)"},
    "k_max"                : {"en": "K range (max)",  "fr": "Plage K (max)",  "es": "Rango K (máx.)"},
    "k_override"           : {"en": "K override (empty = auto)",
                               "fr": "K forcé (vide = auto)",
                               "es": "K forzado (vacío = auto)"},
    "srsc_params"          : {"en": "SRSC parameters",  "fr": "Paramètres SRSC",  "es": "Parámetros SRSC"},
    "spatial_weight"       : {"en": "Spatial weight (alpha)",
                               "fr": "Poids spatial (alpha)",
                               "es": "Peso espacial (alpha)"},
    "sigma_feature"        : {"en": "sigma_feature (auto)", "fr": "sigma_feature (auto)", "es": "sigma_feature (auto)"},
    "gmm_params"           : {"en": "GMM parameters",  "fr": "Paramètres GMM",  "es": "Parámetros GMM"},
    "n_init_gmm"           : {"en": "N init GMM",  "fr": "N init GMM",  "es": "N inicios GMM"},
    "gmm_guard"            : {"en": "Safety guard — override BIC if Silhouette < {thr}",
                               "fr": "Garde — ignorer BIC si Silhouette < {thr}",
                               "es": "Guardia de seguridad — anular BIC si Silhouette < {thr}"},
    "n_refs"               : {"en": "N refs (Gap Statistic)",
                               "fr": "N refs (Statistique Gap)",
                               "es": "N refs (Estadística Gap)"},
    "kmeans_guard"         : {"en": "Safety guard — prefer Silhouette elbow over Gap if they disagree",
                               "fr": "Garde — préférer coude Silhouette sur Gap si désaccord",
                               "es": "Guardia de seguridad — preferir codo de Silhouette sobre Gap"},
    "spatial_options"      : {"en": "Spatial options (V5)",
                               "fr": "Options spatiales (V5)",
                               "es": "Opciones espaciales (V5)"},
    "hierarchical_opt"     : {"en": "Hierarchical k=2 segmentation  (core / periphery split, then sub-segment each zone)",
                               "fr": "Segmentation hiérarchique k=2  (séparation cœur / périphérie, puis sous-segmentation)",
                               "es": "Segmentación jerárquica k=2  (división núcleo / periferia, luego sub-segmentación)"},
    "mrf_opt"              : {"en": "MRF light boundary correction  (GMM only — corrects uncertain border voxels)",
                               "fr": "Correction MRF légère des frontières  (GMM uniquement — corrige les voxels incertains de bord)",
                               "es": "Corrección MRF ligera de bordes  (solo GMM — corrige vóxeles de borde inciertos)"},
    "progress"             : {"en": "Progress",  "fr": "Progression",  "es": "Progreso"},

    # ── Output tab ──────────────────────────────────────────────────────────
    "output_folder"        : {"en": "Output folder name",
                               "fr": "Nom du dossier de sortie",
                               "es": "Nombre de carpeta de salida"},
    "output_folder_hint"   : {"en": "e.g. 'V2'  →  created next to the CSV files",
                               "fr": "ex. 'V2'  →  créé à côté des fichiers CSV",
                               "es": "ej. 'V2'  →  creado junto a los archivos CSV"},
    "output_dir"           : {"en": "Custom output directory (optional)",
                               "fr": "Répertoire de sortie personnalisé (optionnel)",
                               "es": "Directorio de salida personalizado (opcional)"},
    "output_dir_hint"      : {"en": "If empty, the folder is created next to the CSV files.",
                               "fr": "Si vide, le dossier est créé à côté des fichiers CSV.",
                               "es": "Si está vacío, la carpeta se crea junto a los archivos CSV."},

    # ── Comparison tab — inter-animal ───────────────────────────────────────
    "ia_animals"           : {"en": "Animals",    "fr": "Animaux",        "es": "Animales"},
    "mouse_n"              : {"en": "Mouse {n}",  "fr": "Souris {n}",     "es": "Ratón {n}"},
    "no_files"             : {"en": "No files",   "fr": "Aucun fichier",  "es": "Sin archivos"},
    "n_files"              : {"en": "{n} file(s)", "fr": "{n} fichier(s)", "es": "{n} archivo(s)"},
    "ia_output_hint"       : {"en": "If empty, created next to first mouse's CSV files.",
                               "fr": "Si vide, créé à côté des fichiers CSV de la première souris.",
                               "es": "Si está vacío, se crea junto a los archivos CSV del primer ratón."},
    "gmm_guard_short"      : {"en": "GMM safety guard (Silhouette < {thr})",
                               "fr": "Garde GMM (Silhouette < {thr})",
                               "es": "Guardia GMM (Silhouette < {thr})"},
    "kmeans_guard_short"   : {"en": "K-means safety guard (prefer Silhouette elbow)",
                               "fr": "Garde K-means (préférer coude Silhouette)",
                               "es": "Guardia K-means (preferir codo de Silhouette)"},
    "hierarchical_short"   : {"en": "Hierarchical k=2 segmentation",
                               "fr": "Segmentation hiérarchique k=2",
                               "es": "Segmentación jerárquica k=2"},
    "mrf_short"            : {"en": "MRF light boundary correction (GMM only)",
                               "fr": "Correction MRF légère des frontières (GMM uniquement)",
                               "es": "Corrección MRF ligera de bordes (solo GMM)"},
    "srsc_spatial_weight"  : {"en": "SRSC spatial weight",
                               "fr": "Poids spatial SRSC",
                               "es": "Peso espacial SRSC"},
    "gmm_n_init"           : {"en": "GMM n_init",    "fr": "GMM n_init",          "es": "GMM n_inicio"},
    "folder_name"          : {"en": "Folder name",   "fr": "Nom du dossier",       "es": "Nombre de carpeta"},
    "output_section"       : {"en": "Output",        "fr": "Sortie",               "es": "Salida"},
    "parameters_section"   : {"en": "Parameters",    "fr": "Paramètres",           "es": "Parámetros"},

    # ── Comparison tab — inter-method ───────────────────────────────────────
    "im_csv_section"       : {"en": "CSV files — single animal",
                               "fr": "Fichiers CSV — animal unique",
                               "es": "Archivos CSV — un solo animal"},
    "im_methods_section"   : {"en": "Methods to compare",
                               "fr": "Méthodes à comparer",
                               "es": "Métodos a comparar"},
    "im_output_hint"       : {"en": "If empty, created next to the CSV files.",
                               "fr": "Si vide, créé à côté des fichiers CSV.",
                               "es": "Si está vacío, se crea junto a los archivos CSV."},

    # ── File dialogs ────────────────────────────────────────────────────────
    "dlg_select_csv"       : {"en": "Select CSV files",
                               "fr": "Sélectionner les fichiers CSV",
                               "es": "Seleccionar archivos CSV"},
    "dlg_select_nifti"     : {"en": "Select the NIfTI reference image",
                               "fr": "Sélectionner l'image NIfTI de référence",
                               "es": "Seleccionar imagen NIfTI de referencia"},
    "dlg_select_output"    : {"en": "Choose output directory",
                               "fr": "Choisir le répertoire de sortie",
                               "es": "Elegir directorio de salida"},
    "dlg_select_csv_mouse" : {"en": "Select CSV files for this mouse",
                               "fr": "Sélectionner les fichiers CSV pour cette souris",
                               "es": "Seleccionar archivos CSV para este ratón"},

    # ── Error messages ──────────────────────────────────────────────────────
    "err_title"            : {"en": "Error",          "fr": "Erreur",       "es": "Error"},
    "err_min_mice"         : {"en": "Add at least 2 mice.",
                               "fr": "Ajoutez au moins 2 souris.",
                               "es": "Añada al menos 2 ratones."},
    "err_mouse_no_csv"     : {"en": "Every mouse needs at least one CSV file.",
                               "fr": "Chaque souris doit avoir au moins un fichier CSV.",
                               "es": "Cada ratón necesita al menos un archivo CSV."},
    "err_k_range"          : {"en": "K min must be less than K max.",
                               "fr": "K min doit être inférieur à K max.",
                               "es": "K mín. debe ser menor que K máx."},
    "err_no_csv"           : {"en": "Please select at least one CSV file.",
                               "fr": "Veuillez sélectionner au moins un fichier CSV.",
                               "es": "Seleccione al menos un archivo CSV."},
    "err_min_methods"      : {"en": "Select at least 2 methods to compare.",
                               "fr": "Sélectionnez au moins 2 méthodes à comparer.",
                               "es": "Seleccione al menos 2 métodos para comparar."},

    # ── Method help dialog ──────────────────────────────────────────────────
    "help_what"            : {"en": "What is this?",
                               "fr": "Qu'est-ce que c'est ?",
                               "es": "¿Qué es esto?"},
    "help_why"             : {"en": "Why does it work well?",
                               "fr": "Pourquoi fonctionne-t-il bien ?",
                               "es": "¿Por qué funciona bien?"},
    "help_watch"           : {"en": "What should you watch out for?",
                               "fr": "À quoi faut-il faire attention ?",
                               "es": "¿Qué hay que tener en cuenta?"},
    "help_method_info"     : {"en": "Method info",
                               "fr": "Info méthode",
                               "es": "Información del método"},

    # ── Label dialog (labeling.py) ──────────────────────────────────────────
    "label_dlg_title"      : {"en": "Name the habitats",
                               "fr": "Nommer les habitats",
                               "es": "Nombrar los hábitats"},
    "label_dlg_header"     : {"en": "Name each habitat",
                               "fr": "Nommer chaque habitat",
                               "es": "Nombre cada hábitat"},
    "label_dlg_subtitle_bio": {
        "en": "Level 1: vs tumor (robust)   ·   Level 2: bio extent (0 = healthy, 1 = tumor median)",
        "fr": "Niveau 1 : vs tumeur (robuste)   ·   Niveau 2 : étendue biol. (0 = sain, 1 = médiane tumeur)",
        "es": "Nivel 1: vs tumor (robusto)   ·   Nivel 2: ext. biológica (0 = sano, 1 = mediana tumor)",
    },
    "label_dlg_subtitle_simple": {
        "en": "Values > 0: above tumor median   ·   Values < 0: below tumor median",
        "fr": "Valeurs > 0 : au-dessus de la médiane   ·   Valeurs < 0 : en dessous de la médiane",
        "es": "Valores > 0: por encima de la mediana   ·   Valores < 0: por debajo de la mediana",
    },
    "label_vs_tumor"       : {"en": "?  vs tumor",       "fr": "?  vs tumeur",         "es": "?  vs tumor"},
    "label_bio_extent"     : {"en": "?  bio extent",     "fr": "?  étendue biol.",      "es": "?  ext. biológica"},
    "label_cluster"        : {"en": "Cluster {n}",       "fr": "Cluster {n}",           "es": "Clúster {n}"},
    "label_most_agg"       : {"en": "▲ most aggressive", "fr": "▲ plus agressif",       "es": "▲ más agresivo"},
    "label_vs_tumor_row"   : {"en": "vs tumor (robust)  : ",
                               "fr": "vs tumeur (robuste): ",
                               "es": "vs tumor (robusto): "},
    "label_bio_row"        : {"en": "bio extent (0→1)   : ",
                               "fr": "étendue biol.(0→1): ",
                               "es": "ext. biol. (0→1)  : "},
    "label_severity_row"   : {"en": "bio severity       : ",
                               "fr": "sévérité biol.    : ",
                               "es": "sev. biológica    : "},
    "label_pass1_row"      : {"en": "pass 1",    "fr": "passe 1",   "es": "paso 1"},
    "label_reclass"        : {"en": "⟲ reclassified  : ", "fr": "⟲ reclassifié   : ", "es": "⟲ reclasificado : "},
    "label_subcat"         : {"en": "↳ subclass      : ", "fr": "↳ sous-classe   : ",  "es": "↳ subclase      : "},
    "label_warn_small"     : {"en": "⚠ small extent: ", "fr": "⚠ étendue faible: ", "es": "⚠ ext. pequeña: "},
    "label_confidence_high": {"en": "high",    "fr": "haute",   "es": "alta"},
    "label_confidence_med" : {"en": "medium",  "fr": "moyenne", "es": "media"},
    "label_confidence_low" : {"en": "low",     "fr": "faible",  "es": "baja"},

    # ── PDF report (compare_v4.py) ──────────────────────────────────────────
    "pdf_title"            : {"en": "Habitat Comparison Report — V4",
                               "fr": "Rapport de Comparaison des Habitats — V4",
                               "es": "Informe de Comparación de Hábitats — V4"},
    "pdf_params"           : {"en": "Parameters : {p}",
                               "fr": "Paramètres : {p}",
                               "es": "Parámetros : {p}"},
    "pdf_size_floor"       : {"en": "Size floor : >{n} vox",
                               "fr": "Taille min : >{n} vox",
                               "es": "Tamaño mínimo : >{n} vox"},
    "pdf_eta2_floor"       : {"en": "η² floor : ≥{n}",
                               "fr": "η² min : ≥{n}",
                               "es": "η² mínimo : ≥{n}"},
    "pdf_cosine_thr"       : {"en": "Cosine threshold : ≥{n}",
                               "fr": "Seuil cosinus : ≥{n}",
                               "es": "Umbral coseno : ≥{n}"},
    "pdf_global_sil"       : {"en": "Global silhouette (centroid-level, cosine) : {v:.3f}",
                               "fr": "Silhouette globale (niveau centroïde, cosinus) : {v:.3f}",
                               "es": "Silueta global (nivel centroide, coseno) : {v:.3f}"},
    "pdf_coherence_title"  : {"en": "Label coherence summary",
                               "fr": "Résumé de cohérence des étiquettes",
                               "es": "Resumen de coherencia de etiquetas"},
    "pdf_concordant"       : {"en": "concordant",    "fr": "concordant",         "es": "concordante"},
    "pdf_divergent_shape"  : {"en": "divergent shape: ", "fr": "forme divergente: ", "es": "forma divergente: "},
    "pdf_divergent"        : {"en": "divergent: ",   "fr": "divergent: ",        "es": "divergente: "},
    "pdf_near_origin"      : {"en": "near-origin",   "fr": "proche de l'origine", "es": "cerca del origen"},
    "pdf_internally_div"   : {"en": "internally divergent",
                               "fr": "divergent en interne",
                               "es": "divergente internamente"},
    "pdf_coherent_yes"     : {"en": "✓ YES",   "fr": "✓ OUI",  "es": "✓ SÍ"},
    "pdf_coherent_no"      : {"en": "  no ",   "fr": "  non ", "es": "  no "},

    # ── F-figure titles ─────────────────────────────────────────────────────
    "f1_title"             : {"en": "F1 — Centroid heatmap by label",
                               "fr": "F1 — Carte de chaleur des centroïdes par étiquette",
                               "es": "F1 — Mapa de calor de centroides por etiqueta"},
    "f2_title"             : {"en": "F2 — Pairwise cosine similarity matrix",
                               "fr": "F2 — Matrice de similarité cosinus par paire",
                               "es": "F2 — Matriz de similitud coseno por pares"},
    "f3_title"             : {"en": "F3 — Intra / inter dispersion ratio (relative contrast)",
                               "fr": "F3 — Rapport dispersion intra/inter (contraste relatif)",
                               "es": "F3 — Razón dispersión intra/inter (contraste relativo)"},
    "f4_title"             : {"en": "F4 — Bootstrap centroid CI  (95%, voxel resampling)",
                               "fr": "F4 — IC centroïdes bootstrap  (95%, rééchantillonnage)",
                               "es": "F4 — IC de centroides bootstrap (95%, remuestreo)"},
    "f5_title"             : {"en": "F5 — QC table",
                               "fr": "F5 — Table QC",
                               "es": "F5 — Tabla de control de calidad"},
    "f6_title"             : {"en": "F6 — Output components  (C1: inter-source coherence  vs  C2: intra-source quality)",
                               "fr": "F6 — Composantes de sortie  (C1 : cohérence inter-source  vs  C2 : qualité intra-source)",
                               "es": "F6 — Componentes de salida  (C1: coherencia inter-fuente  vs  C2: calidad intra-fuente)"},
    "f7_title"             : {"en": "F7 — Radar charts per label",
                               "fr": "F7 — Graphiques radar par étiquette",
                               "es": "F7 — Gráficos radar por etiqueta"},
    "f8_title"             : {"en": "F8 — Habitat segmentation maps  (common colour legend)",
                               "fr": "F8 — Cartes de segmentation des habitats  (légende couleur commune)",
                               "es": "F8 — Mapas de segmentación de hábitats  (leyenda de color común)"},
    "f9_title"             : {"en": "F9 — CL-relative archetypes per label  (common reference space)",
                               "fr": "F9 — Archétypes CL-relatifs par étiquette  (espace de référence commun)",
                               "es": "F9 — Arquetipos CL-relativos por etiqueta  (espacio de referencia común)"},

    # ── Label dialog info popups ─────────────────────────────────────────────
    "info_vs_tumor_title"  : {"en": "vs tumor  (robust scaling)",
                               "fr": "vs tumeur  (mise à l'échelle robuste)",
                               "es": "vs tumor  (escalado robusto)"},
    "info_bio_title"       : {"en": "Bio extent score  (0 → 1)",
                               "fr": "Score d'étendue biologique  (0 → 1)",
                               "es": "Puntuación ext. biológica  (0 → 1)"},
    "label_field"          : {"en": "Label:",  "fr": "Étiquette :",  "es": "Etiqueta:"},

    "vs_tumor_info"        : {
        "fr": (
            "Mise à l'échelle robuste  (référence intra-tumorale)\n"
            "───────────────────────────────────────────────\n"
            "Formule :  score = (valeur − médiane_tumeur) / IQR\n\n"
            "Compare chaque cluster à la MÉDIANE TUMORALE.\n"
            "L'IQR (écart interquartile) sert d'échelle,\n"
            "le rendant robuste aux voxels extrêmes\n"
            "(nécrose, calcifications, artefacts).\n\n"
            " +1.0  →  1 IQR au-dessus de la médiane tumorale\n"
            "  0.0  →  à la médiane tumorale\n"
            " −1.0  →  1 IQR en dessous de la médiane tumorale\n\n"
            "Seuil : |score| < 0,3  →  Tumeur principale\n\n"
            "Répond à : « Où se situe ce cluster\n"
            "dans la distribution propre de la tumeur ? »\n"
            "Utilisé pour identifier le TYPE D'HABITAT\n"
            "(Nécrose, Infiltratif, Cellulaire…)."
        ),
        "en": (
            "Robust scaling  (intra-tumor reference)\n"
            "───────────────────────────────────────────────\n"
            "Formula:  score = (value − tumor_median) / IQR\n\n"
            "Compares each cluster to the TUMOR MEDIAN.\n"
            "IQR (interquartile range) is used as scale,\n"
            "making it robust to extreme voxels\n"
            "(necrosis, calcifications, artifacts).\n\n"
            " +1.0  →  1 IQR above tumor median\n"
            "  0.0  →  at tumor median\n"
            " −1.0  →  1 IQR below tumor median\n\n"
            "Threshold: |score| < 0.3  →  Bulk tumor\n\n"
            "Answers: \"Where does this cluster sit\n"
            "within the tumor's own distribution?\"\n"
            "Used to identify the HABITAT TYPE\n"
            "(Necrosis, Infiltrative, Cellular…)."
        ),
        "es": (
            "Escalado robusto  (referencia intra-tumoral)\n"
            "───────────────────────────────────────────────\n"
            "Fórmula:  puntuación = (valor − mediana_tumor) / IQR\n\n"
            "Compara cada clúster con la MEDIANA TUMORAL.\n"
            "El IQR (rango intercuartílico) se usa como escala,\n"
            "haciéndolo robusto ante vóxeles extremos\n"
            "(necrosis, calcificaciones, artefactos).\n\n"
            " +1.0  →  1 IQR por encima de la mediana tumoral\n"
            "  0.0  →  en la mediana tumoral\n"
            " −1.0  →  1 IQR por debajo de la mediana tumoral\n\n"
            "Umbral: |puntuación| < 0.3  →  Tumor sólido\n\n"
            "Responde: \"¿Dónde se sitúa este clúster\n"
            "dentro de la distribución del propio tumor?\"\n"
            "Usado para identificar el TIPO DE HÁBITAT\n"
            "(Necrosis, Infiltrativo, Celular…)."
        ),
    },

    "bio_extent_info"      : {
        "fr": (
            "Score d'étendue biologique  (normalisé CL)\n"
            "───────────────────────────────────────────────\n"
            "Formule :\n"
            "  score = (raw − CL_médiane) / (tumeur_médiane − CL_médiane)\n\n"
            "Situe chaque cluster sur le continuum allant du\n"
            "tissu SAIN (CL) à la MÉDIANE TUMORALE.\n\n"
            "  0,0  →  identique au tissu sain (CL)\n"
            "  1,0  →  identique à la médiane tumorale\n"
            " >1,0  →  plus altéré que la médiane tumorale\n\n"
            "Guide d'interprétation (score moyen) :\n"
            "  < 0,3       proche du tissu sain\n"
            "  0,3 – 0,7   infiltratif / bord péri-tumoral\n"
            "  0,7 – 1,2   au niveau tumoral (tumeur active)\n"
            "  1,2 – 3,0   au-delà de la médiane tumorale (agressif)\n"
            "  ≥ 3,0       nettement au-delà → nécrose\n\n"
            "⚠ Indicateur « étendue faible » : affiché quand la différence\n"
            "  entre tumeur_médiane et CL_médiane est < 5 %\n"
            "  de CL_médiane. Score forcé à 1,0 (paramètre\n"
            "  non discriminant — traité comme médiane tumorale).\n\n"
            "Répond à : « À quelle distance ce cluster est-il\n"
            "du tissu sain, en unités tumorales ? »"
        ),
        "en": (
            "Bio extent score  (CL-normalized)\n"
            "───────────────────────────────────────────────\n"
            "Formula:\n"
            "  score = (raw − CL_median) / (tumor_median − CL_median)\n\n"
            "Maps each cluster on the continuum from\n"
            "HEALTHY tissue (CL) to TUMOR MEDIAN.\n\n"
            "  0.0  →  identical to healthy tissue (CL)\n"
            "  1.0  →  identical to tumor median\n"
            " >1.0  →  more altered than the tumor median\n\n"
            "Interpretation guide (mean score):\n"
            "  < 0.3       close to healthy tissue\n"
            "  0.3 – 0.7   infiltrative / peri-tumoral border\n"
            "  0.7 – 1.2   at tumor level (active tumor)\n"
            "  1.2 – 3.0   beyond tumor median (aggressive)\n"
            "  ≥ 3.0       markedly beyond median → necrosis\n\n"
            "⚠ \"small extent\" flag: shown when the difference\n"
            "  between tumor_median and CL_median is < 5 %\n"
            "  of CL_median. Score forced to 1.0 (parameter\n"
            "  not discriminative — treated as tumor median).\n\n"
            "Answers: \"How far is this cluster from\n"
            "healthy tissue, in tumor units?\""
        ),
        "es": (
            "Puntuación de extensión biológica  (normalizada CL)\n"
            "───────────────────────────────────────────────\n"
            "Fórmula:\n"
            "  puntuación = (raw − CL_mediana) / (tumor_mediana − CL_mediana)\n\n"
            "Sitúa cada clúster en el continuo desde\n"
            "tejido SANO (CL) hasta la MEDIANA TUMORAL.\n\n"
            "  0.0  →  idéntico al tejido sano (CL)\n"
            "  1.0  →  idéntico a la mediana tumoral\n"
            " >1.0  →  más alterado que la mediana tumoral\n\n"
            "Guía de interpretación (puntuación media):\n"
            "  < 0.3       cercano al tejido sano\n"
            "  0.3 – 0.7   infiltrativo / borde peri-tumoral\n"
            "  0.7 – 1.2   al nivel tumoral (tumor activo)\n"
            "  1.2 – 3.0   más allá de la mediana tumoral (agresivo)\n"
            "  ≥ 3.0       marcadamente más allá → necrosis\n\n"
            "⚠ Indicador \"ext. pequeña\": se muestra cuando la diferencia\n"
            "  entre tumor_mediana y CL_mediana es < 5 %\n"
            "  de CL_mediana. Puntuación forzada a 1.0 (parámetro\n"
            "  no discriminativo — tratado como mediana tumoral).\n\n"
            "Responde: \"¿Qué tan lejos está este clúster del\n"
            "tejido sano, en unidades tumorales?\""
        ),
    },

    # ── Bio extent interpretation ────────────────────────────────────────────
    "bio_near_normal"       : {"en": "Near-normal",         "fr": "Quasi-normal",         "es": "Casi normal"},
    "bio_infiltrative_lbl"  : {"en": "Infiltrative",        "fr": "Infiltratif",          "es": "Infiltrativo"},
    "bio_active_tumor"      : {"en": "Active tumor",        "fr": "Tumeur active",        "es": "Tumor activo"},
    "bio_aggressive"        : {"en": "Aggressive/Necrotic", "fr": "Agressif/Nécrotique",  "es": "Agresivo/Necrótico"},
    "bio_necrosis_lbl"      : {"en": "Necrosis",            "fr": "Nécrose",              "es": "Necrosis"},
    "bio_level_healthy"     : {
        "en": "close to healthy tissue (mean score = {score:.1f})",
        "fr": "proche du tissu sain (score moyen = {score:.1f})",
        "es": "cercano al tejido sano (puntuación media = {score:.1f})"},
    "bio_level_infiltrative": {
        "en": "halfway between CL and tumor (mean score = {score:.1f}) — likely infiltrative tissue",
        "fr": "à mi-chemin entre CL et tumeur (score moyen = {score:.1f}) — tissu probablement infiltratif",
        "es": "a mitad de camino entre CL y tumor (puntuación media = {score:.1f}) — probable tejido infiltrativo"},
    "bio_level_viable"      : {
        "en": "near tumor median (mean score = {score:.1f}) — active tumor tissue",
        "fr": "proche de la médiane tumorale (score moyen = {score:.1f}) — tissu tumoral actif",
        "es": "cerca de la mediana tumoral (puntuación media = {score:.1f}) — tejido tumoral activo"},
    "bio_level_aggressive"  : {
        "en": "beyond tumor median (mean score = {score:.1f}) — likely necrosis",
        "fr": "au-delà de la médiane tumorale (score moyen = {score:.1f}) — probablement nécrose",
        "es": "más allá de la mediana tumoral (puntuación media = {score:.1f}) — probable necrosis"},
    "bio_level_necrosis"    : {
        "en": "markedly beyond tumor median (mean score = {score:.1f}) — necrosis",
        "fr": "nettement au-delà de la médiane tumorale (score moyen = {score:.1f}) — nécrose",
        "es": "marcadamente más allá de la mediana tumoral (puntuación media = {score:.1f}) — necrosis"},

    # ── Log messages ─────────────────────────────────────────────────────────
    "log_napari_ready"      : {
        "en": "-> Click 'Open Napari' to visualize results.",
        "fr": "-> Cliquez sur 'Ouvrir dans Napari' pour visualiser les résultats.",
        "es": "-> Haga clic en 'Abrir en Napari' para visualizar los resultados."},
    "log_labels_saved"      : {
        "en": "Labels saved : {path}",
        "fr": "Étiquettes sauvegardées : {path}",
        "es": "Etiquetas guardadas : {path}"},
    "log_ia_start"          : {
        "en": "\n--- Inter-animal comparison ({n} mice) ---",
        "fr": "\n--- Comparaison inter-animal ({n} souris) ---",
        "es": "\n--- Comparación inter-animal ({n} ratones) ---"},
    "log_mouse_loading"     : {
        "en": "\n[Mouse {n}] Loading data...",
        "fr": "\n[Souris {n}] Chargement des données...",
        "es": "\n[Ratón {n}] Cargando datos..."},
    "log_mouse_optimal_k"   : {
        "en": "[Mouse {n}] Optimal K = {k}",
        "fr": "[Souris {n}] K optimal = {k}",
        "es": "[Ratón {n}] K óptima = {k}"},
    "log_mouse_done"        : {
        "en": "[Mouse {n}] Done → {path}",
        "fr": "[Souris {n}] Terminé → {path}",
        "es": "[Ratón {n}] Listo → {path}"},
    "log_comparison_saved"  : {
        "en": "\nComparison saved → {path}",
        "fr": "\nComparaison sauvegardée → {path}",
        "es": "\nComparación guardada → {path}"},
    "log_ia_complete"       : {
        "en": "Inter-animal comparison complete.",
        "fr": "Comparaison inter-animal terminée.",
        "es": "Comparación inter-animal completada."},
    "log_im_start"          : {
        "en": "\n--- Inter-method comparison ({methods}) ---",
        "fr": "\n--- Comparaison inter-méthode ({methods}) ---",
        "es": "\n--- Comparación inter-método ({methods}) ---"},
    "log_loading_shared"    : {
        "en": "Loading shared data...",
        "fr": "Chargement des données partagées...",
        "es": "Cargando datos compartidos..."},
    "log_method_start"      : {
        "en": "\n[{name}] Starting pipeline...",
        "fr": "\n[{name}] Démarrage du pipeline...",
        "es": "\n[{name}] Iniciando procesamiento..."},
    "log_method_optimal_k"  : {
        "en": "[{name}] Optimal K = {k}",
        "fr": "[{name}] K optimal = {k}",
        "es": "[{name}] K óptima = {k}"},
    "log_method_done"       : {
        "en": "[{name}] Done → {path}",
        "fr": "[{name}] Terminé → {path}",
        "es": "[{name}] Listo → {path}"},
    "log_im_complete"       : {
        "en": "Inter-method comparison complete.",
        "fr": "Comparaison inter-méthode terminée.",
        "es": "Comparación inter-método completada."},
    "log_output_dir"        : {
        "en": "Output directory : {path}",
        "fr": "Répertoire de sortie : {path}",
        "es": "Directorio de salida : {path}"},
    "log_dti_mode"          : {
        "en": "DTI mode : {mode}  —  parameters : {params}",
        "fr": "Mode DTI : {mode}  —  paramètres : {params}",
        "es": "Modo DTI : {mode}  —  parámetros : {params}"},
    "log_optimal_k"         : {
        "en": "Optimal K : {k}",
        "fr": "K optimal : {k}",
        "es": "K óptima : {k}"},
    "log_cl_loaded"         : {
        "en": "CL reference loaded for {n}/{total} parameters.",
        "fr": "Référence CL chargée pour {n}/{total} paramètres.",
        "es": "Referencia CL cargada para {n}/{total} parámetros."},
    "log_cl_missing"        : {
        "en": "No CL ROI found in CSV files — bio extent annotation disabled.",
        "fr": "Aucune ROI CL trouvée dans les fichiers CSV — annotation d'étendue biologique désactivée.",
        "es": "No se encontró ROI CL en los archivos CSV — anotación de extensión biológica desactivada."},
    "log_labels_received"   : {
        "en": "Labels received -- continuing analysis...",
        "fr": "Étiquettes reçues — continuation de l'analyse...",
        "es": "Etiquetas recibidas — continuando análisis..."},
    "log_analysis_complete" : {
        "en": "\nAnalysis complete.",
        "fr": "\nAnalyse terminée.",
        "es": "\nAnálisis completado."},

    # ── Pass-2 reclassification notes ────────────────────────────────────────
    "reclass_fa_low_md_high"  : {
        "en": "Normalized FA very low ({fa:.2f}) + normalized MD high ({md:.2f}) → Necrosis",
        "fr": "FA normalisée très basse ({fa:.2f}) + MD normalisé élevé ({md:.2f}) → Nécrose",
        "es": "FA normalizada muy baja ({fa:.2f}) + MD normalizado alto ({md:.2f}) → Necrosis"},
    "reclass_md_t2_near_nec"  : {
        "en": "Normalized MD ({md:.2f}) and T2 ({t2:.2f}) close to necrotic cluster → Necrosis",
        "fr": "MD norm ({md:.2f}) et T2 norm ({t2:.2f}) proches du cluster nécrotique → Nécrose",
        "es": "MD normalizado ({md:.2f}) y T2 ({t2:.2f}) cercanos al clúster necrótico → Necrosis"},
    "reclass_md_intermediate" : {
        "en": "Intermediate normalized MD ({md:.2f}) + FA above necrotic → Infiltrative",
        "fr": "MD norm intermédiaire ({md:.2f}) + FA norm supérieure au nécrotique → Infiltratif",
        "es": "MD normalizado intermedio ({md:.2f}) + FA por encima del necrótico → Infiltrativo"},
    "reclass_l2_to_median"    : {
        "en": "Normalized L2 to median = {l2:.2f} → Bulk tumor",
        "fr": "L2 normalisé vers médiane = {l2:.2f} → Tumeur principale",
        "es": "L2 normalizado hacia la mediana = {l2:.2f} → Tumor sólido"},
    "reclass_hypox_far_periphery": {
        "en": "d_topo = {dtopo:.2f} > 0.75 — cluster in tumour periphery, outside hypoxic penumbra → Bulk tumor",
        "fr": "d_topo = {dtopo:.2f} > 0.75 — cluster en périphérie tumorale, hors pénombre hypoxique → Tumeur principale",
        "es": "d_topo = {dtopo:.2f} > 0.75 — clúster en periferia tumoral, fuera de la penumbra hipóxica → Tumor sólido"},

    # ── Pass-3 sub-classification notes ──────────────────────────────────────
    "subcat_t2star_low"  : {
        "en": "T2* << median ({val:.2f} vs {ref:.2f})",
        "fr": "T2* << médiane ({val:.2f} vs {ref:.2f})",
        "es": "T2* << mediana ({val:.2f} vs {ref:.2f})"},
    "subcat_mtr_high"    : {
        "en": "MTR >> median ({val:.2f} vs {ref:.2f})",
        "fr": "MTR >> médiane ({val:.2f} vs {ref:.2f})",
        "es": "MTR >> mediana ({val:.2f} vs {ref:.2f})"},
    "subcat_mtr_low"     : {
        "en": "MTR << median ({val:.2f} vs {ref:.2f})",
        "fr": "MTR << médiane ({val:.2f} vs {ref:.2f})",
        "es": "MTR << mediana ({val:.2f} vs {ref:.2f})"},
    "subcat_fa_moderate" : {
        "en": "FA moderately above median (pos={pos:.2f}) — Jellison pattern 2",
        "fr": "FA modérément au-dessus de la médiane (pos={pos:.2f}) — pattern Jellison 2",
        "es": "FA moderadamente por encima de la mediana (pos={pos:.2f}) — patrón Jellison 2"},
    "subcat_fa_high"     : {
        "en": "FA clearly above median (pos={pos:.2f}) — Jellison pattern 3",
        "fr": "FA nettement au-dessus de la médiane (pos={pos:.2f}) — pattern Jellison 3",
        "es": "FA claramente por encima de la mediana (pos={pos:.2f}) — patrón Jellison 3"},
}

# ---------------------------------------------------------------------------
# Biological habitat label display translation
# Internal canonical labels (stored in CSV) are kept as-is.
# These mappings are only for UI/PDF display.
# ---------------------------------------------------------------------------
_LABEL_DISPLAY = {
    "fr": {
        "Necrosis"                      : "Nécrose",
        "Necrosis (cystic)"             : "Nécrose (kystique)",
        "Necrosis (hemorrhagic)"        : "Nécrose (hémorragique)",
        "Necrosis (proteic)"            : "Nécrose (protéique)",
        "Hypoxic/vascular"              : "Hypoxique/vasculaire",
        "Cellular/proliferative"        : "Cellulaire/prolifératif",
        "Astrocyte barrier"             : "Barrière astrocytaire",
        "Infiltrative"                  : "Infiltratif",
        "Infiltrative (edematous)"      : "Infiltratif (œdémateux)",
        "Infiltrative (infiltrated)"    : "Infiltratif (infiltré)",
        "Bulk tumor"                    : "Tumeur principale",
        "Bulk tumor (atypical)"         : "Tumeur principale (atypique)",
        "Unknown"                       : "Inconnu",
        "Near-normal"                   : "Quasi-normal",
    },
    "es": {
        "Necrosis"                      : "Necrosis",
        "Necrosis (cystic)"             : "Necrosis (quística)",
        "Necrosis (hemorrhagic)"        : "Necrosis (hemorrágica)",
        "Necrosis (proteic)"            : "Necrosis (proteica)",
        "Hypoxic/vascular"              : "Hipóxico/vascular",
        "Cellular/proliferative"        : "Celular/proliferativo",
        "Astrocyte barrier"             : "Barrera astrocítica",
        "Infiltrative"                  : "Infiltrativo",
        "Infiltrative (edematous)"      : "Infiltrativo (edematoso)",
        "Infiltrative (infiltrated)"    : "Infiltrativo (infiltrado)",
        "Bulk tumor"                    : "Tumor sólido",
        "Bulk tumor (atypical)"         : "Tumor sólido (atípico)",
        "Unknown"                       : "Desconocido",
        "Near-normal"                   : "Casi normal",
    },
}


# ---------------------------------------------------------------------------
# Method info for "?" help buttons (English / French / Spanish)
# ---------------------------------------------------------------------------
_METHOD_INFO = {
    "gmm": {
        "fr": {
            "title": "GMM — Modèle de Mélange Gaussien",
            "description": (
                "Le GMM modélise les données comme un mélange de K distributions gaussiennes. "
                "Chaque voxel reçoit une probabilité d'appartenir à chaque cluster "
                "(affectation douce), puis les paramètres gaussiens (μ, Σ, π) sont mis à jour "
                "pour mieux ajuster les données jusqu'à convergence.\n\n"
                "Sélection de K : le BIC (critère d'information bayésien) maximise la vraisemblance "
                "en pénalisant la complexité du modèle ;\n"
                "mais avec peu de voxels, le fléau de la dimensionnalité "
                "dégrade la pertinence de la distance euclidienne (sur-segmentation) ; "
                "on peut alors utiliser le coude de silhouette (espace spectral) comme garde de sécurité."
            ),
            "advantages": (
                "- Probabiliste : capture l'incertitude aux frontières des clusters.\n"
                "- Gère des clusters ellipsoïdaux (pas seulement sphériques comme K-means).\n"
                "- Rapide sur CPU pour les tailles de ROI typiques."
            ),
            "limits": (
                "- Suppose des distributions gaussiennes ; peut échouer sur des données IRM asymétriques.\n"
                "- Le BIC peut surestimer K quand les distributions se chevauchent "
                "(surtout avec haute dimension et peu de données ; comparer avec la k de silhouette "
                "et utiliser une garde de sécurité si la différence est grande).\n"
                "- Sensible à l'initialisation (augmenter n_init améliore la stabilité, mais allonge le calcul).\n"
                "- Pas de régularisation spatiale : résultats spatialement fragmentés possibles."
            ),
        },
        "en": {
            "title": "GMM — Gaussian Mixture Model",
            "description": (
                "GMM models the data as a mixture of K Gaussian distributions. "
                "Each voxel receives a probability of belonging to each cluster "
                "(soft assignment), then updates Gaussian parameters (μ, Σ, π) "
                "to best fit the data and repeat until convergence.\n\n"
                "K selection: BIC (Bayesian information criterion) maximizes likelihood while penalizing model complexity\n"
                "but when there are so few voxels, the curse of dimensionality "
                "impacts the relevant aspect of the euclidean distance (over-segmentation) "
                "to maximize likelihood; that's why we can use the silhouette elbow (spectral space) as a safety guard."
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
                "with the k from silhouette and use a safety guard if you see a big difference of numbers of cluster.\n"
                "- Sensitive to initialization (you can increase n_init restarts but the comput time will also increase).\n"
                "- No spatial regularization so spatially fragmented results possible."
            ),
        },
        "es": {
            "title": "GMM — Modelo de Mezcla Gaussiana",
            "description": (
                "GMM modela los datos como una mezcla de K distribuciones gaussianas. "
                "Cada vóxel recibe una probabilidad de pertenecer a cada clúster "
                "(asignación suave), luego actualiza los parámetros gaussianos (μ, Σ, π) "
                "para ajustarse mejor a los datos y repite hasta la convergencia.\n\n"
                "Selección de K: BIC (criterio de información bayesiano) maximiza la verosimilitud penalizando la complejidad del modelo\n"
                "pero cuando hay muy pocos vóxeles, la «maldición de la dimensionalidad» "
                "impacta la relevancia de la distancia euclidiana (sobre-segmentación) "
                "para maximizar la verosimilitud; por eso se puede usar el codo de silueta (espacio espectral) como guardia de seguridad."
            ),
            "advantages": (
                "- Probabilístico: captura la incertidumbre en los bordes de los clústeres.\n"
                "- Maneja clústeres elipsoidales (no solo esféricos como K-means).\n"
                "- Rápido en CPU para tamaños típicos de ROI."
            ),
            "limits": (
                "- Asume distribuciones gaussianas, puede fallar con datos MRI sesgados.\n"
                "- BIC puede sobreestimar K cuando las distribuciones se superponen "
                "(especialmente con alta dimensión y pocos datos; puedes comparar con la k de silueta "
                "y usar un guardia de seguridad si ves una gran diferencia en el número de clústeres).\n"
                "- Sensible a la inicialización (puedes aumentar los reinicios n_init pero el tiempo de cómputo también aumentará).\n"
                "- Sin regularización espacial, son posibles resultados fragmentados espacialmente."
            ),
        },
    },
    "srsc": {
        "fr": {
            "title": "SRSC — Clustering Spectral à Régularisation Spatiale",
            "description": (
                "Le SRSC construit une matrice d'affinité combinant la similarité des "
                "caractéristiques IRM (kernel RBF) et la proximité spatiale (kernel gaussien en X,Y) "
                "via un produit de Hadamard pondéré : W = W_feat × (W_spat ^ alpha).\n\n"
                "alpha=1 → filtrage classique (régularisation spatiale totale).\n"
                "alpha=0 → clustering purement par caractéristiques (W_spat disparaît).\n"
                "Les valeurs intermédiaires modulent l'influence spatiale en continu.\n\n"
                "Sélection de K : coude de silhouette + statistique Gap dans l'espace spectral "
                "(regroupement type k-means sur la plage de k, génération d'un jeu de référence, "
                "calcul de la fonction de perte WSS et de la statistique Gap, puis choix du k optimal : "
                "Gap(k)≥Gap(k+1)−SE(k+1))."
            ),
            "advantages": (
                "- La régularisation spatiale produit des régions cohérentes et compactes.\n"
                "- Pas d'hypothèse gaussienne : gère des formes de cluster arbitraires (y compris non convexes).\n"
                "- alpha contrôle l'influence spatiale en continu : 0 = purement caractéristiques, 1 = spatial complet.\n"
                "- Bien adapté à la cartographie d'habitats IRM (validé dans la littérature)."
            ),
            "limits": (
                "- Matrice d'affinité en O(n²) : lente et gourmande en mémoire pour de grandes ROI.\n"
                "- Deux paramètres à régler : sigma_feature (σ faible → seuls les voxels très proches sont similaires) "
                "et alpha (α élevé → peut fusionner des zones biologiquement distinctes mais spatialement adjacentes).\n"
                "- Avec des données de grande dimension et peu d'échantillons, "
                "les distances euclidiennes deviennent moins discriminantes (fléau de la dimensionnalité)."
            ),
        },
        "en": {
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
        "es": {
            "title": "SRSC — Agrupamiento Espectral con Regularización Espacial",
            "description": (
                "SRSC construye una matriz de afinidad que combina similitud de características MRI "
                "(kernel RBF) y proximidad espacial (kernel gaussiano en X,Y) mediante un "
                "producto de Hadamard ponderado: W = W_feat × (W_spat ^ alpha).\n\n"
                "alpha=1 → compuerta clásica (regularización espacial completa).\n"
                "alpha=0 → agrupamiento puro por características (W_spat desaparece).\n"
                "Los valores intermedios modulan la influencia espacial de forma continua.\n\n"
                "Selección de K: Codo de Silueta + Estadística Gap en espacio espectral "
                "(primero reagrupa los datos como k-means, para k en rango, luego "
                "genera un conjunto de datos de referencia, calcula la función de pérdida "
                "con WSS y la estadística gap, y finalmente elige el k óptimo: Gap(k)≥Gap(k+1)−SE(k+1))."
            ),
            "advantages": (
                "- La regularización espacial produce regiones coherentes y compactas.\n"
                "- Sin supuestos gaussianos, maneja formas arbitrarias (incluso no convexas) de clúster.\n"
                "- alpha controla la influencia espacial de forma continua: 0 = solo características, 1 = completamente espacial.\n"
                "- Bien adaptado para la cartografía de hábitats MRI (validado en la literatura)."
            ),
            "limits": (
                "- Matriz de afinidad O(n²): lenta y con mucha memoria para ROIs grandes.\n"
                "- Dos parámetros a ajustar: sigma_feature (σ pequeño → solo vóxeles muy cercanos son similares) "
                "y alpha (α alto → puede fusionar zonas biológicamente distintas pero espacialmente adyacentes).\n"
                "- Con datos de alta dimensión y pocas muestras, las distancias euclidianas "
                "se vuelven menos discriminativas (maldición de la dimensionalidad)."
            ),
        },
    },
    "kmeans": {
        "fr": {
            "title": "K-means",
            "description": (
                "Partitionne les données en k groupes distincts selon la similarité, "
                "puis affecte chaque point au centroïde de cluster le plus proche "
                "(distance euclidienne) et minimise l'inertie intra-cluster.\n\n"
                "Sélection de K : coude de silhouette + statistique Gap "
                "(regroupement type k-means sur la plage de k, génération d'un jeu de référence, "
                "calcul de la fonction de perte WSS et de la statistique Gap, puis choix du k optimal "
                "Gap(k)≥Gap(k+1)−SE(k+1), dans l'espace spectral."
            ),
            "advantages": (
                "- Rapide et simple : bon point de référence pour comparaison.\n"
                "- Pas d'hyperparamètres au-delà de K.\n"
                "- La silhouette dans l'espace des caractéristiques est directement interprétable."
            ),
            "limits": (
                "- Suppose des clusters sphériques de taille égale.\n"
                "- Sensible aux valeurs aberrantes (les voxels extrêmes peuvent déformer les centroïdes).\n"
                "- Pas d'affectation probabiliste (contrairement aux autres) : frontières dures uniquement.\n"
                "- La statistique Gap peut surestimer K sur des courbes monotones "
                "(la silhouette peut alors servir de garde de sécurité)."
            ),
        },
        "en": {
            "title": "K-means",
            "description": (
                "Partitions the dataset into k distinct groups based on similarity, then assigns to each point "
                "the nearest cluster centroid (Euclidean distance) and minimizes within-cluster inertia.\n\n"
                "K selection: Silhouette elbow + Gap Statistic "
                "(first it regroups data like k-means, for k in range, then "
                "generates a reference dataset, computes the loss function "
                "with WSS, and the gap statistic and finally chooses the optimal "
                "k Gap(k)≥Gap(k+1)−SE(k+1), in spectral space."
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
                "- Gap Statistic may overestimate K on monotone score curves (so you can use silhouette as safety guard)."
            ),
        },
        "es": {
            "title": "K-means",
            "description": (
                "Divide el conjunto de datos en k grupos distintos basándose en la similitud, luego asigna "
                "a cada punto el centroide de clúster más cercano (distancia euclidiana) y minimiza la inercia dentro del clúster.\n\n"
                "Selección de K: Codo de Silueta + Estadística Gap "
                "(primero reagrupa los datos como k-means, para k en rango, luego "
                "genera un conjunto de datos de referencia, calcula la función de pérdida "
                "con WSS y la estadística gap, y finalmente elige el k óptimo "
                "Gap(k)≥Gap(k+1)−SE(k+1), en espacio espectral."
            ),
            "advantages": (
                "- Rápido y simple: buen punto de referencia para comparación.\n"
                "- Sin hiperparámetros más allá de K.\n"
                "- La silueta en el espacio de características es directamente interpretable."
            ),
            "limits": (
                "- Asume clústeres esféricos de igual tamaño.\n"
                "- Sensible a los valores atípicos (los vóxeles extremos pueden distorsionar los centroides).\n"
                "- Sin asignación probabilística (a diferencia de los otros): solo límites duros.\n"
                "- La Estadística Gap puede sobreestimar K en curvas de puntuación monótonas "
                "(por lo que se puede usar la silueta como guardia de seguridad)."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_lang() -> str:
    return _LANG


def set_lang(lang: str):
    global _LANG
    _LANG = lang


def t(key: str, **kwargs) -> str:
    """Return the translated string for *key* in the current language."""
    row = _TR.get(key)
    if row is None:
        return key
    text = row.get(_LANG, row.get("en", key))
    return text.format(**kwargs) if kwargs else text


def get_method_info(key: str) -> dict:
    """Return method info dict for the current language, falling back to English."""
    row = _METHOD_INFO.get(key, {})
    return row.get(_LANG, row.get("en", {}))


def display_label(canonical: str) -> str:
    """Translate a canonical habitat label for display in the current language.

    If no translation exists the canonical name is returned unchanged.
    Handles compound labels like "Necrosis (kystique) 2" or "Bulk tumor +".
    """
    if _LANG == "en":
        return canonical

    mapping = _LABEL_DISPLAY.get(_LANG, {})

    # Try longest-match first (handles sub-types before base types)
    for src in sorted(mapping, key=len, reverse=True):
        if canonical.startswith(src):
            suffix = canonical[len(src):]
            return mapping[src] + suffix

    return canonical
