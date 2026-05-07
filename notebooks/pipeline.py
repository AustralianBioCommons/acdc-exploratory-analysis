import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # ACDC exploratory analysis

    This notebook analyses the **clinical variables defined in the
    ACDC Gen3 Data Dictionary** across one or more contributing
    cohorts (configured at the top of the notebook). Baseline metadata
    are retrieved live from Gen3 and flattened to one record per
    subject, after which we report (i) cohort characteristics, (ii)
    variable- and subject-level missing data, (iii) within-cohort
    Spearman rank correlations across a pre-specified panel of
    clinical variables drawn from the dictionary, and (iv)
    cross-cohort correlation deviation as a triage map for measurement
    disagreement.

    All analyses are descriptive. No hypothesis tests are performed and
    no inferential statistics are reported.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from gen3_metadata.gen3_metadata_parser import fetch_all_metadata
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist, squareform

    return (
        fetch_all_metadata,
        leaves_list,
        linkage,
        np,
        pd,
        pdist,
        plt,
        squareform,
        warnings,
    )


@app.cell
def _():
    # Knobs surfaced at the top so the reader can see them at a glance.
    # `STUDY_IDS` is the single source of truth for which Gen3 cohorts
    # the pipeline runs against — every downstream cell iterates this.
    KEY_FILE = "/Users/harrijh/keys/acdc_api_key_staging.json"
    PROGRAM = "program1"
    STUDY_IDS = ("CDAH", "EDCAD-PMS", "AusDiab", "Baker-Biobank", "CAUGHT-CAD")
    MIN_PAIRWISE_N = 30  # mask Spearman cells with pairwise N below this
    DASH = "—"
    return DASH, KEY_FILE, MIN_PAIRWISE_N, PROGRAM, STUDY_IDS


@app.cell
def _(mo):
    mo.md("""
    ## 1. Descriptive overview

    For each cohort, the Gen3 clinical graph (`subject` →
    `clinical_descriptor` → leaf nodes: `demographic`,
    `medical_history`, `blood_pressure_test`, `exposure`, `lab_result`,
    `medication`) was joined into a single wide table with one row per
    subject. Only baseline measures were retained
    (`clinical_descriptor.baseline_measure == True`). Leaf-node columns
    are prefixed `<node>__` to disambiguate name collisions across
    nodes (e.g. `heart_rate` appears in both `medical_history` and
    `blood_pressure_test`).
    """)
    return


@app.cell
def _():
    # ---- Flatten helpers (embedded from scripts/flatten.py) ----

    LEAF_NODES = (
        "demographic",
        "medical_history",
        "blood_pressure_test",
        "exposure",
        "lab_result",
        "medication",
    )
    DROP_COMMON = (
        "project_id",
        "data_release",
        "data_release_date",
        "id",
        "type",
    )

    def fk_submitter_id(value):
        """Extract submitter_id from a Gen3 FK column ([{'submitter_id': ...}, ...])."""
        if isinstance(value, list) and value:
            return value[0].get("submitter_id")
        if isinstance(value, dict):
            return value.get("submitter_id")
        return None

    def _slim(df, fk_col, link_col, node_name):
        """Return leaf node with FK extracted, common cols dropped, and node-prefixed columns."""
        df = df.copy()
        df[link_col] = df[fk_col].apply(fk_submitter_id)
        drop = [c for c in list(DROP_COMMON) + [fk_col] if c in df.columns]
        df = df.drop(columns=drop)
        rename = {c: f"{node_name}__{c}" for c in df.columns if c not in (link_col,)}
        rename["submitter_id"] = f"{node_name}_submitter_id"
        df = df.rename(
            columns={**rename, "submitter_id": f"{node_name}_submitter_id"}
        )
        if f"{node_name}__submitter_id" in df.columns:
            df = df.rename(
                columns={f"{node_name}__submitter_id": f"{node_name}_submitter_id"}
            )
        return df

    def flatten(dfs):
        """Join subject ← clinical_descriptor (baseline) ← leaf nodes into one wide frame."""
        subject = (
            getattr(dfs, "subject")
            .copy()
            .rename(columns={"submitter_id": "subject_submitter_id"})
        )

        cd = getattr(dfs, "clinical_descriptor").copy()
        if "baseline_measure" in cd.columns:
            cd = cd[cd["baseline_measure"] == True]  # noqa: E712
        cd["subject_submitter_id"] = cd["subjects"].apply(fk_submitter_id)
        cd = cd.drop(
            columns=[c for c in list(DROP_COMMON) + ["subjects"] if c in cd.columns]
        )
        cd = cd.rename(columns={"submitter_id": "clinical_descriptor_submitter_id"})

        flat = subject.merge(cd, on="subject_submitter_id", how="left")
        for node in LEAF_NODES:
            leaf = getattr(dfs, node)
            if leaf.empty:
                continue
            slim = _slim(
                leaf,
                "clinical_descriptors",
                "clinical_descriptor_submitter_id",
                node,
            )
            flat = flat.merge(
                slim, on="clinical_descriptor_submitter_id", how="left"
            )
        return flat

    return (flatten,)


@app.cell
def _(KEY_FILE, PROGRAM, STUDY_IDS, fetch_all_metadata, flatten, mo):
    # Live fetch + flatten. This is the slow cell — Gen3 round-trip.
    flats = {}
    for _study in STUDY_IDS:
        _result = fetch_all_metadata(KEY_FILE, PROGRAM, _study)
        flats[_study] = flatten(_result.to_df())

    _summary = "\n".join(
        f"- **{_study}**: {_df.shape[0]:,} subjects × {_df.shape[1]} columns"
        for _study, _df in flats.items()
    )
    mo.md(
        "After flattening, the analytic samples comprised:\n\n"
        f"{_summary}"
    )
    return (flats,)


@app.cell
def _(mo):
    mo.md("""
    ### Table 1. Baseline cohort characteristics

    Continuous variables are summarised as `mean ± SD (median; range);
    N=non-missing`. Categorical variables are summarised as `n (%)` per
    level with the per-variable N on the header row. Variables not
    collected by a given cohort are denoted `—`; the *Overall* column
    is populated only when a variable is present in both cohorts.
    """)
    return


@app.cell
def _(DASH, pd):
    # ---- Table 1 helpers (embedded from scripts/table_one.py) ----

    SPEC = [
        # Demographics
        ("Demographics", "Age, years", "demographic__baseline_age", "continuous"),
        ("Demographics", "Sex", "demographic__sex", "categorical"),
        ("Demographics", "Education", "demographic__education", "categorical"),
        ("Demographics", "ABS state", "demographic__abs_state", "categorical"),
        ("Demographics", "BMI, kg/m^2", "demographic__bmi_baseline", "continuous"),
        ("Demographics", "Height (measured), m", "demographic__height_baseline_measured", "continuous"),
        ("Demographics", "Weight (measured), kg", "demographic__weight_baseline_measured", "continuous"),
        ("Demographics", "Height (self-reported), m", "demographic__height_baseline_self_reported", "continuous"),
        ("Demographics", "Weight (self-reported), kg", "demographic__weight_baseline_self_reported", "continuous"),
        # Vitals
        ("Vitals", "Systolic BP, mmHg", "blood_pressure_test__bp_systolic", "continuous"),
        ("Vitals", "Diastolic BP, mmHg", "blood_pressure_test__bp_diastolic", "continuous"),
        ("Vitals", "Heart rate, bpm", "medical_history__heart_rate", "continuous"),
        # Labs
        ("Labs", "Total cholesterol, mmol/L", "lab_result__total_cholesterol", "continuous"),
        ("Labs", "HDL, mmol/L", "lab_result__hdl", "continuous"),
        ("Labs", "LDL, mmol/L", "lab_result__ldl", "continuous"),
        ("Labs", "Triglycerides, mmol/L", "lab_result__triglycerides", "continuous"),
        ("Labs", "Fasting glucose, mmol/L", "lab_result__glucose_fasting", "continuous"),
        ("Labs", "Fasting status", "lab_result__fasting", "categorical"),
        ("Labs", "Creatinine (enzymatic), umol/L", "lab_result__creatinine_serum_enzymatic", "continuous"),
        ("Labs", "Creatinine (Jaffe), umol/L", "lab_result__creatinine_serum_jaffe", "continuous"),
        ("Labs", "eGFR (CKD-EPI)", "lab_result__egfr_baseline_ckdepi", "continuous"),
        ("Labs", "eGFR (MDRD)", "lab_result__egfr_baseline_mdrd", "continuous"),
        ("Labs", "Coronary artery calcium score", "lab_result__cac_score", "continuous"),
        # Medical history
        ("Medical history", "CVD, family history", "medical_history__cvd_family_history", "categorical"),
        ("Medical history", "Premature CVD, family history", "medical_history__premature_cvd_family_history", "categorical"),
        ("Medical history", "CVD, self-reported", "medical_history__cvd_self_reported", "categorical"),
        ("Medical history", "Diabetes, self-reported", "medical_history__diabetes_self_reported", "categorical"),
        ("Medical history", "Diabetes, reported", "medical_history__diabetes_reported", "categorical"),
        ("Medical history", "Diabetes type", "medical_history__diabetes_type", "categorical"),
        ("Medical history", "Hypertension, self-reported", "medical_history__hypertension_self_reported", "categorical"),
        ("Medical history", "Atrial fibrillation", "medical_history__atrial_fibrillation", "categorical"),
        # Medications
        ("Medications", "BP-lowering medication", "medication__bp_lowering_meds", "categorical"),
        ("Medications", "Lipid-lowering medication", "medication__lipid_lowering_meds", "categorical"),
        ("Medications", "Diabetes therapy", "medication__diabetes_therapy", "categorical"),
        ("Medications", "Antithrombotic medication", "medication__antithrombotic_meds", "categorical"),
        # Exposures
        ("Exposures", "Smoking status", "exposure__smoking_status", "categorical"),
        ("Exposures", "Cigarettes per day", "exposure__cigarettes_per_day", "continuous"),
        ("Exposures", "Drinks per week", "exposure__drinks_per_week", "continuous"),
        ("Exposures", "Drinks per week (string)", "exposure__drinks_per_week_string", "continuous"),
    ]

    def fmt_continuous(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return DASH
        return (
            f"{s.mean():.1f} ± {s.std():.1f} "
            f"(median {s.median():.1f}; range {s.min():.1f}–{s.max():.1f}); "
            f"N={len(s)}"
        )

    def categorical_rows(s):
        s = s.dropna()
        if s.empty:
            return []
        s = s.astype(str)
        counts = s.value_counts()
        n = int(counts.sum())
        return [(level, f"{int(c)} ({c / n * 100:.1f}%)") for level, c in counts.items()]

    def build_table(frames, pooled):
        """Build Table 1 from a {study_id: DataFrame} dict.

        Per-cohort columns are emitted in `frames` insertion order, with
        a trailing `Overall` column that is populated only when every
        cohort contains the variable.
        """
        study_cols = (*frames.keys(), "Overall")
        empty_row = {"Domain": "", "Variable": "", "Level": "", **{c: "" for c in study_cols}}

        rows = []
        last_domain = None
        for domain, label, col, kind in SPEC:
            present = {study: (col in df.columns) for study, df in frames.items()}
            if not any(present.values()):
                continue
            if domain != last_domain:
                rows.append({**empty_row, "Domain": domain})
                last_domain = domain

            overall_applicable = all(present.values())

            if kind == "continuous":
                cells = {
                    study: (fmt_continuous(df[col]) if present[study] else DASH)
                    for study, df in frames.items()
                }
                cells["Overall"] = fmt_continuous(pooled[col]) if overall_applicable else DASH
                rows.append({"Domain": "", "Variable": label, "Level": "", **cells})
            else:
                per_study_levels = {
                    study: (categorical_rows(df[col]) if present[study] else [])
                    for study, df in frames.items()
                }
                overall_levels = categorical_rows(pooled[col]) if overall_applicable else []

                # Header row: total N per column (or em-dash if absent).
                header_cells = {}
                for study, levels in per_study_levels.items():
                    if not present[study]:
                        header_cells[study] = DASH
                    else:
                        n = sum(int(v.split(" ")[0]) for _, v in levels) if levels else 0
                        header_cells[study] = f"N={n}"
                if overall_applicable:
                    overall_n = sum(int(v.split(" ")[0]) for _, v in overall_levels) if overall_levels else 0
                    header_cells["Overall"] = f"N={overall_n}"
                else:
                    header_cells["Overall"] = DASH
                rows.append({"Domain": "", "Variable": label, "Level": "", **header_cells})

                # Level union, ordered by overall frequency where available,
                # then by per-cohort frequency.
                level_order = []
                for src in (overall_levels, *per_study_levels.values()):
                    for lvl, _ in src:
                        if lvl not in level_order:
                            level_order.append(lvl)
                study_maps = {study: dict(levels) for study, levels in per_study_levels.items()}
                overall_map = dict(overall_levels)
                for lvl in level_order:
                    level_cells = {
                        study: (
                            study_maps[study].get(lvl, "0 (0.0%)")
                            if present[study]
                            else DASH
                        )
                        for study in frames
                    }
                    level_cells["Overall"] = (
                        overall_map.get(lvl, "0 (0.0%)") if overall_applicable else DASH
                    )
                    rows.append({"Domain": "", "Variable": "", "Level": f"  {lvl}", **level_cells})

        return pd.DataFrame(rows, columns=["Domain", "Variable", "Level", *study_cols])

    def to_markdown(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = "" if pd.isna(row[c]) else str(row[c])
                v = v.replace("|", "\\|").replace("\n", " ")
                cells.append(v)
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    return build_table, to_markdown


@app.cell
def _(build_table, flats, mo, pd, to_markdown):
    _pooled = pd.concat(
        [df.assign(study=study) for study, df in flats.items()],
        ignore_index=True,
        sort=False,
    )
    table_one_df = build_table(flats, _pooled)

    mo.vstack(
        [
            mo.md("Sortable rendering of Table 1 (click a column header to sort)."),
            mo.ui.table(table_one_df, page_size=50),
            mo.md("Manuscript rendering of Table 1."),
            mo.md(to_markdown(table_one_df)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Missing data assessment

    Variable-level completeness was computed as the proportion of
    non-missing values per variable per cohort, after excluding
    foreign-key, identifier, and Gen3 bookkeeping columns. Two states
    are distinguished throughout: a variable not collected by a cohort
    (rendered as a grey `N/A` cell) versus a variable collected but
    incompletely populated (rendered on a 0–100% OrRd scale).
    """)
    return


@app.cell
def _(DASH, leaves_list, linkage, np, pd, pdist):
    # ---- Missingness helpers (embedded from scripts/missingness.py) ----

    EXCLUDE_EXACT = {
        "project_id",
        "data_release",
        "data_release_date",
        "id",
        "type",
        "projects",
        "patient_id",
        "cohort_id",
        "baseline_measure",
        "context_label",
        "site",
    }

    def eligible_columns(df):
        return [
            c for c in df.columns
            if c not in EXCLUDE_EXACT and not c.endswith("_submitter_id")
        ]

    def union_variables(frames):
        all_vars = []
        for df in frames.values():
            for c in eligible_columns(df):
                if c not in all_vars:
                    all_vars.append(c)
        all_vars.sort(key=lambda c: (c.split("__", 1)[0], c))
        return all_vars

    def completeness_matrix(frames, all_vars):
        data = {}
        for study, df in frames.items():
            total = len(df)
            col = []
            for var in all_vars:
                if var not in df.columns or total == 0:
                    col.append(np.nan)
                else:
                    col.append(df[var].notna().sum() / total * 100.0)
            data[study] = col
        return pd.DataFrame(data, index=all_vars).T  # rows=study, cols=variable

    def completeness_summary(frames):
        all_vars = union_variables(frames)
        pct_df = completeness_matrix(frames, all_vars)
        rows = []
        for col in all_vars:
            row = {"variable": col}
            for study, df in frames.items():
                pct = pct_df.loc[study, col]
                if pd.isna(pct):
                    row[f"{study}_pct"] = DASH
                    row[f"{study}_n"] = DASH
                else:
                    non_null = int(df[col].notna().sum())
                    row[f"{study}_pct"] = f"{pct:.1f}%"
                    row[f"{study}_n"] = f"{non_null}/{len(df)}"
            rows.append(row)
        return pd.DataFrame(rows)

    def cluster_order(matrix):
        n_rows = matrix.shape[0]
        if n_rows < 2:
            return np.arange(n_rows)
        distances = pdist(matrix, metric="hamming")
        if not np.any(distances):
            return np.arange(n_rows)
        link = linkage(distances, method="average")
        return leaves_list(link)

    return (
        cluster_order,
        completeness_matrix,
        completeness_summary,
        union_variables,
    )


@app.cell
def _(flats):
    # Alias kept for clarity in the missingness section; the underlying
    # data is the same dict produced by the live-fetch cell.
    frames_miss = flats
    return (frames_miss,)


@app.cell
def _(completeness_summary, frames_miss, mo):
    completeness_df = completeness_summary(frames_miss)
    mo.vstack(
        [
            mo.md(
                "Per-variable completeness for both cohorts. `*_pct` is "
                "the percentage of non-missing values; `*_n` is "
                "`non-missing / total subjects`. An em-dash indicates "
                "the variable was not collected by that cohort."
            ),
            mo.ui.table(completeness_df, page_size=50),
        ]
    )
    return


@app.cell
def _(completeness_matrix, frames_miss, np, pd, plt, union_variables):
    # ---- Variable completeness heatmap (inlined from plot_completeness_heatmap) ----

    _all_vars = union_variables(frames_miss)
    _pct_df = completeness_matrix(frames_miss, _all_vars)
    _matrix = np.ma.masked_invalid(_pct_df.to_numpy(dtype=float))
    _cmap = plt.get_cmap("OrRd").copy()
    _cmap.set_bad("#cccccc")
    _n_studies, _n_vars = _matrix.shape

    completeness_fig, _ax = plt.subplots(
        figsize=(max(10.0, 0.28 * _n_vars), 4.0 + 0.6 * _n_studies),
        constrained_layout=True,
    )
    _im = _ax.imshow(_matrix, aspect="auto", cmap=_cmap, interpolation="nearest", vmin=0, vmax=100)

    for _i in range(_n_studies):
        for _j in range(_n_vars):
            _val = _pct_df.iat[_i, _j]
            if pd.isna(_val):
                _text = "N/A"
                _color = "black"
            else:
                _text = f"{_val:.0f}%"
                _color = "white" if _val >= 60 else "black"
            _ax.text(_j, _i, _text, ha="center", va="center", fontsize=6, color=_color)

    _ax.set_yticks(range(_n_studies))
    _ax.set_yticklabels(list(_pct_df.index), fontsize=10)
    _ax.set_xticks(range(_n_vars))
    _ax.set_xticklabels(_all_vars, rotation=90, fontsize=6)
    _ax.set_xlabel("variables (grouped by node)")

    _last_prefix = None
    for _j, _var in enumerate(_all_vars):
        _prefix = _var.split("__", 1)[0]
        if _last_prefix is not None and _prefix != _last_prefix:
            _ax.axvline(_j - 0.5, color="red", linewidth=0.4, alpha=0.6)
        _last_prefix = _prefix

    _cbar = completeness_fig.colorbar(_im, ax=_ax, fraction=0.025, pad=0.01)
    _cbar.set_label("% subjects with non-missing value")

    completeness_fig.suptitle(
        "Variable completeness by study (gray = variable not collected)",
        fontsize=11,
    )
    completeness_fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Figure 1.** Variable completeness by cohort. Each cell is the
    percentage of subjects with a non-missing value for that variable
    (OrRd scale, 0–100%); grey cells annotated `N/A` indicate the
    variable was not collected by that cohort. Variables are grouped
    by Gen3 node, with red vertical lines separating the groups.
    """)
    return


@app.cell
def _(cluster_order, frames_miss, np, plt, union_variables):
    # ---- Subject × variable missingness heatmap (inlined from plot_heatmap) ----

    _all_vars = union_variables(frames_miss)
    _studies = list(frames_miss.items())
    _width_ratios = [max(1, len(df)) for _, df in _studies]

    missingness_fig, _axes = plt.subplots(
        1,
        len(_studies),
        figsize=(6 + 4 * len(_studies), 0.18 * len(_all_vars) + 3),
        gridspec_kw={"width_ratios": _width_ratios},
        sharey=True,
        constrained_layout=True,
    )
    if len(_studies) == 1:
        _axes = [_axes]

    for _ax, (_study, _df) in zip(_axes, _studies):
        _matrix = np.zeros((len(_all_vars), len(_df)), dtype=np.uint8)
        for _i, _var in enumerate(_all_vars):
            if _var in _df.columns:
                _matrix[_i, :] = _df[_var].notna().to_numpy(dtype=np.uint8)

        _order = cluster_order(_matrix.T)
        _ordered = _matrix[:, _order]

        _ax.imshow(_ordered, aspect="auto", cmap="binary_r", interpolation="nearest", vmin=0, vmax=1)
        _ax.set_title(f"{_study} — n={len(_df)} subjects")
        _ax.set_xticks([])
        _ax.set_xlabel("subjects (clustered)")

        _last_prefix = None
        for _i, _var in enumerate(_all_vars):
            _prefix = _var.split("__", 1)[0]
            if _last_prefix is not None and _prefix != _last_prefix:
                _ax.axhline(_i - 0.5, color="red", linewidth=0.4, alpha=0.6)
            _last_prefix = _prefix

    _axes[0].set_yticks(range(len(_all_vars)))
    _axes[0].set_yticklabels(_all_vars, fontsize=6)
    _axes[0].set_ylabel("variables (grouped by node)")

    missingness_fig.suptitle(
        "Variable × subject missingness, faceted by study (white = present, black = missing)",
        fontsize=11,
    )
    missingness_fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Figure 2.** Subject × variable missingness, faceted by cohort.
    Cells are binary (white = present, black = missing). Subjects
    (columns) are ordered by hierarchical clustering on the
    missingness vector (Hamming distance, average linkage); vertical
    bands therefore indicate groups of subjects sharing the same
    missingness pattern, consistent with structured rather than random
    missingness. Variables (rows) are grouped by Gen3 node.
    """)
    return


@app.cell
def _(MIN_PAIRWISE_N, mo):
    mo.md(f"""
    ## 3. Within-cohort correlation structure

    Spearman rank correlations were computed within each cohort across
    a pre-specified panel of continuous, ordinal, and binary clinical
    variables. Spearman was preferred over Pearson because clinical
    variables are typically skewed and contain outliers, and because
    rank correlation captures any monotonic relationship rather than
    only linear ones. Ordinal variables (education, smoking status)
    were coded as integers preserving their natural ordering; binary
    variables (sex, fasting status, self-reported conditions, and
    medication-use flags) were coded 0/1. Multi-level nominal
    variables without a natural order were excluded.

    Correlations were estimated using pairwise-complete observations
    so that each cell of the matrix uses the maximum available sample
    size for that variable pair. Cells in which the pairwise N fell
    below {MIN_PAIRWISE_N} were masked. A single canonical variable
    order was derived from the pooled-cohort correlation matrix via
    hierarchical clustering (distance = `1 − |ρ|`, average linkage)
    and applied to every heatmap, so equivalent positions across
    figures correspond to the same variable pair. Cohort-exclusive
    variables were appended after the clustered core, grouped by Gen3
    node.

    These analyses are intended as an internal-consistency check on
    measurement and coding rather than as discovery analyses. No
    inferential statistics are reported.
    """)
    return


@app.cell
def _(MIN_PAIRWISE_N, leaves_list, linkage, np, pd, plt, squareform, warnings):
    # ---- Correlation helpers (embedded from scripts/correlations.py) ----

    CONTINUOUS_VARS = (
        "demographic__baseline_age",
        "demographic__bmi_baseline",
        "demographic__height_baseline_measured",
        "demographic__weight_baseline_measured",
        "demographic__height_baseline_self_reported",
        "demographic__weight_baseline_self_reported",
        "blood_pressure_test__bp_systolic",
        "blood_pressure_test__bp_diastolic",
        "medical_history__heart_rate",
        "lab_result__total_cholesterol",
        "lab_result__hdl",
        "lab_result__ldl",
        "lab_result__triglycerides",
        "lab_result__glucose_fasting",
        "lab_result__creatinine_serum_enzymatic",
        "lab_result__creatinine_serum_jaffe",
        "lab_result__egfr_baseline_ckdepi",
        "lab_result__egfr_baseline_mdrd",
        "lab_result__cac_score",
        "exposure__cigarettes_per_day",
        "exposure__drinks_per_week",
    )

    ORDINAL_CODINGS = {
        "demographic__education": {"low": 0, "medium": 1, "high": 2},
        "exposure__smoking_status": {"never": 0, "former": 1, "current": 2},
    }

    BINARY_CODINGS = {
        "demographic__sex": {"female": 0, "male": 1},
        "lab_result__fasting": {"no": 0, "yes": 1},
        "medical_history__cvd_self_reported": {"no": 0, "yes": 1},
        "medical_history__diabetes_self_reported": {"no": 0, "yes": 1},
        "medical_history__cvd_family_history": {"no": 0, "yes": 1},
        "medical_history__premature_cvd_family_history": {"no": 0, "yes": 1},
        "medical_history__hypertension_self_reported": {"no": 0, "yes": 1},
        "medical_history__atrial_fibrillation": {"no": 0, "yes": 1},
        "medication__bp_lowering_meds": {"no": 0, "yes": 1},
        "medication__lipid_lowering_meds": {"no": 0, "yes": 1},
        "medication__antithrombotic_meds": {"no": 0, "yes": 1},
    }

    ALL_VARS = list(CONTINUOUS_VARS) + list(ORDINAL_CODINGS) + list(BINARY_CODINGS)

    def prepare_frame(df):
        """Numeric analysis frame: continuous coerced, ordinals/binaries coded."""
        df = df.copy()
        if (
            "exposure__drinks_per_week" not in df.columns
            and "exposure__drinks_per_week_string" in df.columns
        ):
            df["exposure__drinks_per_week"] = pd.to_numeric(
                df["exposure__drinks_per_week_string"], errors="coerce"
            )

        out = pd.DataFrame(index=df.index)
        for col in CONTINUOUS_VARS:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce")
        for col, mapping in ORDINAL_CODINGS.items():
            if col in df.columns:
                out[col] = df[col].astype("string").str.strip().str.lower().map(mapping)
        for col, mapping in BINARY_CODINGS.items():
            if col in df.columns:
                out[col] = df[col].astype("string").str.strip().str.lower().map(mapping)
        return out

    def _populated_columns(df):
        return [c for c in df.columns if df[c].notna().any()]

    def canonical_order(frames):
        populated = {k: set(_populated_columns(df)) for k, df in frames.items()}
        shared = [c for c in ALL_VARS if all(c in cols for cols in populated.values())]

        if len(shared) >= 2:
            pooled = pd.concat([df[shared] for df in frames.values()], axis=0, ignore_index=True)
            rho = pooled.corr(method="spearman", min_periods=MIN_PAIRWISE_N)
            dist = (1.0 - rho.abs()).fillna(1.0).to_numpy(dtype=float)
            np.fill_diagonal(dist, 0.0)
            dist = (dist + dist.T) / 2.0
            condensed = squareform(dist, checks=False)
            if np.any(condensed):
                link = linkage(condensed, method="average")
                ordered_shared = [shared[i] for i in leaves_list(link)]
            else:
                ordered_shared = shared
        else:
            ordered_shared = shared

        seen = set(ordered_shared)
        union_populated = set().union(*populated.values())
        exclusives = [c for c in ALL_VARS if c not in seen and c in union_populated]
        exclusives.sort(key=lambda c: (c.split("__", 1)[0], c))
        return ordered_shared + exclusives

    def study_correlation(df, order):
        present = [c for c in order if c in df.columns and df[c].notna().any()]
        sub = df[present]
        rho = sub.corr(method="spearman", min_periods=MIN_PAIRWISE_N)
        rho_full = rho.reindex(index=order, columns=order)
        not_collected = np.array([c not in present for c in order], dtype=bool)
        return rho_full, not_collected

    def _annotation_color(value, threshold):
        return "white" if abs(value) > threshold else "black"

    def _draw_node_separators(ax, order):
        last = None
        for k, var in enumerate(order):
            prefix = var.split("__", 1)[0]
            if last is not None and prefix != last:
                ax.axvline(k - 0.5, color="red", linewidth=0.4, alpha=0.6)
                ax.axhline(k - 0.5, color="red", linewidth=0.4, alpha=0.6)
            last = prefix

    def make_correlation_fig(rho, not_collected, title):
        n = len(rho)
        fig, ax = plt.subplots(
            figsize=(max(10.0, 0.36 * n), max(10.0, 0.36 * n)),
            constrained_layout=True,
        )
        matrix = np.ma.masked_invalid(rho.to_numpy(dtype=float))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#cccccc")

        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", aspect="equal")

        for i in range(n):
            for j in range(n):
                v = rho.iat[i, j]
                if pd.isna(v):
                    text = "" if (not_collected[i] or not_collected[j]) else "n/a"
                    if text:
                        ax.text(j, i, text, ha="center", va="center", fontsize=5, color="black")
                    continue
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=5,
                    color=_annotation_color(v, threshold=0.6),
                )

        ax.set_xticks(range(n))
        ax.set_xticklabels(rho.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(rho.index, fontsize=6)
        _draw_node_separators(ax, list(rho.columns))

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("Spearman ρ")
        fig.suptitle(title, fontsize=11)
        return fig

    def max_deviation_matrix(rhos, order):
        stack = np.stack([r.to_numpy(dtype=float) for r in rhos.values()], axis=0)
        valid = ~np.isnan(stack)
        n_valid = valid.sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_v = np.nanmax(stack, axis=0)
            min_v = np.nanmin(stack, axis=0)
        dev = max_v - min_v
        dev[n_valid < 2] = np.nan
        np.fill_diagonal(dev, np.nan)
        return pd.DataFrame(dev, index=order, columns=order)

    def make_max_deviation_fig(dev_df):
        n = len(dev_df)
        fig, ax = plt.subplots(
            figsize=(max(10.0, 0.36 * n), max(10.0, 0.36 * n)),
            constrained_layout=True,
        )
        arr = dev_df.to_numpy(dtype=float)
        matrix = np.ma.masked_invalid(arr)
        cmap = plt.get_cmap("Reds").copy()
        cmap.set_bad("#cccccc")

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="equal")

        for i in range(n):
            for j in range(n):
                v = arr[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=5,
                    color=_annotation_color(v, threshold=1.0),
                )

        ax.set_xticks(range(n))
        ax.set_xticklabels(dev_df.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(dev_df.index, fontsize=6)
        _draw_node_separators(ax, list(dev_df.columns))

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("max ρ − min ρ across studies")
        fig.suptitle(
            "Max absolute deviation across studies (bright = studies disagree)",
            fontsize=11,
        )
        return fig

    return (
        canonical_order,
        make_correlation_fig,
        make_max_deviation_fig,
        max_deviation_matrix,
        prepare_frame,
        study_correlation,
    )


@app.cell
def _(canonical_order, flats, prepare_frame):
    frames_corr = {study: prepare_frame(df) for study, df in flats.items()}
    corr_order = canonical_order(frames_corr)
    return corr_order, frames_corr


@app.cell
def _(
    MIN_PAIRWISE_N,
    corr_order,
    frames_corr,
    make_correlation_fig,
    mo,
    study_correlation,
):
    rhos = {}
    figs_corr = {}
    for _study, _df in frames_corr.items():
        _rho, _nc = study_correlation(_df, corr_order)
        rhos[_study] = _rho
        figs_corr[_study] = make_correlation_fig(
            _rho,
            _nc,
            f"{_study} — Spearman ρ "
            f"(blank = variable not collected; 'n/a' = pairwise N < {MIN_PAIRWISE_N})",
        )

    mo.md(
        f"Spearman rank correlation matrices were computed for "
        f"{len(rhos)} cohorts ({', '.join(rhos.keys())}); each is "
        f"shown in its own tab below."
    )
    return figs_corr, rhos


@app.cell
def _(MIN_PAIRWISE_N, figs_corr, mo):
    mo.vstack([
        mo.md(
            "**Figure 3.** Within-cohort Spearman rank correlation "
            "matrices, one tab per cohort. All matrices use the "
            "canonical variable order described in §3 and the same "
            "diverging colour scale (RdBu_r, [−1, +1]); annotated "
            "values are ρ. Blank cells indicate the variable was not "
            "collected by that cohort; cells annotated `n/a` indicate "
            f"the pairwise sample size was below {MIN_PAIRWISE_N}. "
            "Variables are grouped by Gen3 node (red separators)."
        ),
        mo.ui.tabs(figs_corr),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Cross-cohort correlation deviation

    To summarise disagreement between cohorts, we computed for each
    variable pair the difference between the maximum and minimum
    Spearman ρ observed across cohorts (`max ρ − min ρ`). Cells in
    which fewer than two cohorts contribute a valid ρ (i.e.
    cohort-exclusive variables, or pairs with insufficient pairwise N
    in any cohort) and the matrix diagonal are masked. Higher values
    therefore identify variable pairs whose estimated correlation is
    least consistent across cohorts; under perfect agreement the
    statistic is 0, and the worst-case theoretical value (sign flip
    from +1 to −1) is 2.

    The matrix is presented as a triage tool: it identifies pairs
    warranting closer inspection of measurement procedures, units, or
    coding, but does not adjudicate which cohort estimate is correct.
    """)
    return


@app.cell
def _(corr_order, make_max_deviation_fig, max_deviation_matrix, mo, rhos):
    deviation_df = max_deviation_matrix(rhos, corr_order)
    deviation_fig = make_max_deviation_fig(deviation_df)

    # Sortable long-form table so the reader can find the brightest cells
    # without eyeballing the matrix.
    _long = (
        deviation_df.stack()
        .reset_index()
        .rename(columns={"level_0": "var_1", "level_1": "var_2", 0: "max_dev"})
    )
    # Keep only one half of the symmetric matrix.
    _long = _long[_long["var_1"] < _long["var_2"]].sort_values("max_dev", ascending=False)

    mo.vstack(
        [
            deviation_fig,
            mo.md(
                "**Figure 4.** Cross-cohort correlation deviation, "
                "computed as `max ρ − min ρ` across cohorts for each "
                "variable pair. Cells are coloured on a sequential "
                "scale (Reds, [0, 2]); brighter cells indicate greater "
                "between-cohort disagreement on the corresponding "
                "pair. The diagonal and pairs estimated in fewer than "
                "two cohorts are masked. Variables follow the "
                "canonical order used in Figure 3."
            ),
            mo.md(
                "**Table 2.** Pairwise correlation deviations sorted "
                "in descending order (upper triangle only). Pairs at "
                "the top of the table contribute the brightest cells "
                "in Figure 4."
            ),
            mo.ui.table(_long.reset_index(drop=True), page_size=30),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Summary

    Section 1 documented cohort sizes and baseline distributions.
    Section 2 characterised variable- and subject-level missingness
    and identified outcome and follow-up coverage. Section 3 reported
    within-cohort Spearman correlation structure across a
    pre-specified clinical panel. Section 4 quantified cross-cohort
    correlation deviation as a triage statistic for measurement
    disagreement.

    The reported statistics are descriptive and intended to inform
    subsequent data-cleaning, harmonisation, and analytic decisions.
    Pairs with the largest cross-cohort deviation in Figure 4 and
    Table 2 are the natural entry points for follow-up review of
    measurement procedures, units, and coding conventions.

    This notebook does not write any artefacts to disk; the
    command-line scripts in `scripts/` remain the canonical source for
    reproducible output files.
    """)
    return


if __name__ == "__main__":
    app.run()
