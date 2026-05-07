"""Step 5 — Within-study Spearman correlation matrices.

Why this analysis exists
------------------------
Steps 1–4 told us *what is in the data* and *what is missing*. This step asks
the next question: do the variables relate to each other in the ways the
clinical literature would predict? Cholesterol and LDL should rise together,
height and weight likewise, age should track upward with most cardiovascular
variables. If a known relationship is present in one study and absent in the
other, the *fourth* study (or in our case, the second) probably has a data
problem worth investigating before any modelling.

The frame here is **internal consistency, not discovery.** Treat strong
expected correlations as a sanity tick. Treat surprising or weak ones as a
flag to investigate, not a finding to report.

A biologist's guide to what the script is doing
-----------------------------------------------
1. **Why Spearman, not Pearson.**
   Spearman correlates the *ranks* of values, not the values themselves. So
   a subject with a triglyceride value 10× the mean barely budges the
   correlation — that subject is just "the highest", same as if they were
   1.5× the mean. Pearson, by contrast, would let that one subject swing the
   number. Spearman also catches any *monotonic* relationship (always going
   up, or always going down), not only straight-line ones. Most clinical
   variables are skewed and contain outliers, which is exactly the situation
   Spearman is designed for.

2. **Why ranks work for ordinal and binary variables too.**
   Education coded as `low=0, medium=1, high=2` and yes/no flags coded as
   `0/1` rank perfectly well. Spearman between a 0/1 variable and a
   continuous one is mathematically equivalent to a *point-biserial rank
   correlation* — a known, valid statistic. We deliberately exclude
   multi-level nominal variables (e.g. `diabetes_type`, `diabetes_therapy`)
   because they have no natural order; ranking them would be meaningless.

3. **Why pairwise-complete observations.**
   For each pair of variables we use only the subjects who have a value for
   *both* of them. We don't drop a whole subject because *one* lab is
   missing. This keeps each cell's sample size as large as possible, but it
   means different cells in the same matrix may rest on different N. We mask
   any cell with N < 30 ("n/a") because Spearman on tiny samples is noise.

4. **Why one canonical variable order.**
   Every matrix is rendered with the same row/column order. We derive it
   *once* by hierarchically clustering the pooled (CDAH + EDCAD-PMS)
   correlation matrix, so variables that "behave alike" sit next to each
   other. This makes the per-study heatmaps directly comparable by eye:
   if the lipid block glows red in one study and is pale in the other, the
   eye catches it instantly. Different orderings would defeat the whole
   point of the exercise.

5. **What the max-deviation matrix shows.**
   For each pair of variables we compute the difference between the largest
   and smallest correlation observed across studies (`max ρ − min ρ`).
   Bright cells are pairs where studies *disagree* most. This is a triage
   map — it does not say which study is right, only that they tell different
   stories about that pair. Where this analysis grows to four studies, the
   same matrix highlights the outlier without you having to compare four
   panels by hand.

6. **What this analysis does NOT do.**
   It is not a hypothesis test. There are no p-values reported. It does not
   adjust for confounders. It is a visual data-quality check, full stop.

Outputs
-------
- ``outputs/correlation_<study>.png`` / ``.csv`` per study
- ``outputs/correlation_max_deviation.png`` / ``.csv``
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

CDAH_CSV = "outputs/cdah_flat.csv"
EDCAD_CSV = "outputs/edcad_pms_flat.csv"

PNG_CDAH = "outputs/correlation_cdah.png"
PNG_EDCAD = "outputs/correlation_edcad_pms.png"
PNG_DEV = "outputs/correlation_max_deviation.png"
CSV_CDAH = "outputs/correlation_cdah.csv"
CSV_EDCAD = "outputs/correlation_edcad_pms.csv"
CSV_DEV = "outputs/correlation_max_deviation.csv"

# Spearman is unreliable on tiny pairwise samples — mask cells with fewer
# than this many subjects having values for *both* variables.
MIN_PAIRWISE_N = 30

# Continuous variables: numeric measurements, used as-is (after coercing).
CONTINUOUS_VARS: tuple[str, ...] = (
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

# Ordinal variables: have a natural low→high order. Coded as integers so
# Spearman ranks them correctly. The numeric *values* don't matter — only
# their order — because Spearman uses ranks.
ORDINAL_CODINGS: dict[str, dict[str, int]] = {
    "demographic__education": {"low": 0, "medium": 1, "high": 2},
    "exposure__smoking_status": {"never": 0, "former": 1, "current": 2},
}

# Binary variables: coded 0/1. Spearman against a 0/1 column is a valid
# point-biserial rank correlation.
BINARY_CODINGS: dict[str, dict[str, int]] = {
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

# Master inventory of analysis variables, in stable input order. Used as the
# upper bound on what *might* be included; per-study presence is checked at
# runtime.
ALL_VARS: list[str] = (
    list(CONTINUOUS_VARS) + list(ORDINAL_CODINGS) + list(BINARY_CODINGS)
)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric analysis frame: continuous coerced, ordinals/binaries coded.

    CDAH stores alcohol as ``exposure__drinks_per_week_string`` (numeric
    values stored as text); EDCAD-PMS uses ``exposure__drinks_per_week``.
    We unify both into ``exposure__drinks_per_week`` for analysis only —
    the source CSVs are untouched.
    """
    df = df.copy()

    # Alcohol harmonisation: fold the CDAH "string"-typed numeric column into
    # the canonical name so the two studies share one variable.
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
            out[col] = (
                df[col].astype("string").str.strip().str.lower().map(mapping)
            )
    for col, mapping in BINARY_CODINGS.items():
        if col in df.columns:
            out[col] = (
                df[col].astype("string").str.strip().str.lower().map(mapping)
            )
    return out


def _populated_columns(df: pd.DataFrame) -> list[str]:
    """Columns that actually carry signal (present + at least one value)."""
    return [c for c in df.columns if df[c].notna().any()]


def canonical_order(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Derive the single variable order used by every matrix.

    Strategy: hierarchically cluster the pooled correlation matrix on the
    *intersection* of populated variables across studies (so the clustering
    input is dense), then append study-exclusive variables at the end,
    grouped by node prefix. This keeps the clustered "core" intact and
    pushes asymmetric variables to the periphery rather than letting them
    fragment the layout.
    """
    populated = {k: set(_populated_columns(df)) for k, df in frames.items()}
    shared = [c for c in ALL_VARS if all(c in cols for cols in populated.values())]

    if len(shared) >= 2:
        # Pool subjects from both studies and rank-correlate. min_periods
        # leaves cells NaN if pairwise N is too small to trust.
        pooled = pd.concat(
            [df[shared] for df in frames.values()], axis=0, ignore_index=True
        )
        rho = pooled.corr(method="spearman", min_periods=MIN_PAIRWISE_N)

        # Distance for clustering = 1 - |rho|. NaN cells (rare on pooled
        # data of populated vars) are treated as "no relationship" so they
        # don't blow up the clustering.
        dist = (1.0 - rho.abs()).fillna(1.0).to_numpy(dtype=float)
        np.fill_diagonal(dist, 0.0)
        # Symmetrise to absorb tiny float asymmetries before squareform.
        dist = (dist + dist.T) / 2.0

        condensed = squareform(dist, checks=False)
        if np.any(condensed):
            link = linkage(condensed, method="average")
            ordered_shared = [shared[i] for i in leaves_list(link)]
        else:
            ordered_shared = shared
    else:
        ordered_shared = shared

    # Append exclusives (populated in at least one study but not all),
    # grouped by node prefix for readability.
    seen = set(ordered_shared)
    union_populated = set().union(*populated.values())
    exclusives = [c for c in ALL_VARS if c not in seen and c in union_populated]
    exclusives.sort(key=lambda c: (c.split("__", 1)[0], c))
    return ordered_shared + exclusives


def study_correlation(
    df: pd.DataFrame, order: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-study Spearman matrix in canonical order, plus a 'not collected' mask.

    Variables absent (or fully missing) in this study become NaN rows/cols in
    the output, distinguished from low-N cells by the boolean mask so the
    plotter can render them differently.
    """
    present = [c for c in order if c in df.columns and df[c].notna().any()]
    sub = df[present]
    # Pairwise-complete Spearman: we keep every subject who has both values
    # in the pair under consideration, rather than dropping subjects who
    # are missing any single variable.
    rho = sub.corr(method="spearman", min_periods=MIN_PAIRWISE_N)
    rho_full = rho.reindex(index=order, columns=order)
    not_collected = np.array([c not in present for c in order], dtype=bool)
    return rho_full, not_collected


def _annotation_color(value: float, threshold: float) -> str:
    """White text on dark cells, black on light — keeps annotations legible."""
    return "white" if abs(value) > threshold else "black"


def _draw_node_separators(ax: plt.Axes, order: list[str]) -> None:
    """Red separators between node-prefix groups (matches missingness.py style)."""
    last = None
    for k, var in enumerate(order):
        prefix = var.split("__", 1)[0]
        if last is not None and prefix != last:
            ax.axvline(k - 0.5, color="red", linewidth=0.4, alpha=0.6)
            ax.axhline(k - 0.5, color="red", linewidth=0.4, alpha=0.6)
        last = prefix


def plot_correlation(
    rho: pd.DataFrame,
    not_collected: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """Per-study heatmap. Diverging RdBu_r on [-1, 1]; gray for masked cells."""
    n = len(rho)
    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.36 * n), max(10.0, 0.36 * n)),
        constrained_layout=True,
    )
    matrix = np.ma.masked_invalid(rho.to_numpy(dtype=float))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#cccccc")

    im = ax.imshow(
        matrix, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", aspect="equal"
    )

    for i in range(n):
        for j in range(n):
            v = rho.iat[i, j]
            if pd.isna(v):
                # Either: (a) at least one of the two variables wasn't
                # collected by this study — leave blank so it visually reads
                # as "absent"; or (b) both vars were collected but their
                # pairwise N fell under MIN_PAIRWISE_N — annotate "n/a".
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def max_deviation_matrix(
    rhos: dict[str, pd.DataFrame], order: list[str]
) -> pd.DataFrame:
    """max ρ − min ρ across studies per cell; NaN where < 2 studies have a value.

    A bright cell means the studies disagree most about that pair — go look
    at it. The diagonal is forced to NaN: comparing a variable to itself is
    always 1.0 and always agrees, so it contributes nothing to triage.
    """
    stack = np.stack([r.to_numpy(dtype=float) for r in rhos.values()], axis=0)
    valid = ~np.isnan(stack)
    n_valid = valid.sum(axis=0)
    # Cells where every study is NaN (study-exclusive variable pairs) emit
    # an "All-NaN slice" warning; we mask them on the next line, so suppress.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        max_v = np.nanmax(stack, axis=0)
        min_v = np.nanmin(stack, axis=0)
    dev = max_v - min_v
    dev[n_valid < 2] = np.nan
    np.fill_diagonal(dev, np.nan)
    return pd.DataFrame(dev, index=order, columns=order)


def plot_max_deviation(dev_df: pd.DataFrame, out_path: str) -> None:
    """Sequential Reds heatmap on [0, 2]; gray for masked cells."""
    n = len(dev_df)
    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.36 * n), max(10.0, 0.36 * n)),
        constrained_layout=True,
    )
    arr = dev_df.to_numpy(dtype=float)
    matrix = np.ma.masked_invalid(arr)
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#cccccc")

    # Range [0, 2]: a flip from ρ=+1 to ρ=−1 is the worst-case, so cells
    # where studies fundamentally disagree about sign light up brightest.
    im = ax.imshow(
        matrix, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="equal"
    )

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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    raw = {
        "CDAH": pd.read_csv(CDAH_CSV),
        "EDCAD-PMS": pd.read_csv(EDCAD_CSV),
    }
    frames = {k: prepare_frame(v) for k, v in raw.items()}
    for k, df in frames.items():
        n_vars = len(_populated_columns(df))
        print(f"[{k}] {n_vars} populated analysis vars, n={len(df)} subjects")

    order = canonical_order(frames)
    print(f"\ncanonical variable order ({len(order)} vars):")
    for c in order:
        print(f"  {c}")

    paths = {
        "CDAH": (CSV_CDAH, PNG_CDAH),
        "EDCAD-PMS": (CSV_EDCAD, PNG_EDCAD),
    }
    rhos: dict[str, pd.DataFrame] = {}
    for study, df in frames.items():
        rho, not_collected = study_correlation(df, order)
        rhos[study] = rho

        out_csv, out_png = paths[study]
        rho.to_csv(out_csv)
        plot_correlation(
            rho,
            not_collected,
            f"{study} — Spearman ρ "
            f"(blank = variable not collected; 'n/a' = pairwise N < {MIN_PAIRWISE_N})",
            out_png,
        )
        print(f"\nwrote {out_csv}")
        print(f"wrote {out_png}")

    dev_df = max_deviation_matrix(rhos, order)
    dev_df.to_csv(CSV_DEV)
    plot_max_deviation(dev_df, PNG_DEV)
    print(f"\nwrote {CSV_DEV}")
    print(f"wrote {PNG_DEV}")


if __name__ == "__main__":
    main()
