"""Step 2 — missing-data assessment for the per-subject flat CSVs.

Four views, each answering a different question:

1. Completeness summary (stdout) — variable x study, % non-missing.
2. Variable completeness heatmap (PNG) — study x variable, OrRd gradient on
   % present, gray cells where a study didn't collect a variable.
3. Subject x variable heatmap (PNG) — binary present/missing, subjects
   clustered so systematic patterns surface.
4. CV death censoring note (stdout) — per study, regex-detect any death/
   outcome/follow-up columns and report counts; flag whether survival
   analysis is feasible from the current pull.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

CDAH_CSV = "outputs/cdah_flat.csv"
EDCAD_CSV = "outputs/edcad_pms_flat.csv"
HEATMAP_PNG = "outputs/missingness_heatmap.png"
COMPLETENESS_PNG = "outputs/completeness_heatmap.png"

DASH = "—"

STUDIES: tuple[tuple[str, str], ...] = (
    ("CDAH", CDAH_CSV),
    ("EDCAD-PMS", EDCAD_CSV),
)

# Bookkeeping / ID columns excluded from completeness analysis.
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

# Patterns to detect potential CV-death / outcome / follow-up columns.
CV_DEATH_PATTERNS = (
    r"death",
    r"deceased",
    r"mortal",
    r"vital_status",
    r"cv_event",
    r"incident_cv",
    r"censor",
)
FOLLOW_UP_PATTERNS = (
    r"follow_up_time",
    r"followup_time",
    r"fu_time",
    r"follow_up",
)
ANY_PATTERNS = CV_DEATH_PATTERNS + FOLLOW_UP_PATTERNS


def eligible_columns(df: pd.DataFrame) -> list[str]:
    """Return analysable columns: drop FK/PK plumbing and Gen3 bookkeeping."""
    return [
        c
        for c in df.columns
        if c not in EXCLUDE_EXACT and not c.endswith("_submitter_id")
    ]


def union_variables(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Union of eligible variables across studies, in node-prefix-grouped order."""
    all_vars: list[str] = []
    for df in frames.values():
        for c in eligible_columns(df):
            if c not in all_vars:
                all_vars.append(c)
    all_vars.sort(key=lambda c: (c.split("__", 1)[0], c))
    return all_vars


def completeness_matrix(
    frames: dict[str, pd.DataFrame], all_vars: list[str]
) -> pd.DataFrame:
    """Numeric % present, rows=studies, cols=variables, NaN where absent.

    Shared between the stdout summary and the variable-completeness heatmap
    so both views are guaranteed to agree.
    """
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


def completeness_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Variable x study completeness table for stdout.

    Columns per study: pct non-missing and N non-missing. Variables not
    present in a study render as DASH so the absent-vs-empty distinction
    stays visible. Numerics come from `completeness_matrix` so this view
    and the heatmap can never disagree.
    """
    all_vars = union_variables(frames)
    pct_df = completeness_matrix(frames, all_vars)

    rows = []
    for col in all_vars:
        row: dict[str, object] = {"variable": col}
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


def cluster_order(matrix: np.ndarray) -> np.ndarray:
    """Hamming-clustered row order; safe on degenerate (all-equal) inputs."""
    n_rows = matrix.shape[0]
    if n_rows < 2:
        return np.arange(n_rows)
    distances = pdist(matrix, metric="hamming")
    if not np.any(distances):
        return np.arange(n_rows)
    link = linkage(distances, method="average")
    return leaves_list(link)


def plot_heatmap(frames: dict[str, pd.DataFrame], out_path: str) -> None:
    """Variable x subject heatmap, one panel per study, shared variable axis."""
    # Union of variables; variables absent from a study render as a fully-black
    # row in that panel (consistent with "all subjects missing").
    all_vars = union_variables(frames)
    studies = list(frames.items())
    width_ratios = [max(1, len(df)) for _, df in studies]
    fig, axes = plt.subplots(
        1,
        len(studies),
        figsize=(6 + 4 * len(studies), 0.18 * len(all_vars) + 3),
        gridspec_kw={"width_ratios": width_ratios},
        sharey=True,
        constrained_layout=True,
    )
    if len(studies) == 1:
        axes = [axes]

    for ax, (study, df) in zip(axes, studies):
        # vars x subjects matrix, with absent vars filled as missing (0).
        matrix = np.zeros((len(all_vars), len(df)), dtype=np.uint8)
        for i, var in enumerate(all_vars):
            if var in df.columns:
                matrix[i, :] = df[var].notna().to_numpy(dtype=np.uint8)

        # Cluster subjects (columns) by their missingness vector across vars.
        order = cluster_order(matrix.T)
        ordered = matrix[:, order]

        ax.imshow(
            ordered,
            aspect="auto",
            cmap="binary_r",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{study} — n={len(df)} subjects")
        ax.set_xticks([])
        ax.set_xlabel("subjects (clustered)")

        # Horizontal separators between node-prefix groups.
        last_prefix = None
        for i, var in enumerate(all_vars):
            prefix = var.split("__", 1)[0]
            if last_prefix is not None and prefix != last_prefix:
                ax.axhline(i - 0.5, color="red", linewidth=0.4, alpha=0.6)
            last_prefix = prefix

    axes[0].set_yticks(range(len(all_vars)))
    axes[0].set_yticklabels(all_vars, fontsize=6)
    axes[0].set_ylabel("variables (grouped by node)")

    fig.suptitle(
        "Variable x subject missingness, faceted by study "
        "(white = present, black = missing)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_completeness_heatmap(
    frames: dict[str, pd.DataFrame], out_path: str
) -> None:
    """Compact study x variable heatmap of % present (white→orange→red).

    Cells where a study didn't collect the variable are masked and render
    as solid gray with an "N/A" overlay, distinguishing "not collected"
    from "collected but 0% populated".
    """
    all_vars = union_variables(frames)
    pct_df = completeness_matrix(frames, all_vars)

    matrix = np.ma.masked_invalid(pct_df.to_numpy(dtype=float))
    cmap = plt.get_cmap("OrRd").copy()
    cmap.set_bad("#cccccc")

    n_studies, n_vars = matrix.shape
    # Width: 0.28in per variable (annotation legibility). Height: enough for
    # the panel itself plus ~3.5in of vertical x-tick label space below.
    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.28 * n_vars), 4.0 + 0.6 * n_studies),
        constrained_layout=True,
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=100,
    )

    # Cell annotations: numeric % for collected vars, "N/A" for absent.
    for i in range(n_studies):
        for j in range(n_vars):
            val = pct_df.iat[i, j]
            if pd.isna(val):
                text = "N/A"
                color = "black"
            else:
                text = f"{val:.0f}%"
                color = "white" if val >= 60 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)

    ax.set_yticks(range(n_studies))
    ax.set_yticklabels(list(pct_df.index), fontsize=10)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(all_vars, rotation=90, fontsize=6)
    ax.set_xlabel("variables (grouped by node)")

    # Vertical separators between node-prefix groups.
    last_prefix = None
    for j, var in enumerate(all_vars):
        prefix = var.split("__", 1)[0]
        if last_prefix is not None and prefix != last_prefix:
            ax.axvline(j - 0.5, color="red", linewidth=0.4, alpha=0.6)
        last_prefix = prefix

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("% subjects with non-missing value")

    fig.suptitle(
        "Variable completeness by study (gray = variable not collected)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cv_death_note(frames: dict[str, pd.DataFrame]) -> str:
    """Detection-based report; doesn't assume a specific column name."""
    any_re = re.compile("|".join(ANY_PATTERNS), re.IGNORECASE)
    fu_re = re.compile("|".join(FOLLOW_UP_PATTERNS), re.IGNORECASE)

    lines = ["=== CV death / censoring note ==="]
    for study, df in frames.items():
        matches = [c for c in df.columns if any_re.search(c)]
        if not matches:
            lines.append(
                f"[{study}] No CV death / outcome / follow-up columns detected. "
                f"Survival analysis not possible from current pull."
            )
            continue

        lines.append(f"[{study}] detected {len(matches)} candidate column(s):")
        for c in matches:
            non_null = int(df[c].notna().sum())
            total = len(df)
            sample = df[c].dropna().astype(str).unique()[:3].tolist()
            lines.append(
                f"  - {c}: {non_null}/{total} non-empty; "
                f"sample values: {sample}"
            )

        has_fu = any(fu_re.search(c) for c in matches)
        if not has_fu:
            lines.append(
                f"[{study}] No follow-up time variable — any death indicator "
                f"must be treated as a binary outcome with caveats."
            )
    return "\n".join(lines)


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for study, path in STUDIES:
        df = pd.read_csv(path)
        print(f"loaded {study}: {df.shape}")
        frames[study] = df

    print("\n=== Completeness summary (variable x study) ===")
    summary = completeness_summary(frames)
    print(summary.to_string(index=False))

    print()
    print(cv_death_note(frames))

    plot_completeness_heatmap(frames, COMPLETENESS_PNG)
    print(f"\nwrote {COMPLETENESS_PNG}")

    plot_heatmap(frames, HEATMAP_PNG)
    print(f"wrote {HEATMAP_PNG}")


if __name__ == "__main__":
    main()
