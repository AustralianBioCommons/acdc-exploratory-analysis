"""Build a manuscript-style Table 1 from the per-subject flat CSVs.

Rows: clinical variables grouped by domain.
Cols: CDAH | EDCAD-PMS | Overall.
Continuous: mean +/- SD (median; range); N=...
Categorical: n (%) per level.
Cohort-specific variables show '--' in the non-applicable cohort and Overall.
"""

from __future__ import annotations

import pandas as pd

CDAH_CSV = "outputs/cdah_flat.csv"
EDCAD_CSV = "outputs/edcad_pms_flat.csv"
OUT_CSV = "outputs/table_one.csv"
OUT_MD = "outputs/table_one.md"

DASH = "—"  # em dash for missing/NA cells

# (domain, label, column, kind)
SPEC: list[tuple[str, str, str, str]] = [
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

STUDY_COLS = ("CDAH", "EDCAD-PMS", "Overall")


def fmt_continuous(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return DASH
    return (
        f"{s.mean():.1f} ± {s.std():.1f} "
        f"(median {s.median():.1f}; range {s.min():.1f}–{s.max():.1f}); "
        f"N={len(s)}"
    )


def categorical_rows(s: pd.Series) -> list[tuple[str, str]]:
    """Return [(level_label, formatted_count_pct), ...]. Empty if all NaN."""
    s = s.dropna()
    if s.empty:
        return []
    s = s.astype(str)
    counts = s.value_counts()
    n = int(counts.sum())
    out = []
    for level, c in counts.items():
        out.append((level, f"{int(c)} ({c / n * 100:.1f}%)"))
    return out


def build_table(cdah: pd.DataFrame, edcad: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    last_domain = None
    for domain, label, col, kind in SPEC:
        in_cdah = col in cdah.columns
        in_edcad = col in edcad.columns
        if not (in_cdah or in_edcad):
            continue  # column doesn't exist anywhere; skip silently
        # Insert domain header before first row of each domain.
        if domain != last_domain:
            rows.append({"Domain": domain, "Variable": "", "Level": "", **{c: "" for c in STUDY_COLS}})
            last_domain = domain

        overall_applicable = in_cdah and in_edcad

        if kind == "continuous":
            cdah_cell = fmt_continuous(cdah[col]) if in_cdah else DASH
            edcad_cell = fmt_continuous(edcad[col]) if in_edcad else DASH
            overall_cell = fmt_continuous(pooled[col]) if overall_applicable else DASH
            rows.append({
                "Domain": "", "Variable": label, "Level": "",
                "CDAH": cdah_cell, "EDCAD-PMS": edcad_cell, "Overall": overall_cell,
            })
        else:  # categorical
            cdah_levels = categorical_rows(cdah[col]) if in_cdah else []
            edcad_levels = categorical_rows(edcad[col]) if in_edcad else []
            overall_levels = categorical_rows(pooled[col]) if overall_applicable else []
            cdah_n = sum(int(v.split(" ")[0]) for _, v in cdah_levels) if cdah_levels else (0 if in_cdah else None)
            edcad_n = sum(int(v.split(" ")[0]) for _, v in edcad_levels) if edcad_levels else (0 if in_edcad else None)
            overall_n = sum(int(v.split(" ")[0]) for _, v in overall_levels) if overall_levels else None

            rows.append({
                "Domain": "", "Variable": label, "Level": "",
                "CDAH": (f"N={cdah_n}" if in_cdah else DASH),
                "EDCAD-PMS": (f"N={edcad_n}" if in_edcad else DASH),
                "Overall": (f"N={overall_n}" if overall_applicable else DASH),
            })
            # Union of levels, ordered by overall frequency where available else by cdah/edcad
            level_order: list[str] = []
            for src in (overall_levels, cdah_levels, edcad_levels):
                for lvl, _ in src:
                    if lvl not in level_order:
                        level_order.append(lvl)
            cdah_map = dict(cdah_levels)
            edcad_map = dict(edcad_levels)
            overall_map = dict(overall_levels)
            for lvl in level_order:
                rows.append({
                    "Domain": "", "Variable": "", "Level": f"  {lvl}",
                    "CDAH": cdah_map.get(lvl, "0 (0.0%)") if in_cdah else DASH,
                    "EDCAD-PMS": edcad_map.get(lvl, "0 (0.0%)") if in_edcad else DASH,
                    "Overall": overall_map.get(lvl, "0 (0.0%)") if overall_applicable else DASH,
                })

    return pd.DataFrame(rows, columns=["Domain", "Variable", "Level", *STUDY_COLS])


def to_markdown(df: pd.DataFrame) -> str:
    """Pipe-table markdown writer (no tabulate dependency)."""
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


def main() -> None:
    cdah = pd.read_csv(CDAH_CSV)
    edcad = pd.read_csv(EDCAD_CSV)
    print(f"loaded CDAH: {cdah.shape}, EDCAD-PMS: {edcad.shape}")

    cdah_t = cdah.assign(study="CDAH")
    edcad_t = edcad.assign(study="EDCAD-PMS")
    pooled = pd.concat([cdah_t, edcad_t], ignore_index=True, sort=False)

    table = build_table(cdah, edcad, pooled)

    domain_counts = table[table["Variable"].ne("") & table["Domain"].eq("")]
    print(f"variable rows: {len(domain_counts)}")
    print(table.head(15).to_string(index=False))

    table.to_csv(OUT_CSV, index=False)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(table))
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
