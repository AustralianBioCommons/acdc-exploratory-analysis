"""Flatten clinical nodes into one row per subject, per project.

Usage: poetry run python scripts/flatten.py [PROJECT_CODE ...]
       defaults to: CDAH EDCAD-PMS

Join graph:
    subject (PK: submitter_id)
       ▲
       │ clinical_descriptor.subjects -> subject.submitter_id
       │
    clinical_descriptor (PK: submitter_id)
       ▲
       │ <leaf>.clinical_descriptors -> clinical_descriptor.submitter_id
       │
    leaf nodes: demographic, medical_history, blood_pressure_test,
                exposure, lab_result, medication

Leaf-node columns are prefixed with `<node>__` to avoid name collisions
(e.g. `heart_rate` exists in both medical_history and blood_pressure_test).
"""

import sys

import pandas as pd
from gen3_metadata.gen3_metadata_parser import fetch_all_metadata

KEY_FILE = "/Users/harrijh/keys/acdc_api_key_staging.json"
PROGRAM = "program1"
DEFAULT_PROJECTS = ("CDAH", "EDCAD-PMS")
LEAF_NODES = (
    "demographic",
    "medical_history",
    "blood_pressure_test",
    "exposure",
    "lab_result",
    "medication",
)
DROP_COMMON = ("project_id", "data_release", "data_release_date", "id", "type")


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
    # `submitter_id` was already in the rename above via the loop; ensure final name.
    df = df.rename(columns={**rename, "submitter_id": f"{node_name}_submitter_id"})
    # Strip the `__` prefix off the submitter_id we just renamed.
    if f"{node_name}__submitter_id" in df.columns:
        df = df.rename(columns={f"{node_name}__submitter_id": f"{node_name}_submitter_id"})
    return df


def flatten(dfs) -> pd.DataFrame:
    subject = getattr(dfs, "subject").copy().rename(
        columns={"submitter_id": "subject_submitter_id"}
    )

    # clinical_descriptor: keep its own un-prefixed columns since it's a hub, not a leaf.
    # Filter to baseline only so each subject maps to a single CD (handles EDCAD-PMS
    # follow-up CDs and prevents 1:N explosion when leaf nodes also have follow-ups).
    cd = getattr(dfs, "clinical_descriptor").copy()
    if "baseline_measure" in cd.columns:
        cd = cd[cd["baseline_measure"] == True]  # noqa: E712
    cd["subject_submitter_id"] = cd["subjects"].apply(fk_submitter_id)
    cd = cd.drop(columns=[c for c in list(DROP_COMMON) + ["subjects"] if c in cd.columns])
    cd = cd.rename(columns={"submitter_id": "clinical_descriptor_submitter_id"})

    flat = subject.merge(cd, on="subject_submitter_id", how="left")
    expected_rows = len(subject)

    for node in LEAF_NODES:
        leaf = getattr(dfs, node)
        if leaf.empty:
            print(f"  skip {node}: 0 records")
            continue
        slim = _slim(leaf, "clinical_descriptors", "clinical_descriptor_submitter_id", node)
        flat = flat.merge(slim, on="clinical_descriptor_submitter_id", how="left")
        if len(flat) != expected_rows:
            print(
                f"  WARN {node}: row count changed {expected_rows} -> {len(flat)} "
                f"(likely 1:N from clinical_descriptor)"
            )
            expected_rows = len(flat)
    return flat


def run(project_code: str) -> pd.DataFrame:
    result = fetch_all_metadata(KEY_FILE, PROGRAM, project_code)
    flat = flatten(result.to_df())
    out = f"outputs/{project_code.lower().replace('-', '_')}_flat.csv"
    flat.to_csv(out, index=False)
    print(f"\n[{project_code}] shape={flat.shape} -> {out}")
    return flat


if __name__ == "__main__":
    projects = sys.argv[1:] or list(DEFAULT_PROJECTS)
    for p in projects:
        run(p)
