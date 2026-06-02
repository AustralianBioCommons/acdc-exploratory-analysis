# ACDC Gen3 demo — boxplot every numeric variable in the `demographic` node,
# split by sex. Designed to be pasted straight into a single Jupyter cell.
# Minimal dependencies: pandas + matplotlib (plus gen3_metadata for the fetch).

import matplotlib.pyplot as plt
import pandas as pd
from gen3_metadata.gen3_metadata_parser import fetch_all_metadata

KEY_FILE = "/home/jovyan/key.txt"
PROGRAM = "program1"
STUDY_ID = "CDAH"
GROUP_COL = "sex"

# Fetch all nodes, then grab the demographic node as a pandas DataFrame.
result = fetch_all_metadata(KEY_FILE, PROGRAM, STUDY_ID)
demo = result.to_df().demographic

# Auto-detect numeric variables. Gen3 columns often come back as object dtype,
# so coerce a copy and keep columns that are mostly parseable as numbers.
demo_num = demo.apply(pd.to_numeric, errors="coerce")
numeric_cols = [
    c
    for c in demo_num.columns
    if c != GROUP_COL
    and demo_num[c].notna().sum() > 0
    and demo_num[c].notna().mean() >= 0.5  # mostly-parseable => treat as numeric
]
print(f"{STUDY_ID}: {len(numeric_cols)} numeric variables — {numeric_cols}")

# One boxplot per numeric variable, grouped by sex.
levels = sorted(demo[GROUP_COL].dropna().unique())
ncols = 3
nrows = -(-len(numeric_cols) // ncols)  # ceil division
fig, axes = plt.subplots(
    nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False
)
for ax, col in zip(axes.flat, numeric_cols):
    data = [
        demo_num.loc[demo[GROUP_COL] == lvl, col].dropna().values for lvl in levels
    ]
    ax.boxplot(data, tick_labels=levels)
    ax.set_title(col)
    ax.set_xlabel(GROUP_COL)
    ax.set_ylabel(col)
for ax in axes.flat[len(numeric_cols):]:  # hide unused panels
    ax.axis("off")
fig.suptitle(f"Numeric variables by {GROUP_COL} ({STUDY_ID})")
fig.tight_layout()
plt.show()
