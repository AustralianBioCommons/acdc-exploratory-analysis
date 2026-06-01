# acdc-exploratory-analysis

Exploratory analysis of ACDC clinical data pulled from the Gen3 commons via the
[`gen3-metadata`](https://pypi.org/project/gen3-metadata/) package. Notebooks are authored
with [marimo](https://marimo.io/) for interactive, reproducible exploration.

## Quickstart

```bash
# install dependencies
poetry install

# launch the exploratory notebook
poetry run marimo edit notebooks/exploratory.py
```

You will need a Gen3 credentials JSON file. Place it outside the repo (or somewhere
that is gitignored) and update the `key_file` path in the first cell of the notebook.

## Layout

- `notebooks/exploratory.py` — marimo notebook; first cell fetches ACDC metadata.
- `pyproject.toml` — Poetry-managed dependencies (`marimo`, `gen3-metadata==1.4.0`, `pandas`).


## To export to html
```bash
DATETIME=$(date +%Y%m%d_%H%M%S)
poetry run marimo export html --no-include-code notebooks/pipeline.py -o outputs/${DATETIME}_acdc_exploratory_analysis.html
```