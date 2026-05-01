import marimo

__generated_with = "0.0.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # ACDC Exploratory Analysis

        Pull all ACDC clinical metadata from Gen3 via `gen3-metadata` and inspect
        each node as raw JSON or as a pandas DataFrame.
        """
    )
    return


@app.cell
def _():
    from gen3_metadata.gen3_metadata_parser import fetch_all_metadata

    key_file = "path/to/credentials.json"
    result = fetch_all_metadata(key_file, "program1", "project1")

    # Access each node as raw JSON
    result.subject          # dict
    result.demographic      # dict

    # Or get DataFrames
    dfs = result.to_df()
    dfs.subject             # pandas DataFrame
    dfs.demographic         # pandas DataFrame
    return dfs, fetch_all_metadata, key_file, result


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
