# Current stable Python environment

Dependencies are pinned to stable releases checked against PyPI on 2026-11-01.
Run from this repository's root with Python 3.14.10 (see `.python-version`):

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python scripts/validate_notebooks.py
python scripts/smoke_test.py
python -m jupyterlab
```

Open the notebook and run cells in order. The kernel must use `.venv`.
Python 3.14.10 is selected for the compiled dependency stack. Python 3.12.16
was also used for baseline compatibility checks. Python 3.15.2 requires source
builds for dependencies without published wheels (including PyYAML, wordcloud,
and python-crfsuite where used). The Windows validation environment has no C++
build tools, so this change uses the newest runtime with available dependency wheels.

## Data and validation

Put these original input files under `dataset/`, or set the `DATA_DIR` environment variable: `sample30.csv`.

The validation script checks notebook structure, compiles all transformed code cells,
and imports declared dependencies. It does not execute training or claim to
reproduce results without the original datasets. Saved code-cell outputs were cleared to avoid presenting results from the old
environment as current. Rerun to generate results.

Install the NLTK datasets used by the notebooks:

```sh
python -m nltk.downloader punkt_tab wordnet stopwords averaged_perceptron_tagger_eng omw-1.4
```

Retrain and export the pickle files in notebook order. Pickles produced by older scikit-learn/XGBoost versions are not guaranteed compatible with this environment. Existing model artifacts have not been regenerated.


Version sources: [PyPI](https://pypi.org/), [Python releases](https://www.python.org/downloads/), [checkout releases](https://github.com/actions/checkout/releases), and [setup-python releases](https://github.com/actions/setup-python/releases).
