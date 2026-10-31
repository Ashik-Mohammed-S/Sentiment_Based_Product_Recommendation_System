# Reference datasets stored in this repository

The reference files are stored under dataset/. Files up to 8 MiB are ordinary files;
larger files are gzip-compressed in bounded parts under dataset/bundled/.
dataset/bundled-index.json records reconstruction details and SHA-256 checksums.
These are complete reference tables, not quick-mode or synthetic samples.

```sh
python scripts/fetch_data.py
python scripts/run_notebook.py --mode full
```

No Kaggle account or source download is needed after cloning the stored data.
Set DATA_OFFLINE=1 to enforce offline restoration. Scientific dependencies and the
spaCy English model, where used, still need installation. External pinned URLs
remain as provenance and fallback only. Different unrecognized local data is left untouched.

| File | Rows / tokens | Provenance |
| --- | --- | --- |
| sample30.csv | 30000 | [Source](https://raw.githubusercontent.com/Ashik-Mohammed-S/Sentiment_Based_Product_Recommendation_System/6c582130b7c5bc77b9a1535585adde6a7c9008f7/dataset/sample30.csv) |

The stored data retains its original upstream terms and attribution; this project
does not relicense third-party datasets. Source metadata is in dataset-sources.json.
