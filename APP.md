# Review Compass

A working product discovery app over the repository's original review dataset.
Choose favorite products, search the catalog, and inspect the evaluation page.
Favorites stay in browser local storage and can be cleared. The serving artifact
contains aggregate product signals, not usernames, raw reviews or user histories.

## Hosted application

[Open Review Compass](https://review-compass-ashik.swoonysnipe19.chatgpt.site)

The hosted Sites edition performs inference in the browser. The same interface
and scoring representation are served by Flask below. Sites does not host a
Python/Flask process; do not treat the hosted site as the Flask API endpoint.
The Site starts owner-private and requires its owner's ChatGPT sign-in.

## Run Flask

Use Python 3.14 as specified in .python-version. From the repository root:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-web.txt
waitress-serve --host=127.0.0.1 --port=8080 app:app
```

Visit http://127.0.0.1:8080. A trained, JSON-only artifact is committed in
web/model.json, so serving does not download data, load historical pickle files,
or retrain on each request. Waitress is used instead of Flask's development server.

```sh
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/api/recommend -H "Content-Type: application/json" -d '{"liked_ids":[],"limit":6}'
```

GET /api/products lists valid IDs. POST /api/recommend accepts liked_ids,
excluded_ids, limit (1–20), category and query. IDs must be known products;
lists are capped at 50 and bodies at 16 KiB. GET /api/evaluation returns the
held-out evaluation. No login, upload or arbitrary model-loading endpoint exists.

## Reproduce and rebuild

```sh
python -m pip install -r requirements.txt -r requirements-web.txt
python -m scripts.recommender
python -m unittest discover -s tests
python scripts/check_web.py
```

The last check requires Node and compares six Python/browser ranking scenarios.
The training command performs validation selection, final evaluation, then fits
the deployment artifact on all source pairs. Results are saved in
results/recommendation-evaluation.json. The notebook also executes this evaluation
alongside the separate held-out review-sentiment classifier.

After retraining, commit the refreshed web/model.json and evaluation report.
For Sites, copy web/index.html to dist/index.html and the remaining web files to
dist/assets/, then package and publish that exact source through Sites. The model
card explains what the metrics establish and their limitations.
