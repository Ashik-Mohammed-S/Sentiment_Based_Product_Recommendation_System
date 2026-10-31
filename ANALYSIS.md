# Sentiment classification and held-out rating recommendation

This is the primary analysis for Sentiment_Based_Product_Recommendation_System. Run `Main.ipynb` or:

```sh
python scripts/run_notebook.py --mode full
```

The implementation is in scripts/analysis.py. It writes results/analysis.json,
results/target_summary.csv and, where appropriate, results/overview.png.
Targets, holdout design, assumptions, baselines and metrics are recorded in the
report itself. Predictive preprocessing is fitted on training data only.
Descriptive EDA reports rates and group denominators rather than claiming predictive accuracy.
Time-series models use an untouched final chronological holdout with expanding-window CV.

The original walkthrough is retained in historical/ for reference. Its old
in-sample or exploratory numbers are not the primary model assessment.
