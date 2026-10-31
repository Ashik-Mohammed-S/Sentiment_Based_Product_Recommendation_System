"""Exercise the upgraded scientific stack with synthetic data; no original datasets needed."""
from pathlib import Path
import ast
import json
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

frame = pd.DataFrame({"group": ["a", "b", "a", "b"], "value": [1., 2., np.nan, 4.]})
frame["value"] = frame["value"].fillna(0)
assert frame["value"].isna().sum() == 0
design = pd.get_dummies(frame, dtype=int)
assert all(np.issubdtype(dtype, np.number) for dtype in design.dtypes)
assert frame.corr(numeric_only=True).shape == (1, 1)
sns.countplot(x=frame["group"])
plt.close("all")
sns.scatterplot(x=frame["value"], y=frame["value"])
plt.close("all")
root = Path(__file__).resolve().parents[1]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.feature_selection import RFE
x = np.arange(80, dtype=float).reshape(20, 4)
y = np.array([0, 1] * 10)
model = LogisticRegression().fit(x, y)
RocCurveDisplay.from_estimator(model, x, y)
plt.close("all")
assert RFE(LogisticRegression(), n_features_to_select=2).fit(x, y).support_.sum() == 2

print("Scientific stack smoke checks passed.")
