"""Data-mining and machine-learning pipeline for house-price prediction.

This module has NO Django dependency, so it can be used from the web app,
from the command line (run_pipeline.py) or from a notebook.

Steps
-----
1. load_dataframe      - read an uploaded .csv / .xlsx file
2. detect_target       - find the price column
3. clean_data          - drop identifier / mostly-empty / constant columns,
                         remove duplicates, remove extreme target outliers
4. outlier_report      - IQR and Z-score detection for numeric features
5. select_features     - correlation analysis (low-correlation and
                         multicollinear numeric columns are removed)
6. build_preprocessor  - median/mode imputation, IQR capping, StandardScaler,
                         One-Hot encoding
7. train_models        - Ridge, Random Forest, Gradient Boosting, compared
                         with cross-validation; best model evaluated on a
                         held-out test set (RMSE, MAE, R2, MAPE)
"""
import io
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_NAMES = ["saleprice", "sale_price", "price", "house_price", "houseprice", "median_house_value",
                "medv", "sellingprice", "selling_price", "target"]
MAX_MISSING_RATIO = 0.50      # columns with more missing values are dropped
MIN_ROWS = 50


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_dataframe(source, filename):
    """Read a .csv or .xlsx upload (file object, bytes or path) into a DataFrame."""
    ext = str(filename).lower().rsplit(".", 1)[-1]
    if hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        with open(source, "rb") as fh:
            raw = fh.read()

    if ext == "csv":
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                if df.shape[1] == 1:                       # maybe ';' or tab separated
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("The CSV file could not be decoded. Please save it as UTF-8.")
    elif ext in ("xlsx", "xlsm"):
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[0] < MIN_ROWS or df.shape[1] < 3:
        raise ValueError("The dataset is too small. It needs at least %d rows and 3 columns." % MIN_ROWS)
    return df


def detect_target(df, preferred=None):
    """Return the name of the price column."""
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    if preferred and preferred.strip():
        key = preferred.strip().lower().replace(" ", "_")
        if key in cols:
            return cols[key]
        raise ValueError("Target column '%s' was not found in the file." % preferred.strip())
    for name in TARGET_NAMES:
        if name in cols:
            return cols[name]
    raise ValueError("The price column could not be detected automatically. "
                     "Type its exact name in the 'Target column' box (for Kaggle data use SalePrice).")


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _tidy_categoricals(df):
    """Make text columns plain object columns with np.nan for missing values."""
    for c in df.columns:
        if not is_numeric(df[c]):
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


# ----------------------------------------------------------------------------
# 2-3. Cleaning
# ----------------------------------------------------------------------------
def clean_data(df_raw, target):
    """Data cleaning. Returns (clean_df, info) where info documents every step."""
    df = df_raw.copy()
    n_raw, c_raw = df.shape
    log, dropped = [], []

    if not is_numeric(df[target]):          # e.g. "$250,000" -> 250000
        df[target] = df[target].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    missing_target = int(df[target].isna().sum())
    df = df[df[target].notna()].copy()
    if missing_target:
        log.append("Removed %d rows with a missing %s value." % (missing_target, target))

    # --- irrelevant fields ------------------------------------------------
    features = [c for c in df.columns if c != target]
    for c in features:
        s = df[c]
        reason = None
        if re.fullmatch(r"(id|index|unnamed: ?0|row_?id|house_?id|property_?id)", c.lower()):
            reason = "identifier column (no predictive meaning)"
        elif s.nunique(dropna=True) <= 1:
            reason = "constant column (only one value)"
        elif s.isna().mean() > MAX_MISSING_RATIO:
            reason = "more than %d%% of values are missing (%.1f%%)" % (MAX_MISSING_RATIO * 100, s.isna().mean() * 100)
        elif not is_numeric(s) and s.nunique(dropna=True) > max(50, 0.5 * len(s)):
            reason = "free-text / identifier-like column (too many unique values)"
        elif is_numeric(s) and s.nunique() == len(s) and pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
            reason = "running serial number"
        if reason:
            dropped.append({"column": c, "reason": reason})
    drop_cols = [d["column"] for d in dropped]
    df = df.drop(columns=drop_cols)
    if drop_cols:
        log.append("Dropped %d irrelevant column(s): %s." % (len(drop_cols), ", ".join(drop_cols)))

    # --- duplicates -------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates().copy()
    n_dup = before - len(df)
    log.append("Removed %d duplicate record(s)." % n_dup if n_dup else "No duplicate records were found.")

    # --- extreme target outliers (3 x IQR rule) --------------------------
    q1, q3 = df[target].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    extreme = (df[target] < lo) | (df[target] > hi)
    n_extreme = int(extreme.sum())
    df = df[~extreme].copy()
    if n_extreme:
        log.append("Removed %d extreme price outlier(s) above %s (3 x IQR rule)."
                   % (n_extreme, format(hi, ",.0f")) if lo <= df[target].min() else
                   "Removed %d extreme price outlier(s) outside [%s, %s] (3 x IQR rule)."
                   % (n_extreme, format(lo, ",.0f"), format(hi, ",.0f")))
    df = _tidy_categoricals(df.reset_index(drop=True))

    info = {
        "rows_raw": n_raw, "cols_raw": c_raw, "rows_clean": len(df), "cols_clean": df.shape[1],
        "duplicates_removed": int(n_dup), "missing_target_rows": missing_target,
        "target_outliers_removed": n_extreme, "dropped_columns": dropped, "log": log,
    }
    return df, info


def split_columns(df, target):
    num = [c for c in df.columns if c != target and is_numeric(df[c])]
    cat = [c for c in df.columns if c != target and not is_numeric(df[c])]
    return num, cat


# ----------------------------------------------------------------------------
# 4. Outliers (features)
# ----------------------------------------------------------------------------
def outlier_report(df, numeric_cols, top=12):
    """Count outliers per numeric column using the IQR rule and the Z-score rule."""
    rows = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.nunique() < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum()) if iqr > 0 else 0
        sd = s.std()
        n_z = int((np.abs((s - s.mean()) / sd) > 3).sum()) if sd > 0 else 0
        rows.append({"column": c, "iqr_outliers": n_iqr, "z_outliers": n_z,
                     "lower_fence": float(lo), "upper_fence": float(hi)})
    rows.sort(key=lambda r: r["iqr_outliers"], reverse=True)
    return {"total_iqr": int(sum(r["iqr_outliers"] for r in rows)),
            "total_z": int(sum(r["z_outliers"] for r in rows)), "top": rows[:top]}


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns to the IQR fences learned on the training data."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        flat = iqr == 0                      # almost constant column: do not cap
        self.lower_[flat] = np.nanmin(X, axis=0)[flat]
        self.upper_[flat] = np.nanmax(X, axis=0)[flat]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ----------------------------------------------------------------------------
# 5. Feature selection (correlation analysis)
# ----------------------------------------------------------------------------
def select_features(X, y, numeric_cols, categorical_cols, min_corr=0.05, max_inter_corr=0.85):
    """Keep numeric columns that correlate with price and are not redundant."""
    corr = {c: float(X[c].corr(y)) for c in numeric_cols}
    corr = {c: (0.0 if np.isnan(v) else v) for c, v in corr.items()}
    low = [c for c in numeric_cols if abs(corr[c]) < min_corr]
    candidates = sorted([c for c in numeric_cols if c not in low], key=lambda c: -abs(corr[c]))

    kept, collinear = [], []
    for c in candidates:
        partner = None
        for k in kept:
            r = X[c].corr(X[k])
            if not np.isnan(r) and abs(r) > max_inter_corr:
                partner = (k, float(r))
                break
        if partner:
            collinear.append({"column": c, "correlated_with": partner[0], "r": round(partner[1], 3)})
        else:
            kept.append(c)
    kept_num = [c for c in numeric_cols if c in kept]
    if not kept_num and not categorical_cols:
        raise ValueError("No usable feature columns were found in the dataset.")
    return {
        "numeric": kept_num, "categorical": list(categorical_cols),
        "dropped_low_corr": [{"column": c, "r": round(corr[c], 3)} for c in low],
        "dropped_collinear": collinear,
        "correlations": sorted(({"feature": c, "corr": round(v, 3)} for c, v in corr.items()),
                               key=lambda d: -abs(d["corr"])),
    }


# ----------------------------------------------------------------------------
# 6-7. Transformation + models
# ----------------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),          # missing numeric -> median
        ("cap", IQRCapper(1.5)),                               # outlier treatment
        ("scale", StandardScaler()),                           # feature scaling
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),   # missing categorical -> mode
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
    ])
    parts = []
    if numeric_cols:
        parts.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        parts.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(parts, remainder="drop")


def candidate_models():
    return {
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, max_features=0.5,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, learning_rate=0.06, max_depth=3,
                                                       subsample=0.8, random_state=RANDOM_STATE),
    }


def make_model(estimator, numeric_cols, categorical_cols):
    pipe = Pipeline([("prep", build_preprocessor(numeric_cols, categorical_cols)), ("model", estimator)])
    # prices are right-skewed: learn log(price) and convert back to real prices
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100),
    }


def original_column(feature_name, categorical_cols):
    """Map a transformed feature name ('cat__Neighborhood_NAmes') to its source column."""
    kind, rest = feature_name.split("__", 1)
    if kind == "num":
        return rest
    matches = [c for c in categorical_cols if rest == c or rest.startswith(c + "_")]
    return max(matches, key=len) if matches else rest


def train_models(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    """Compare the candidate models and return the evaluation results."""
    results, fitted = [], {}
    for name, est in candidate_models().items():
        model = make_model(est, numeric_cols, categorical_cols)
        cv = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
        model.fit(X_train, y_train)
        m = regression_metrics(y_test, model.predict(X_test))
        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        results.append({"name": name, "cv_r2": float(cv.mean()), "train_r2": train_r2, **m})
        fitted[name] = model

    best = max(results, key=lambda r: r["cv_r2"])["name"]      # chosen on CV, NOT on the test set
    best_model = fitted[best]
    rf = fitted["Random Forest"].regressor_
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = rf.named_steps["model"].feature_importances_
    agg = {}
    for n, v in zip(names, imp):
        col = original_column(n, categorical_cols)
        agg[col] = agg.get(col, 0.0) + float(v)
    importance = sorted(({"feature": k, "importance": v} for k, v in agg.items()), key=lambda d: -d["importance"])
    return {"results": results, "best": best, "model": best_model, "importance": importance,
            "n_encoded_features": int(len(names)), "test_pred": best_model.predict(X_test)}


# ----------------------------------------------------------------------------
# Prediction for one custom input
# ----------------------------------------------------------------------------
def build_form_fields(X_train, numeric_cols, categorical_cols, importance, n_fields=8):
    """Choose the most important columns for the interactive prediction form."""
    fields = []
    for item in importance:
        col = item["feature"]
        if col in numeric_cols:
            s = X_train[col]
            fields.append({"name": col, "kind": "number", "default": float(s.median()),
                           "min": float(s.min()), "max": float(s.max()),
                           "integer": bool(np.allclose(s.dropna() % 1, 0))})
        elif col in categorical_cols:
            vc = X_train[col].value_counts()
            fields.append({"name": col, "kind": "choice", "default": str(vc.index[0]),
                           "choices": [str(v) for v in vc.index[:30]]})
        if len(fields) >= n_fields:
            break
    return fields


def column_defaults(X_train, numeric_cols, categorical_cols):
    d = {c: float(X_train[c].median()) for c in numeric_cols}
    d.update({c: str(X_train[c].mode().iloc[0]) for c in categorical_cols})
    return d


def predict_one(model, columns, defaults, values):
    """Predict the price for one house. Unspecified columns use typical (median / mode) values."""
    row = dict(defaults)
    row.update(values)
    frame = pd.DataFrame([row], columns=columns)
    return float(model.predict(frame)[0])
