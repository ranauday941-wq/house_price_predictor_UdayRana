"""Runs the complete pipeline for one uploaded dataset and stores the artifacts.

Artifacts written to <run_dir>:
    results.json   - everything the web pages display
    bundle.joblib  - trained model + defaults used by the prediction form
    clean.joblib   - cleaned DataFrame (for the interactive feature explorer)
    charts/*.png   - the dashboard images
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import charts, pipeline as pl


def _py(o):
    """JSON helper for numpy types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not serialisable: %r" % type(o))


def _missing_table(df_raw, target, dropped, sel, num_cols):
    dropped_reason = {d["column"]: d["reason"] for d in dropped}
    low = {d["column"] for d in sel["dropped_low_corr"]}
    col_lin = {d["column"] for d in sel["dropped_collinear"]}
    rows = []
    for c in df_raw.columns:
        n = int(df_raw[c].isna().sum())
        if not n:
            continue
        if c == target:
            how, after = "Rows removed (target must be known)", "-"
        elif c in dropped_reason:
            how, after = "Column dropped (%s)" % dropped_reason[c].split(" (")[0], "-"
        elif c in low or c in col_lin:
            how, after = "Column removed by feature selection", "-"
        elif c in num_cols:
            how, after = "Median imputation", 0
        else:
            how, after = "Most-frequent (mode) imputation", 0
        rows.append({"column": c, "before": n, "before_pct": round(100.0 * n / len(df_raw), 2),
                     "treatment": how, "after": after})
    rows.sort(key=lambda r: -r["before"])
    return rows


def process_dataset(df_raw, filename, run_dir, target_hint=None):
    started = time.time()
    os.makedirs(os.path.join(run_dir, "charts"), exist_ok=True)

    target = pl.detect_target(df_raw, target_hint)
    clean, info = pl.clean_data(df_raw, target)
    if len(clean) < pl.MIN_ROWS:
        raise ValueError("Too few usable rows remain after cleaning (%d)." % len(clean))
    num_all, cat_all = pl.split_columns(clean, target)
    if not num_all and not cat_all:
        raise ValueError("No feature columns are left after cleaning.")

    out_rep = pl.outlier_report(clean, num_all)

    X, y = clean.drop(columns=[target]), clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=pl.RANDOM_STATE)
    sel = pl.select_features(X_train, y_train, num_all, cat_all)
    num_cols, cat_cols = sel["numeric"], sel["categorical"]
    cols = num_cols + cat_cols
    X_train, X_test = X_train[cols], X_test[cols]

    tr = pl.train_models(X_train, X_test, y_train, y_test, num_cols, cat_cols)
    model = tr["model"]
    best_row = next(r for r in tr["results"] if r["name"] == tr["best"])

    # missing values after processing (imputation is part of the fitted pipeline)
    prep = model.regressor_.named_steps["prep"]
    transformed = prep.transform(X[cols])
    missing_after = int(np.isnan(transformed).sum())
    raw_missing = int(df_raw.isna().sum().sum())
    kept_missing = int(X[cols].isna().sum().sum())
    missing_rows = _missing_table(df_raw, target, info["dropped_columns"], sel, num_all)

    form_fields = pl.build_form_fields(X_train, num_cols, cat_cols, tr["importance"])
    defaults = pl.column_defaults(X_train, num_cols, cat_cols)

    # ---- charts -------------------------------------------------------
    corr_order = [d["feature"] for d in sel["correlations"]]
    cat_best = charts.best_category(clean, cat_all, target)
    chart_list = []

    def add(name, title, caption, png):
        if png is None:
            return
        charts.save(png, os.path.join(run_dir, "charts", name + ".png"))
        chart_list.append({"name": name, "title": title, "caption": caption})

    add("target_dist", "Price distribution",
        "House prices are right-skewed, so the model learns log(price) and converts predictions back.",
        charts.target_distribution(clean, target))
    add("missing", "Missing values", "Columns with missing values in the uploaded file.",
        charts.missing_values(missing_rows))
    add("boxplots", "Box plots of the most price-related features",
        "Orange dots are IQR outliers; they are capped to the IQR fences inside the model pipeline.",
        charts.box_plots(clean, corr_order))
    add("scatter", "Scatter plots against price", "Each dot is a house; the red line is the linear trend.",
        charts.scatter_plots(clean, corr_order, target))
    if len(num_all) >= 2:
        add("heatmap", "Correlation heatmap", "Strongly correlated feature pairs carry the same information.",
            charts.correlation_heatmap(clean, num_all, target))
    if cat_best:
        add("category", "Price by %s" % cat_best, "Categorical column whose groups differ most in average price.",
            charts.category_price(clean, cat_best, target))
    add("importance", "Feature importance", "Which columns influence the predicted price the most.",
        charts.feature_importance(tr["importance"]))
    add("models", "Model comparison", "Best model is chosen by cross-validation, then scored on unseen test data.",
        charts.model_comparison(tr["results"]))
    add("actual_pred", "Actual vs predicted", "Points close to the dashed line are accurate predictions.",
        charts.actual_vs_predicted(y_test.values, tr["test_pred"]))
    add("residuals", "Residual analysis", "Errors should be centred on zero without a strong pattern.",
        charts.residuals(y_test.values, tr["test_pred"]))

    # ---- tables for the summary page -------------------------------------
    dtypes = []
    dropped_names = {d["column"] for d in info["dropped_columns"]}
    for c in df_raw.columns:
        if c == target:
            role = "Target"
        elif c in dropped_names:
            role = "Dropped"
        elif c in cols:
            role = "Numeric" if c in num_cols else "Categorical"
        else:
            role = "Removed (feature selection)"
        dtypes.append({"column": c, "dtype": str(df_raw[c].dtype), "non_null": int(df_raw[c].notna().sum()),
                       "unique": int(df_raw[c].nunique(dropna=True)), "role": role})
    describe = clean[[target] + num_all].describe().T.round(2)
    describe.insert(0, "column", describe.index)

    results = {
        "filename": filename, "target": target, "seconds": round(time.time() - started, 1),
        "overview": {
            "rows_raw": info["rows_raw"], "cols_raw": info["cols_raw"], "rows_clean": info["rows_clean"],
            "cols_model": len(cols), "numeric": len(num_cols), "categorical": len(cat_cols),
            "duplicates_removed": info["duplicates_removed"], "target_outliers_removed": info["target_outliers_removed"],
            "missing_raw": raw_missing, "missing_kept_before": kept_missing, "missing_after": missing_after,
            "encoded_features": tr["n_encoded_features"],
        },
        "preview_html": df_raw.head(10).to_html(classes="data-table", border=0, na_rep="NaN", index=False),
        "describe_html": describe.to_html(classes="data-table", border=0, index=False),
        "dtypes": dtypes, "missing": missing_rows, "dropped_columns": info["dropped_columns"],
        "log": info["log"], "outliers": out_rep,
        "selection": {k: sel[k] for k in ("dropped_low_corr", "dropped_collinear")},
        "top_corr": sel["correlations"][:10], "kept_numeric": num_cols, "kept_categorical": cat_cols,
        "models": tr["results"], "best_model": tr["best"], "metrics": best_row,
        "importance": tr["importance"][:15], "form_fields": form_fields,
        "explorer_columns": cols, "charts": chart_list,
        "train_rows": len(X_train), "test_rows": len(X_test),
    }

    joblib.dump({"model": model, "columns": cols, "defaults": defaults, "form_fields": form_fields,
                 "target": target, "rmse": best_row["rmse"], "best_model": tr["best"]},
                os.path.join(run_dir, "bundle.joblib"))
    joblib.dump({"clean": clean, "target": target}, os.path.join(run_dir, "clean.joblib"))
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, default=_py)
    return results
