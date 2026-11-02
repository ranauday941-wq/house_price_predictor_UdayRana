"""Run the full data-mining + ML pipeline from the command line (no Django needed).

    python run_pipeline.py sample_data/house_prices_sample.csv
    python run_pipeline.py path/to/train.csv --target SalePrice --out outputs
"""
import argparse
import os

from predictor.ml import pipeline as pl
from predictor.ml.service import process_dataset


def main():
    ap = argparse.ArgumentParser(description="House price data-mining pipeline")
    ap.add_argument("csv", help="path to a .csv or .xlsx file")
    ap.add_argument("--target", default=None, help="price column (auto-detected when omitted)")
    ap.add_argument("--out", default="outputs", help="folder for charts, model and results.json")
    a = ap.parse_args()

    df = pl.load_dataframe(a.csv, os.path.basename(a.csv))
    r = process_dataset(df, os.path.basename(a.csv), a.out, a.target)
    o = r["overview"]
    print("\n=== DATASET ===")
    print("target column      :", r["target"])
    print("rows (raw -> clean):", o["rows_raw"], "->", o["rows_clean"])
    print("columns raw/model  :", o["cols_raw"], "/", o["cols_model"], "(%d numeric, %d categorical)" % (o["numeric"], o["categorical"]))
    print("\n=== CLEANING LOG ===")
    for line in r["log"]:
        print(" -", line)
    print("\nmissing values     : raw file %d | kept columns %d -> after imputation %d" % (o["missing_raw"], o["missing_kept_before"], o["missing_after"]))
    print("IQR outliers found :", r["outliers"]["total_iqr"], "| Z-score outliers:", r["outliers"]["total_z"])
    print("\n=== FEATURE SELECTION ===")
    print(" removed (low correlation):", [d["column"] for d in r["selection"]["dropped_low_corr"]])
    print(" removed (collinear)      :", [(d["column"], d["correlated_with"], d["r"]) for d in r["selection"]["dropped_collinear"]])
    print("\n=== MODELS (train %d / test %d rows) ===" % (r["train_rows"], r["test_rows"]))
    for m in r["models"]:
        print(" %-18s CV R2 %.3f | test R2 %.3f | RMSE %9.0f | MAE %9.0f | MAPE %.1f%%" % (m["name"], m["cv_r2"], m["r2"], m["rmse"], m["mae"], m["mape"]))
    print(" best model:", r["best_model"])
    print("\ntop features:", [(i["feature"], round(i["importance"], 3)) for i in r["importance"][:6]])
    print("\nSaved charts, model and results.json in:", os.path.abspath(a.out), "(%.1f s)" % r["seconds"])


if __name__ == "__main__":
    main()
