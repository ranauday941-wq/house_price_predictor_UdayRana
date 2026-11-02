"""All charts of the project (Exploratory Data Analysis + model diagnostics).

The matplotlib object API (Figure) is used instead of pyplot, because pyplot
keeps global state and is not safe inside a multi-threaded web server.
"""
import io
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .pipeline import is_numeric

PRIMARY = "#2563eb"
ACCENT = "#f97316"
GREEN = "#16a34a"
RED = "#dc2626"
GREY = "#64748b"
INK = "#0f172a"


def _fig(w=8.2, h=4.2):
    fig = Figure(figsize=(w, h), dpi=100, facecolor="white")
    return fig


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.grid(True, color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#94a3b8")
    ax.tick_params(colors=INK, labelsize=8.5)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)


def _png(fig):
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    return buf.getvalue()


def _money(x, _pos=None):
    if abs(x) >= 1e6:
        return "%.1fM" % (x / 1e6)
    if abs(x) >= 1e3:
        return "%.0fK" % (x / 1e3)
    return "%.0f" % x


def _fmt_axis(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(FuncFormatter(_money))


# ----------------------------------------------------------------------------
# EDA charts
# ----------------------------------------------------------------------------
def target_distribution(df, target):
    fig = _fig(8.6, 3.9)
    y = df[target].dropna()
    ax1, ax2 = fig.subplots(1, 2)
    ax1.hist(y, bins=40, color=PRIMARY, alpha=0.85, edgecolor="white", zorder=3)
    ax1.axvline(y.mean(), color=RED, linestyle="--", linewidth=1.4, label="Mean %s" % _money(y.mean()))
    ax1.axvline(y.median(), color=GREEN, linestyle="--", linewidth=1.4, label="Median %s" % _money(y.median()))
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1, "%s distribution" % target, target, "Houses")
    _fmt_axis(ax1)
    ax2.hist(np.log1p(y), bins=40, color=ACCENT, alpha=0.85, edgecolor="white", zorder=3)
    _style(ax2, "After log transform (used for training)", "log(1 + %s)" % target, "Houses")
    return _png(fig)


def missing_values(missing_rows):
    """missing_rows: list of dicts with column / before_pct (columns that had missing values)."""
    rows = missing_rows[:15]
    fig = _fig(8.2, max(2.6, 0.34 * len(rows) + 1.3))
    ax = fig.subplots()
    if not rows:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=13, color=GREEN)
        ax.axis("off")
        return _png(fig)
    names = [r["column"] for r in rows][::-1]
    vals = [r["before_pct"] for r in rows][::-1]
    colors = [RED if v > 50 else (ACCENT if v > 10 else PRIMARY) for v in vals]
    ax.barh(names, vals, color=colors, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, "%.1f%%" % v, va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(vals) * 1.15 + 2)
    _style(ax, "Missing values per column (raw file)", "% of rows missing")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def box_plots(df, features):
    features = features[:6]
    fig = _fig(8.8, 5.2)
    axes = np.atleast_1d(fig.subplots(2, 3)).ravel()
    for ax, c in zip(axes, features):
        data = df[c].dropna()
        ax.boxplot(data, patch_artist=True, widths=0.55,
                   boxprops=dict(facecolor="#bfdbfe", color=PRIMARY),
                   medianprops=dict(color=RED, linewidth=1.6),
                   whiskerprops=dict(color=PRIMARY), capprops=dict(color=PRIMARY),
                   flierprops=dict(marker="o", markerfacecolor=ACCENT, markeredgecolor=ACCENT, markersize=3, alpha=0.6))
        _style(ax, c)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
    for ax in axes[len(features):]:
        ax.axis("off")
    return _png(fig)


def scatter_plots(df, features, target):
    features = features[:4]
    fig = _fig(8.8, 6.0)
    axes = np.atleast_1d(fig.subplots(2, 2)).ravel()
    for ax, c in zip(axes, features):
        d = df[[c, target]].dropna()
        ax.scatter(d[c], d[target], s=9, alpha=0.45, color=PRIMARY, zorder=3)
        if d[c].nunique() > 1:
            k, b = np.polyfit(d[c], d[target], 1)
            xs = np.linspace(d[c].min(), d[c].max(), 50)
            ax.plot(xs, k * xs + b, color=RED, linewidth=1.8, zorder=4)
            r = d[c].corr(d[target])
            ax.text(0.03, 0.93, "r = %.2f" % r, transform=ax.transAxes, fontsize=9, color=INK,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cbd5e1"))
        _style(ax, "%s vs %s" % (c, target), c, target)
        _fmt_axis(ax, "y")
    for ax in axes[len(features):]:
        ax.axis("off")
    return _png(fig)


def correlation_heatmap(df, numeric_cols, target, top=12):
    cols = [c for c in numeric_cols if df[c].nunique() > 1]
    corr_t = df[cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
    use = list(corr_t.index[:top]) + [target]
    m = df[use].corr().values
    fig = _fig(8.2, 6.6)
    ax = fig.subplots()
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(use)))
    ax.set_yticks(range(len(use)))
    ax.set_xticklabels(use, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(use, fontsize=8)
    for i in range(len(use)):
        for j in range(len(use)):
            ax.text(j, i, "%.2f" % m[i, j], ha="center", va="center", fontsize=6.8,
                    color="white" if abs(m[i, j]) > 0.6 else INK)
    ax.set_title("Correlation heatmap (top %d features by |correlation| with %s)" % (len(use) - 1, target),
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return _png(fig)


def best_category(df, categorical_cols, target):
    """Categorical column whose groups differ most in average price (eta squared)."""
    best, score = None, -1
    total = ((df[target] - df[target].mean()) ** 2).sum()
    for c in categorical_cols:
        n = df[c].nunique()
        if n < 2 or n > 25:
            continue
        g = df.groupby(c)[target]
        between = (g.count() * (g.mean() - df[target].mean()) ** 2).sum()
        eta = between / total if total else 0
        if eta > score:
            best, score = c, eta
    return best


def category_price(df, col, target):
    if not col:
        return None
    g = df.groupby(col)[target].agg(["mean", "count"]).sort_values("mean")
    g = g[g["count"] >= 5].tail(15)
    fig = _fig(8.2, max(3.0, 0.32 * len(g) + 1.4))
    ax = fig.subplots()
    ax.barh(g.index.astype(str), g["mean"], color=PRIMARY, zorder=3)
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(m * 1.01, i, "%s  (n=%d)" % (_money(m), n), va="center", fontsize=8, color=INK)
    ax.set_xlim(0, g["mean"].max() * 1.22)
    _style(ax, "Average %s by %s" % (target, col), "Average %s" % target)
    _fmt_axis(ax)
    ax.grid(axis="y", visible=False)
    return _png(fig)


# ----------------------------------------------------------------------------
# Model diagnostics
# ----------------------------------------------------------------------------
def feature_importance(importance, top=12):
    items = importance[:top][::-1]
    fig = _fig(8.2, max(3.2, 0.33 * len(items) + 1.4))
    ax = fig.subplots()
    ax.barh([i["feature"] for i in items], [i["importance"] * 100 for i in items], color=ACCENT, zorder=3)
    _style(ax, "Feature importance (Random Forest)", "Importance (%)")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def model_comparison(results):
    fig = _fig(8.2, 3.8)
    ax = fig.subplots()
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["cv_r2"] for r in results], 0.38, label="Cross-validation R\u00b2", color=PRIMARY, zorder=3)
    ax.bar(x + 0.2, [r["r2"] for r in results], 0.38, label="Test-set R\u00b2", color=GREEN, zorder=3)
    for i, r in enumerate(results):
        ax.text(i - 0.2, r["cv_r2"] + 0.01, "%.3f" % r["cv_r2"], ha="center", fontsize=8)
        ax.text(i + 0.2, r["r2"] + 0.01, "%.3f" % r["r2"], ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    _style(ax, "Model comparison", None, "R\u00b2 score")
    return _png(fig)


def actual_vs_predicted(y_true, y_pred):
    fig = _fig(5.6, 5.2)
    ax = fig.subplots()
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, color=PRIMARY, zorder=3)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=RED, linestyle="--", linewidth=1.6, label="Perfect prediction")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Actual vs predicted (test set)", "Actual price", "Predicted price")
    _fmt_axis(ax, "x")
    _fmt_axis(ax, "y")
    return _png(fig)


def residuals(y_true, y_pred):
    res = np.asarray(y_true, float) - np.asarray(y_pred, float)
    fig = _fig(8.6, 3.9)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.hist(res, bins=35, color=GREEN, alpha=0.85, edgecolor="white", zorder=3)
    ax1.axvline(0, color=RED, linestyle="--")
    _style(ax1, "Residual distribution", "Actual \u2212 predicted", "Houses")
    _fmt_axis(ax1)
    ax2.scatter(y_pred, res, s=10, alpha=0.5, color=PRIMARY, zorder=3)
    ax2.axhline(0, color=RED, linestyle="--")
    _style(ax2, "Residuals vs predicted", "Predicted price", "Residual")
    _fmt_axis(ax2, "x")
    _fmt_axis(ax2, "y")
    return _png(fig)


# ----------------------------------------------------------------------------
# Interactive feature explorer (generated per request)
# ----------------------------------------------------------------------------
def feature_explorer(df, feature, target):
    fig = _fig(9.0, 3.8)
    if is_numeric(df[feature]):
        d = df[[feature, target]].dropna()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.hist(d[feature], bins=30, color=PRIMARY, alpha=0.85, edgecolor="white", zorder=3)
        _style(ax1, "Distribution", feature, "Houses")
        ax2.boxplot(d[feature], patch_artist=True, widths=0.55,
                    boxprops=dict(facecolor="#bfdbfe", color=PRIMARY), medianprops=dict(color=RED, linewidth=1.6),
                    flierprops=dict(marker="o", markerfacecolor=ACCENT, markeredgecolor=ACCENT, markersize=3, alpha=0.6))
        _style(ax2, "Box plot")
        ax2.set_xticks([])
        ax3.scatter(d[feature], d[target], s=8, alpha=0.4, color=PRIMARY, zorder=3)
        if d[feature].nunique() > 1:
            k, b = np.polyfit(d[feature], d[target], 1)
            xs = np.linspace(d[feature].min(), d[feature].max(), 40)
            ax3.plot(xs, k * xs + b, color=RED, linewidth=1.6)
        _style(ax3, "vs %s" % target, feature, target)
        _fmt_axis(ax3, "y")
    else:
        top = df[feature].value_counts().head(10).index
        d = df[df[feature].isin(top)]
        ax1, ax2 = fig.subplots(1, 2)
        vc = d[feature].value_counts().sort_values()
        ax1.barh(vc.index.astype(str), vc.values, color=PRIMARY, zorder=3)
        _style(ax1, "Counts", "Houses")
        ax1.grid(axis="y", visible=False)
        order = d.groupby(feature)[target].median().sort_values().index
        ax2.boxplot([d.loc[d[feature] == k, target] for k in order], vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#bfdbfe", color=PRIMARY), medianprops=dict(color=RED, linewidth=1.4),
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=ACCENT, markeredgecolor=ACCENT, alpha=0.5))
        ax2.set_yticklabels([str(k) for k in order], fontsize=8)
        _style(ax2, "%s by %s" % (target, feature), target)
        _fmt_axis(ax2)
    return _png(fig)


def save(png_bytes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(png_bytes)
