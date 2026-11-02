import logging
import shutil

import joblib
from django.conf import settings
from django.http import Http404, HttpResponse
from django.shortcuts import redirect, render

from . import runs
from .forms import PredictionForm, UploadForm
from .ml import charts, pipeline as pl, service

logger = logging.getLogger(__name__)

SAMPLE_PATH = settings.BASE_DIR / "sample_data" / "house_prices_sample.csv"
EDA_CHARTS = {"target_dist", "missing", "boxplots", "scatter", "heatmap", "category"}


def _current_run(request):
    """(folder, results) of the analysis stored in the session, or (None, None)."""
    path = runs.run_path(request.session.get("run_id"))
    if path is None:
        return None, None
    return path, runs.load_results(path)


# ---------------------------------------------------------------------------
# 1. Upload
# ---------------------------------------------------------------------------
def upload(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        run_dir = None
        try:
            if form.cleaned_data["use_sample"]:
                name = SAMPLE_PATH.name
                df = pl.load_dataframe(SAMPLE_PATH, name)
            else:
                uploaded = form.cleaned_data["file"]
                name = uploaded.name
                df = pl.load_dataframe(uploaded, name)
            runs.cleanup()
            run_id, run_dir = runs.new_run()
            service.process_dataset(df, name, str(run_dir), form.cleaned_data["target"])
        except ValueError as exc:
            _discard(run_dir)
            form.add_error(None, str(exc))
        except Exception as exc:                      # unexpected problem in the uploaded data
            logger.exception("Processing failed")
            _discard(run_dir)
            form.add_error(None, "The file could not be processed (%s). Check that it is a house-price "
                                 "dataset with a numeric price column." % exc)
        else:
            request.session["run_id"] = run_id
            return redirect("summary")
    path, _ = _current_run(request)
    return render(request, "predictor/upload.html", {"form": form, "has_run": path is not None, "step": "upload"})


def _discard(run_dir):
    if run_dir is not None:
        shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Data summary
# ---------------------------------------------------------------------------
def summary(request):
    path, results = _current_run(request)
    if results is None:
        return redirect("upload")
    return render(request, "predictor/summary.html", {"r": results, "has_run": True, "step": "summary"})


# ---------------------------------------------------------------------------
# 3. Visual analytics dashboard
# ---------------------------------------------------------------------------
def dashboard(request):
    path, results = _current_run(request)
    if results is None:
        return redirect("upload")
    columns = results["explorer_columns"]
    feature = request.GET.get("feature")
    if feature not in columns:
        feature = (results["kept_numeric"] or columns)[0]
    ctx = {
        "r": results, "has_run": True, "step": "dashboard", "columns": columns, "feature": feature,
        "eda": [c for c in results["charts"] if c["name"] in EDA_CHARTS],
        "diagnostics": [c for c in results["charts"] if c["name"] not in EDA_CHARTS],
    }
    return render(request, "predictor/dashboard.html", ctx)


def chart_image(request, name):
    path, results = _current_run(request)
    if results is None or name not in {c["name"] for c in results["charts"]}:
        raise Http404("Chart not found")
    with open(path / "charts" / (name + ".png"), "rb") as fh:
        return HttpResponse(fh.read(), content_type="image/png")


def explorer_image(request):
    """Chart for one column chosen in the dashboard (generated on demand)."""
    path, results = _current_run(request)
    feature = request.GET.get("feature")
    if results is None or feature not in results["explorer_columns"]:
        raise Http404("Unknown column")
    data = joblib.load(path / "clean.joblib")
    png = charts.feature_explorer(data["clean"], feature, data["target"])
    return HttpResponse(png, content_type="image/png")


# ---------------------------------------------------------------------------
# 4. Model results + live prediction
# ---------------------------------------------------------------------------
def predict(request):
    path, results = _current_run(request)
    if results is None:
        return redirect("upload")
    bundle = joblib.load(path / "bundle.joblib")
    form = PredictionForm(bundle["form_fields"], request.POST or None)
    prediction = None
    if request.method == "POST" and form.is_valid():
        values = form.column_values()
        price = max(pl.predict_one(bundle["model"], bundle["columns"], bundle["defaults"], values), 0.0)
        rmse = bundle["rmse"]
        outside = [f["name"] for f in bundle["form_fields"]
                   if f["kind"] == "number" and not (f["min"] <= values[f["name"]] <= f["max"])]
        prediction = {"price": price, "low": max(price - rmse, 0.0), "high": price + rmse, "outside": outside}
    return render(request, "predictor/predict.html", {
        "r": results, "form": form, "prediction": prediction, "has_run": True, "step": "predict",
        "chart_names": {c["name"] for c in results["charts"]},
    })
