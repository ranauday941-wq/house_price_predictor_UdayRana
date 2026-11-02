from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError


class UploadForm(forms.Form):
    file = forms.FileField(required=False, widget=forms.ClearableFileInput(attrs={"accept": ".csv,.xlsx", "id": "file"}))
    target = forms.CharField(required=False, max_length=100,
                             widget=forms.TextInput(attrs={"placeholder": "Auto-detect (e.g. SalePrice)", "class": "input"}))
    use_sample = forms.BooleanField(required=False)

    def clean_file(self):
        f = self.cleaned_data.get("file")
        if f:
            ext = f.name.lower().rsplit(".", 1)[-1] if "." in f.name else ""
            if ext not in ("csv", "xlsx"):
                raise ValidationError("Only .csv and .xlsx files are supported.")
            if f.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValidationError("The file is larger than %d MB." % settings.MAX_UPLOAD_MB)
        return f

    def clean(self):
        data = super().clean()
        if "file" in self.errors:
            return data
        if not data.get("file") and not data.get("use_sample"):
            raise ValidationError("Choose a .csv or .xlsx file, or click \"Use sample dataset\".")
        return data


class PredictionForm(forms.Form):
    """Form built at run time from the most important columns of the trained model."""

    def __init__(self, fields_meta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = fields_meta
        for i, f in enumerate(fields_meta):
            key = "f%d" % i
            if f["kind"] == "number":
                integer = f.get("integer")
                default = int(round(f["default"])) if integer else round(f["default"], 2)
                hint = "typical range %s to %s" % (
                    format(f["min"], ",.0f" if integer else ",.2f"), format(f["max"], ",.0f" if integer else ",.2f"))
                self.fields[key] = forms.FloatField(
                    label=f["name"], initial=default, help_text=hint,
                    widget=forms.NumberInput(attrs={"step": "1" if integer else "any", "class": "input"}))
            else:
                self.fields[key] = forms.ChoiceField(
                    label=f["name"], initial=f["default"], choices=[(c, c) for c in f["choices"]],
                    widget=forms.Select(attrs={"class": "input"}))

    def column_values(self):
        """Cleaned values keyed by the real column names."""
        return {f["name"]: self.cleaned_data["f%d" % i] for i, f in enumerate(self.meta)}
