from django.urls import path

from . import views

urlpatterns = [
    path("", views.upload, name="upload"),
    path("summary/", views.summary, name="summary"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("predict/", views.predict, name="predict"),
    path("chart/<slug:name>.png", views.chart_image, name="chart"),
    path("explore.png", views.explorer_image, name="explorer"),
]
