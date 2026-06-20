"""URL routes for the core application."""

from django.urls import path

from core import views

urlpatterns = [
    path("", views.home, name="home"),
    path("eda/", views.eda, name="eda"),
    path("recommender/", views.recommender, name="recommender"),
    path("keywords/", views.keywords, name="keywords"),
    path("classification/", views.classification, name="classification"),
    path("about/", views.about, name="about"),
    path("demo/", views.demo, name="demo"),
    path("demo/click/<int:article_idx>/", views.record_click, name="record_click"),
    path("demo/reset/", views.reset_preferences, name="reset_preferences"),
    path("demo/feed/", views.demo_feed, name="demo_feed"),
]
