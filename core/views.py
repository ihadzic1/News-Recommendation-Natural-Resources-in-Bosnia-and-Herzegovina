"""Views for the six application pages. All heavy lifting is delegated to
the MLService singleton; views only assemble template context."""

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from core.forms import KeywordForm, RecommendForm
from core.models import ClickEvent, ResourceKeyword
from ml.dashboard import charts
from ml.data.loader import DataLoader
from ml.recommender.nr_recommender import get_nr_service
from ml.service import get_service

SUCCESS_CRITERIA = [
    "Category Match Rate ≥ 60% za top-5 preporuke (sadržajna konzistentnost).",
    "F1 score klasifikacije kategorija ≥ 0.80 na test skupu.",
    "Vrijeme odgovora preporuke ispod jedne sekunde po upitu.",
    "Pretraga po ključnim riječima vraća relevantne članke sa scoreom > 0.",
]

CRISP_PHASES = [
    ("Business Understanding", "Definisanje cilja: preporuka sličnih bh. vijesti i povezivanje međunarodnog interesa za prirodne resurse sa lokalnim sadržajem."),
    ("Data Understanding", "EDA nad klix.ba skupom: kategorije, dužine tekstova, angažman, nedostajuće vrijednosti i outlieri."),
    ("Data Preparation", "Čišćenje teksta, tokenizacija, uklanjanje bosanskih stop riječi, inženjering atributa i TF-IDF vektorizacija."),
    ("Modelling", "Tri metrike sličnosti (kosinusna, euklidska, Jaccard) uz k-NN dohvat te tri klasifikatora kategorija."),
    ("Evaluation", "Category Match Rate i prosječni similarity score za k ∈ {3, 5, 10}; poređenje klasifikatora standardnim metrikama."),
    ("Deployment", "Django web aplikacija sa interaktivnim dashboardom, preporukama i keyword discovery modulom."),
]


def home(request):
    service = get_service()
    context = {
        "active": "home",
        "n_articles": len(service.df),
        "n_categories": service.df["article_class_name"].nunique(),
        "n_features": service.tfidf.matrix.shape[1],
        "best_method": max(service.evaluation,
                           key=lambda m: service.evaluation[m][5]["category_match_rate"]),
        "success_criteria": SUCCESS_CRITERIA,
        "phases": CRISP_PHASES,
    }
    return render(request, "core/home.html", context)


def eda(request):
    service = get_service()
    df = service.df
    report = DataLoader.missing_value_report(df[[
        "title", "text", "article_class_name", "num_of_comments",
        "num_of_shares", "link", "picture_path",
    ]])
    missing = [
        {"column": col, "count": int(row["missing_count"]), "pct": float(row["missing_pct"])}
        for col, row in report.iterrows()
    ]
    context = {
        "active": "eda",
        "chart_categories": charts.category_distribution(df),
        "chart_lengths": charts.article_length_histogram(df),
        "chart_avg_length": charts.avg_length_per_category(df),
        "chart_engagement": charts.engagement_distributions(df),
        "chart_outliers": charts.outlier_boxplots(df),
        "chart_heatmap": charts.similarity_heatmap(service.tfidf.matrix[:300], df["title"].tolist()[:300]),
        "missing": missing,
        "outlier_info": service.outlier_info,
        "stats": df[["article_length", "word_count", "num_of_comments", "num_of_shares"]]
        .describe().round(2).to_html(classes="data-table", border=0),
    }
    return render(request, "core/eda.html", context)


def recommender(request):
    service = get_service()
    form = RecommendForm(request.GET or None)
    results, query = None, None
    if form.is_valid():
        article = form.cleaned_data["article"]
        query = article
        results = service.recommend(
            article.corpus_index,
            method=form.cleaned_data["method"],
            k=form.cleaned_data["k"],
        )
        method_labels = dict(form.fields["method"].choices)
        query_method = method_labels[form.cleaned_data["method"]]
    else:
        query_method = None
    return render(request, "core/recommender.html", {
        "active": "recommender", "form": form, "results": results,
        "query": query, "query_method": query_method,
    })


def keywords(request):
    service = get_service()
    form = KeywordForm(request.GET or None)
    results, keyword = None, None
    if form.is_valid():
        keyword = form.cleaned_data["keyword"]
        results = service.keyword_search(
            keyword.phrase, keyword.expansion_terms, k=form.cleaned_data["k"],
        )
    grouped = {
        label: ResourceKeyword.objects.filter(region=code)
        for code, label in ResourceKeyword.REGIONS
    }
    return render(request, "core/keywords.html", {
        "active": "keywords", "form": form, "results": results,
        "keyword": keyword, "grouped": grouped,
    })


def classification(request):
    service = get_service()
    clf = service.classification
    confusion = [
        {"model": name, "png": charts.confusion_matrix_png(
            data["confusion_matrix"], clf["labels"], f"Matrica konfuzije — {name}")}
        for name, data in clf["results"].items()
    ]
    return render(request, "core/classification.html", {
        "active": "classification",
        "table": clf["table"],
        "params": clf["params"],
        "chart_comparison": charts.classifier_comparison_chart(clf["table"]),
        "confusion": confusion,
        "chart_evaluation": charts.evaluation_chart(service.evaluation),
        "evaluation": service.evaluation,
        "best_method": max(service.evaluation,
                           key=lambda m: service.evaluation[m][5]["category_match_rate"]),
    })


def about(request):
    return render(request, "core/about.html", {
        "active": "about", "phases": CRISP_PHASES, "success_criteria": SUCCESS_CRITERIA,
    })


# ---------------------------------------------------------------------------
# Demo — klix.ba style homepage s personalizacijom
# ---------------------------------------------------------------------------
COLD_START_THRESHOLD = 3


def demo(request):
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    clicked_qs = ClickEvent.objects.filter(session_key=session_key)
    clicked_indices = list(clicked_qs.values_list("article_idx", flat=True))
    click_count = len(clicked_indices)

    nr = get_nr_service()

    if click_count < COLD_START_THRESHOLD:
        articles = nr.get_newest(k=20)
        mode = "cold"
    else:
        articles = nr.recommend(clicked_indices, k=20)
        mode = "personalized"

    hero   = articles[0]   if len(articles) > 0  else None
    large  = articles[1:2] if len(articles) > 1  else []
    medium = articles[2:5] if len(articles) > 2  else []
    small  = articles[5:]  if len(articles) > 5  else []

    return render(request, "core/demo.html", {
        "active":       "demo",
        "hero":         hero,
        "large":        large,
        "medium":       medium,
        "small":        small,
        "mode":         mode,
        "click_count":  click_count,
        "threshold":    COLD_START_THRESHOLD,
    })


@require_POST
def record_click(request, article_idx):
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key
    nr = get_nr_service()
    if nr.article(article_idx) is None:
        return JsonResponse({"ok": False, "error": "not found"}, status=404)
    ClickEvent.objects.get_or_create(session_key=session_key, article_idx=article_idx)
    click_count = ClickEvent.objects.filter(session_key=session_key).count()
    return JsonResponse({"ok": True, "click_count": click_count})


@require_POST
def reset_preferences(request):
    """Briše sve ClickEvent zapise za ovu sesiju."""
    if not request.session.session_key:
        return JsonResponse({"ok": True, "deleted": 0})
    session_key = request.session.session_key
    deleted, _ = ClickEvent.objects.filter(session_key=session_key).delete()
    return JsonResponse({"ok": True, "deleted": deleted})


def demo_feed(request):
    """AJAX endpoint — vraća samo rendered article grid (bez base layouta)."""
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    clicked_indices = list(
        ClickEvent.objects.filter(session_key=session_key).values_list("article_idx", flat=True)
    )
    click_count = len(clicked_indices)
    nr = get_nr_service()

    if click_count < COLD_START_THRESHOLD:
        articles = nr.get_newest(k=20)
        mode = "cold"
    else:
        articles = nr.recommend(clicked_indices, k=20)
        mode = "personalized"

    hero   = articles[0]   if len(articles) > 0 else None
    large  = articles[1:2] if len(articles) > 1 else []
    medium = articles[2:5] if len(articles) > 2 else []
    small  = articles[5:]  if len(articles) > 5 else []

    return render(request, "core/_demo_grid.html", {
        "hero":        hero,
        "large":       large,
        "medium":      medium,
        "small":       small,
        "mode":        mode,
        "click_count": click_count,
        "threshold":   COLD_START_THRESHOLD,
    })
