"""Chart builders for the EDA dashboard, recommender evaluation and
classification pages. Plotly figures are returned as embeddable HTML;
matplotlib/seaborn figures are returned as base64 PNG strings."""

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(13,17,23,0.4)",
    "font": {"family": "Inter, sans-serif", "color": "#c9d4e3"},
    "margin": {"t": 48, "r": 24, "b": 48, "l": 56},
}
ACCENT = "#3ddbc4"
ACCENT2 = "#f0b441"


def _html(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight",
                facecolor="#10151d", edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def category_distribution(df):
    counts = df["article_class_name"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.bar(counts, x="category", y="count", color_discrete_sequence=[ACCENT],
                 title="Distribucija kategorija")
    return _html(fig)


def article_length_histogram(df):
    fig = px.histogram(df, x="article_length", nbins=40, color_discrete_sequence=[ACCENT],
                       title="Histogram dužine članaka (broj znakova)")
    return _html(fig)


def avg_length_per_category(df):
    avg = df.groupby("article_class_name")["article_length"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg, x="article_class_name", y="article_length",
                 color_discrete_sequence=[ACCENT2], title="Prosječna dužina članka po kategoriji")
    return _html(fig)


def engagement_distributions(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["num_of_comments"], name="Komentari",
                               marker_color=ACCENT, opacity=0.75))
    fig.add_trace(go.Histogram(x=df["num_of_shares"], name="Dijeljenja",
                               marker_color=ACCENT2, opacity=0.75))
    fig.update_layout(barmode="overlay", title="Distribucija komentara i dijeljenja")
    return _html(fig)


def outlier_boxplots(df):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["num_of_comments"], name="Komentari (prije)", marker_color="#e0635f"))
    fig.add_trace(go.Box(y=df["num_of_comments_clipped"], name="Komentari (poslije)", marker_color=ACCENT))
    fig.add_trace(go.Box(y=df["num_of_shares"], name="Dijeljenja (prije)", marker_color="#e0635f"))
    fig.add_trace(go.Box(y=df["num_of_shares_clipped"], name="Dijeljenja (poslije)", marker_color=ACCENT))
    fig.update_layout(title="IQR detekcija outliera — prije i poslije obrade")
    return _html(fig)


def evaluation_chart(evaluation):
    methods = list(evaluation.keys())
    k_values = sorted(next(iter(evaluation.values())).keys())
    fig = go.Figure()
    palette = [ACCENT, ACCENT2, "#9b7bf5"]
    for color, method in zip(palette, methods):
        fig.add_trace(go.Bar(
            x=[f"k={k}" for k in k_values],
            y=[evaluation[method][k]["category_match_rate"] for k in k_values],
            name=method, marker_color=color,
        ))
    fig.update_layout(barmode="group", title="Category Match Rate po metodi i k",
                      yaxis_title="Match rate (%)")
    return _html(fig)


def classifier_comparison_chart(table):
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig = go.Figure()
    palette = [ACCENT, ACCENT2, "#9b7bf5"]
    for color, row in zip(palette, table):
        fig.add_trace(go.Bar(x=metrics, y=[row[m] for m in metrics],
                             name=row["model"], marker_color=color))
    fig.update_layout(barmode="group", title="Poređenje klasifikatora", yaxis_range=[0, 1.05])
    return _html(fig)


def confusion_matrix_png(matrix, labels, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="mako",
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, color="#e6edf3")
    ax.set_xlabel("Predviđeno", color="#c9d4e3")
    ax.set_ylabel("Stvarno", color="#c9d4e3")
    ax.tick_params(colors="#c9d4e3", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _png(fig)


def similarity_heatmap(tfidf_matrix, titles, n=18):
    sub = tfidf_matrix[:n]
    sim = cosine_similarity(sub)
    short = [t[:34] + "…" for t in titles[:n]]
    fig = px.imshow(sim, x=short, y=short, color_continuous_scale="Tealgrn",
                    title=f"Heatmapa kosinusne sličnosti (prvih {n} članaka)")
    fig.update_layout(height=620)
    return _html(fig)
