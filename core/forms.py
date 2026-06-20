"""Forms for the recommender, keyword discovery and classification pages."""

from django import forms

from core.models import Article, ResourceKeyword

K_CHOICES = [(3, "k = 3"), (5, "k = 5"), (10, "k = 10")]
METHOD_CHOICES = [
    ("cosine", "Kosinusna sličnost"),
    ("euclidean", "Euklidska udaljenost"),
    ("jaccard", "Jaccard sličnost"),
]


class RecommendForm(forms.Form):
    article = forms.ModelChoiceField(
        queryset=Article.objects.filter(corpus_index__lt=500).order_by("corpus_index"),
        label="Članak",
        widget=forms.Select(attrs={"class": "control"}),
    )
    method = forms.ChoiceField(choices=METHOD_CHOICES, label="Metoda",
                               widget=forms.Select(attrs={"class": "control"}))
    k = forms.TypedChoiceField(choices=K_CHOICES, coerce=int, initial=5, label="Broj preporuka",
                               widget=forms.Select(attrs={"class": "control"}))


class KeywordForm(forms.Form):
    region = forms.ChoiceField(choices=ResourceKeyword.REGIONS, label="Regija",
                               widget=forms.Select(attrs={"class": "control"}))
    keyword = forms.ModelChoiceField(
        queryset=ResourceKeyword.objects.all(), label="Ključna riječ",
        widget=forms.Select(attrs={"class": "control"}),
    )
    k = forms.TypedChoiceField(choices=K_CHOICES, coerce=int, initial=10, label="Broj rezultata",
                               widget=forms.Select(attrs={"class": "control"}))
