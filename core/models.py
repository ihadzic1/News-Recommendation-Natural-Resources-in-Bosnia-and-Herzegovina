"""Database models: persisted articles and the natural-resource keyword repository."""

from django.db import models


class Article(models.Model):
    """A klix.ba news article imported from the dataset."""

    title = models.CharField(max_length=300)
    text = models.TextField()
    article_class = models.CharField(max_length=100, blank=True, default="")
    article_class_name = models.CharField(max_length=100)
    num_of_comments = models.IntegerField(default=0)
    num_of_shares = models.IntegerField(default=0)
    link = models.URLField(max_length=500, blank=True)
    picture_path = models.CharField(max_length=500, blank=True)
    corpus_index = models.IntegerField(unique=True)

    class Meta:
        ordering = ["corpus_index"]

    def __str__(self):
        return self.title


class ResourceKeyword(models.Model):
    """A natural-resource search keyword associated with a region of interest."""

    REGIONS = [("US", "United States"), ("WE", "Western Europe")]

    region = models.CharField(max_length=2, choices=REGIONS)
    phrase = models.CharField(max_length=100)
    expansion_terms = models.CharField(
        max_length=300, blank=True,
        help_text="Bosnian expansion terms appended to the query vector.",
    )

    class Meta:
        unique_together = ("region", "phrase")
        ordering = ["region", "phrase"]

    def __str__(self):
        return f"{self.get_region_display()}: {self.phrase}"


class ClickEvent(models.Model):
    """Tracks which NR article a session clicked — used for personalization."""

    session_key = models.CharField(max_length=40, db_index=True)
    article_idx = models.IntegerField()
    clicked_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("session_key", "article_idx")
        ordering = ["-clicked_at"]

    def __str__(self):
        return f"{self.session_key[:8]}… → article #{self.article_idx}"
