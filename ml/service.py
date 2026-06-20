"""Application-level ML service.

Loads the dataset, runs preprocessing and feature engineering, builds the
TF-IDF representation, the three recommenders, the keyword recommender and
the classification suite. Heavy artifacts are cached to disk with joblib and
the whole service is kept as a process-wide singleton for the Django app.
"""

import threading
from pathlib import Path

import joblib
import random
from collections import Counter

from ml.data.loader import DataLoader
from ml.evaluation.metrics import RecommenderEvaluator
from ml.models.classifiers import ClassifierSuite
from ml.preprocessing.cleaner import TextCleaner
from ml.preprocessing.features import FeatureEngineer, TfidfBuilder
from ml.recommender.keywords import KeywordRecommender
from ml.recommender.similarity import build_recommenders

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "klix_articles.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CACHE_PATH = ARTIFACTS_DIR / "pipeline.joblib"

_lock = threading.Lock()
_instance = None


class MLService:
    """Holds the fitted pipeline and exposes high-level query methods."""

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.cleaner = TextCleaner()
        self.tfidf = TfidfBuilder(max_features=max_features, ngram_range=ngram_range)
        self.df = None
        self.token_sets = None
        self.recommenders = None
        self.keyword_recommender = None
        self.outlier_info = None
        self.classification = None
        self.evaluation = None

    def build(self, csv_path=DATA_PATH, max_articles=4000):
        """Run the full data preparation and modelling pipeline."""
        df = DataLoader(csv_path).load()
        if len(df) > max_articles:
            df = df.sample(n=max_articles, random_state=42).reset_index(drop=True)
        counts = df["article_class_name"].value_counts()
        df = df[df["article_class_name"].isin(counts[counts >= 2].index)].reset_index(drop=True)
        df = FeatureEngineer.add_features(df)
        df, self.outlier_info = DataLoader.clip_outliers(df)
        df["processed_text"] = df["text"].map(self.cleaner.preprocess)
        self.df = df
        self.token_sets = [self.cleaner.token_set(t) for t in df["text"]]
        self.tfidf.fit_transform(df["processed_text"])
        self._wire()
        self._evaluate()
        return self

    def _wire(self):
        self.recommenders = build_recommenders(self.tfidf.matrix, self.token_sets)
        self.keyword_recommender = KeywordRecommender(self.tfidf, self.cleaner)

    def _evaluate(self, sample_size=4000):
        labels_all = self.df["article_class_name"].tolist()
        n = self.tfidf.matrix.shape[0]
        idx = sorted(random.Random(42).sample(range(n), sample_size)) if n > sample_size else list(range(n))
        counts = Counter(labels_all[i] for i in idx)
        idx = [i for i in idx if counts[labels_all[i]] >= 2]
        sub_matrix = self.tfidf.matrix[idx]
        sub_labels = [labels_all[i] for i in idx]
        suite = ClassifierSuite()
        suite.run(sub_matrix, sub_labels)
        self.classification = {
            "results": suite.results,
            "table": suite.comparison_table(),
            "labels": suite.labels,
            "params": {"max_depth": suite.max_depth, "n_estimators": suite.n_estimators},
        }
        evaluator = RecommenderEvaluator(self.recommenders, labels_all)
        self.evaluation = evaluator.category_match_rate()

    def recommend(self, article_index, method="cosine", k=5):
        """Return recommendation rows for the UI."""
        rec = self.recommenders[method]
        rows = []
        for idx, score in rec.recommend(article_index, k=k):
            article = self.df.iloc[idx]
            rows.append({
                "index": idx,
                "title": article["title"],
                "category": article["article_class_name"],
                "score": round(score, 4),
                "preview": str(article["text"])[:220] + "…",
                "link": article.get("link", ""),
            })
        return rows

    def keyword_search(self, phrase, expansion="", k=10):
        """Return keyword discovery rows for the UI."""
        rows = []
        for idx, score in self.keyword_recommender.query(phrase, expansion, k=k):
            article = self.df.iloc[idx]
            rows.append({
                "index": idx,
                "title": article["title"],
                "category": article["article_class_name"],
                "score": round(score, 4),
                "preview": str(article["text"])[:220] + "…",
            })
        return rows

    def batch_keyword_search(self, k=5):
        """Run batch query for all scraped keywords and return enriched DataFrame."""
        results = self.keyword_recommender.batch_query(k=k)
        titles      = self.df["title"].tolist()
        categories  = self.df["article_class_name"].tolist()
        links       = self.df["link"].tolist() if "link" in self.df.columns else [""] * len(self.df)
        results["title"]    = results["article_idx"].map(lambda i: titles[i])
        results["category"] = results["article_idx"].map(lambda i: categories[i])
        results["link"]     = results["article_idx"].map(lambda i: links[i])
        return results

    def save(self, path=CACHE_PATH):
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=CACHE_PATH):
        service = joblib.load(path)
        service._wire()
        return service


def get_service():
    """Return the process-wide MLService singleton, building it on first use."""
    global _instance
    with _lock:
        if _instance is None:
            if CACHE_PATH.exists():
                _instance = MLService.load()
            else:
                _instance = MLService().build()
                _instance.save()
        return _instance
