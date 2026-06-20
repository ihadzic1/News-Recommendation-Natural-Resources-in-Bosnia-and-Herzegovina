"""Feature engineering and TF-IDF vectorization for article text."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineer:
    """Derives numerical features from raw article data."""

    @staticmethod
    def add_features(df):
        """Add article_length, word_count and comment_share_ratio columns."""
        df = df.copy()
        df["article_length"] = df["text"].fillna("").str.len()
        df["word_count"] = df["text"].fillna("").str.split().str.len()
        df["comment_share_ratio"] = df["num_of_comments"] / (df["num_of_shares"] + 1)
        return df


class TfidfBuilder:
    """Builds and holds a configurable TF-IDF representation of a corpus."""

    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None
        self.matrix = None

    def fit_transform(self, documents):
        """Fit the vectorizer on preprocessed documents and return the TF-IDF matrix."""
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        return self.matrix

    def transform(self, documents):
        """Transform new documents (e.g. keyword queries) into TF-IDF space."""
        if self.vectorizer is None:
            raise RuntimeError("TfidfBuilder must be fitted before calling transform().")
        return self.vectorizer.transform(documents)

    def top_terms(self, n=20):
        """Return the n terms with the highest total TF-IDF weight across the corpus."""
        if self.matrix is None:
            raise RuntimeError("TfidfBuilder must be fitted before calling top_terms().")
        weights = np.asarray(self.matrix.sum(axis=0)).ravel()
        terms = np.array(self.vectorizer.get_feature_names_out())
        order = weights.argsort()[::-1][:n]
        return list(zip(terms[order], weights[order]))
