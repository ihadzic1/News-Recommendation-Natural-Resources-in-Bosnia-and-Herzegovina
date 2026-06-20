"""Content-based recommenders built on TF-IDF vectors and token sets."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class BaseRecommender:
    """Common k-NN retrieval interface for all similarity methods."""

    name = "base"

    def scores_for(self, index):
        """Return an array of similarity scores between article `index`
        and every article in the corpus (higher = more similar)."""
        raise NotImplementedError

    def recommend(self, index, k=5):
        """Return the top-k most similar article indices with their scores,
        excluding the query article itself."""
        scores = self.scores_for(index)
        order = np.argsort(scores)[::-1]
        order = order[order != index][:k]
        return [(int(i), float(scores[i])) for i in order]


class CosineRecommender(BaseRecommender):
    """Cosine similarity over TF-IDF vectors."""

    name = "cosine"

    def __init__(self, tfidf_matrix):
        self.matrix = tfidf_matrix

    def scores_for(self, index):
        return cosine_similarity(self.matrix[index], self.matrix).ravel()


class EuclideanRecommender(BaseRecommender):
    """Euclidean distance over TF-IDF vectors, converted to a similarity
    score via 1 / (1 + distance) so that higher means more similar."""

    name = "euclidean"

    def __init__(self, tfidf_matrix):
        self.matrix = tfidf_matrix

    def scores_for(self, index):
        distances = euclidean_distances(self.matrix[index], self.matrix).ravel()
        return 1.0 / (1.0 + distances)


class JaccardRecommender(BaseRecommender):
    """Jaccard similarity over preprocessed token sets."""

    name = "jaccard"

    def __init__(self, token_sets):
        self.token_sets = token_sets

    def scores_for(self, index):
        query = self.token_sets[index]
        scores = np.zeros(len(self.token_sets))
        if not query:
            return scores
        for i, other in enumerate(self.token_sets):
            union = len(query | other)
            scores[i] = len(query & other) / union if union else 0.0
        return scores


def build_recommenders(tfidf_matrix, token_sets):
    """Construct all three recommenders keyed by method name."""
    return {
        "cosine": CosineRecommender(tfidf_matrix),
        "euclidean": EuclideanRecommender(tfidf_matrix),
        "jaccard": JaccardRecommender(token_sets),
    }
