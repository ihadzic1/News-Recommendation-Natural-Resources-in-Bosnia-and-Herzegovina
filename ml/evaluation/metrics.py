"""Evaluation of recommendation quality: category match rate and
average similarity score across similarity methods and values of k."""

import random


class RecommenderEvaluator:
    """Computes evaluation metrics for a dict of recommenders over a corpus."""

    def __init__(self, recommenders, categories, sample_size=150, random_state=42):
        self.recommenders = recommenders
        self.categories = list(categories)
        self.sample_size = min(sample_size, len(self.categories))
        self.random_state = random_state

    def _sample_indices(self):
        rng = random.Random(self.random_state)
        return rng.sample(range(len(self.categories)), self.sample_size)

    def category_match_rate(self, k_values=(3, 5, 10)):
        """For each method and k, compute the percentage of top-k
        recommendations sharing the query article's category, plus the
        average similarity score of those recommendations."""
        indices = self._sample_indices()
        results = {}
        for name, rec in self.recommenders.items():
            results[name] = {}
            for k in k_values:
                matches, total, score_sum = 0, 0, 0.0
                for idx in indices:
                    query_cat = self.categories[idx]
                    for rec_idx, score in rec.recommend(idx, k=k):
                        total += 1
                        score_sum += score
                        if self.categories[rec_idx] == query_cat:
                            matches += 1
                results[name][k] = {
                    "category_match_rate": round(100.0 * matches / total, 2) if total else 0.0,
                    "avg_similarity": round(score_sum / total, 4) if total else 0.0,
                }
        return results

    @staticmethod
    def best_method(results, k=5):
        """Return the method name with the highest category match rate at k."""
        return max(results, key=lambda m: results[m][k]["category_match_rate"])
