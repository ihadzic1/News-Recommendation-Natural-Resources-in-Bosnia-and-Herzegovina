"""Keyword-based discovery: maps natural-resource search interests from
Western Europe and the United States onto Bosnian news articles."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
KEYWORDS_CSV = _DATA_DIR / "keywords_translated.csv"


def _load_keywords_df():
    if KEYWORDS_CSV.exists():
        return pd.read_csv(KEYWORDS_CSV)
    raise FileNotFoundError(
        f"Keyword fajl nije pronađen: {KEYWORDS_CSV}\n"
        "Pokreni scripts/translate_keywords.py da ga generišeš."
    )


class KeywordRecommender:
    """Retrieves articles most relevant to a natural-resource keyword by
    projecting the keyword phrase (plus its Bosnian expansion terms) into
    the corpus TF-IDF space and ranking articles by cosine similarity.

    Single query: query()      — koristi se iz Django UI-a
    Batch query:  batch_query() — jedan matrix multiply za sve keywordove odjednom
    """

    def __init__(self, tfidf_builder, cleaner):
        self.tfidf_builder = tfidf_builder
        self.cleaner = cleaner
        self._kw_df = _load_keywords_df()

    @property
    def keywords_df(self):
        return self._kw_df.copy()

    def query(self, keyword_phrase, expansion_terms="", k=10):
        """Return top-k (article_index, relevance_score) pairs for a keyword."""
        text = f"{keyword_phrase} {expansion_terms}".strip()
        processed = self.cleaner.preprocess(text) or text.lower()
        vector = self.tfidf_builder.transform([processed])
        scores = cosine_similarity(vector, self.tfidf_builder.matrix).ravel()
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def batch_query(self, k=5):
        """Process all keywords in one optimized batch.

        Umjesto 164 odvojenih cosine_similarity poziva, vektorizujemo sve
        query stringove odjednom i radimo jedan matrix multiply:
            (n_kw × vocab) · (vocab × n_articles)ᵀ → (n_kw × n_articles)

        Returns DataFrame sa kolonama:
            seed_keyword, suggested_keyword, source, country_sim,
            rank, article_idx, score
        """
        df = self._kw_df.copy()

        # Preprocess svih query stringova odjednom
        queries = [
            self.cleaner.preprocess(str(q)) or str(q).lower()
            for q in df["query_bs"]
        ]

        # Vektorizuj sve odjednom → sparse (n_kw × vocab)
        query_matrix = self.tfidf_builder.transform(queries)

        # L2-normalizacija (ekvivalentno cosine similarity)
        query_norm = normalize(query_matrix, norm="l2", copy=True)
        corpus_norm = normalize(self.tfidf_builder.matrix, norm="l2", copy=True)

        # Jedan matrix multiply → dense (n_kw × n_articles)
        scores_matrix = (query_norm @ corpus_norm.T).toarray()

        rows = []
        for i, row in df.iterrows():
            scores = scores_matrix[i]
            top_k = np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)):]
            top_k = top_k[np.argsort(scores[top_k])[::-1]]
            top_k = [j for j in top_k if scores[j] > 0]
            for rank, j in enumerate(top_k):
                rows.append({
                    "seed_keyword":      row["seed_keyword"],
                    "suggested_keyword": row["suggested_keyword"],
                    "source":            row["source"],
                    "country_sim":       row["country_sim"],
                    "rank":              rank + 1,
                    "article_idx":       int(j),
                    "score":             round(float(scores[j]), 4),
                })

        return pd.DataFrame(rows)
