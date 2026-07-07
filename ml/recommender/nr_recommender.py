"""
NR (Natural Resources) Recommender — in-memory, session-based personalization
with diversity-aware re-ranking.

Cold start  : najnoviji clanci, diversifikovani po temi
Personalized: prosjecni TF-IDF profil kliknutih clanaka, pa diversifikovani feed
Diversity   : per-topic cap + kumulativna similarity kazna + recency bonus,
              da jedna tema (gorivo, rudari...) ne preplavi feed
"""

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from ml.preprocessing.cleaner import TextCleaner
from ml.preprocessing.features import TfidfBuilder

_ROOT      = Path(__file__).resolve().parent.parent.parent
NR_CSV     = _ROOT / "data" / "nr_articles.csv"
CACHE_PATH = _ROOT / "artifacts" / "nr_pipeline.joblib"

_lock     = threading.Lock()
_instance = None


class NRRecommender:
    def __init__(self):
        self.df      = None
        self.matrix  = None
        self.recency = None
        self.cleaner = TextCleaner()
        self.tfidf   = TfidfBuilder(max_features=8000, ngram_range=(1, 2), min_df=2)

    def build(self):
        if not NR_CSV.exists():
            raise FileNotFoundError(
                f"{NR_CSV} nije pronaden. Izvezi nr_articles.csv iz Colaba (df_final)."
            )
        df = pd.read_csv(NR_CSV, parse_dates=["date"], low_memory=False)
        df["text"]      = df["text"].fillna("").astype(str)
        df["title"]     = df["title"].fillna("").astype(str)
        df["category"]  = df["category"].fillna("none").astype(str)
        df["link"]      = df["link"].fillna("").astype(str) if "link" in df.columns else ""
        df["preview"]   = df["text"].str[:180]
        df["image_url"] = df["image_url"].fillna("").astype(str) if "image_url" in df.columns else ""
        self.df = df.reset_index(drop=True)

        processed = self.df["title"].map(self.cleaner.preprocess) + " " + \
                    self.df["title"].map(self.cleaner.preprocess) + " " + \
                    self.df["text"].str[:500].map(self.cleaner.preprocess)
        raw = self.tfidf.fit_transform(processed)
        self.matrix = normalize(raw, norm="l2", copy=False)

        days = (self.df["date"] - self.df["date"].min()).dt.days
        denom = days.max() if days.max() and days.max() > 0 else 1
        self.recency = (days / denom).fillna(0.0).values
        return self

    def save(self, path=CACHE_PATH):
        path.parent.mkdir(exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=CACHE_PATH):
        return joblib.load(path)

    def _diversify(self, cand, base, k=20, lam=0.5, tau=0.25, cap=3, beta=0.10):
        cand = list(cand)
        if not cand:
            return []
        base = np.asarray(base, dtype=float) + beta * self.recency[cand]
        S = cosine_similarity(self.matrix[cand])
        cats = self.df["category"].values
        n = len(cand)
        pen = np.zeros(n)
        avail = list(range(n))
        chosen, tcount = [], {}
        while len(chosen) < k and avail:
            a = np.array(avail)
            adj = base[a] - lam * pen[a]
            blocked = np.array([tcount.get(cats[cand[i]], 0) >= cap for i in a])
            adj = np.where(blocked, -1e9, adj)
            if adj.max() <= -1e8:
                break
            b = int(a[int(np.argmax(adj))])
            ci = cand[b]
            chosen.append(ci)
            tcount[cats[ci]] = tcount.get(cats[ci], 0) + 1
            avail.remove(b)
            pen += S[b]
        return chosen

    def get_newest(self, k=20, cap=3, pool=120):
        order = self.df.sort_values("date", ascending=False, na_position="last").index.tolist()
        cand = order[:pool]
        chosen = self._diversify(cand, self.recency[cand], k=k, cap=cap)
        return self._to_records(self.df.loc[chosen], indices=chosen)

    def recommend(self, clicked_indices, k=20, cap=3, pool=200):
        valid = [i for i in clicked_indices if 0 <= i < self.matrix.shape[0]]
        if not valid:
            return self.get_newest(k)
        profile = np.asarray(self.matrix[valid].mean(axis=0))
        scores = (self.matrix @ profile.T).ravel()
        scores[valid] = -1.0
        cand = [int(i) for i in np.argsort(scores)[::-1][:pool] if scores[i] > 0]
        if not cand:
            return self.get_newest(k)
        chosen = self._diversify(cand, scores[cand], k=k, cap=cap)
        return self._to_records(self.df.loc[chosen], indices=chosen)

    def article(self, idx):
        if idx < 0 or idx >= len(self.df):
            return None
        row = self.df.iloc[idx]
        return {"idx": idx, **self._row_to_dict(row)}

    def _to_records(self, sub, indices=None):
        out = []
        for i, (_, row) in enumerate(sub.iterrows()):
            d = {"idx": int(indices[i]) if indices is not None else int(row.name)}
            d.update(self._row_to_dict(row))
            out.append(d)
        return out

    @staticmethod
    def _row_to_dict(row):
        date_val = row.get("date", None)
        date_str = date_val.strftime("%d.%m.%Y") if pd.notna(date_val) else ""
        img = str(row.get("image_url", ""))
        return {
            "title":     str(row.get("title", ""))[:120],
            "preview":   str(row.get("preview", ""))[:180],
            "category":  str(row.get("category", "")),
            "date":      date_str,
            "link":      str(row.get("link", "")),
            "image_url": img if img not in ("", "nan") else "",
        }


def get_nr_service():
    global _instance
    with _lock:
        if _instance is None:
            if CACHE_PATH.exists():
                _instance = NRRecommender.load()
            else:
                _instance = NRRecommender().build()
                _instance.save()
    return _instance
