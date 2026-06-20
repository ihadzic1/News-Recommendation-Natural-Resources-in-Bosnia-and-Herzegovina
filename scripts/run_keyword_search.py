"""
Pokreće batch keyword search nad klix.ba artiklima i sprema rezultate u data/.

Output fajlovi:
  data/keyword_recommendations.csv  — sve preporuke (top-k po keywordu)
  data/keyword_top1.csv             — samo rank-1 po keywordu
  data/content_gaps.csv             — keywordovi s lošim pokrivanjem (score < prag)

Upotreba:
  python scripts/run_keyword_search.py
  python scripts/run_keyword_search.py --k 10 --gap-threshold 0.05
"""

import argparse
import sys
import time
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.service import MLService

DATA_PATH     = ROOT / "data" / "v2-klix_df.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_PATH    = ARTIFACTS_DIR / "pipeline.joblib"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k",             type=int,   default=5,    help="Top-k preporuka po keywordu")
    p.add_argument("--max-articles",  type=int,   default=4000, help="Broj artikala za učitavanje")
    p.add_argument("--gap-threshold", type=float, default=0.05, help="Prag ispod kojeg je content gap")
    p.add_argument("--no-cache",      action="store_true",       help="Ignoriši cache i rebuildi pipeline")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Keyword Recommender — Batch Search")
    print("=" * 60)

    # Učitaj ili rebuildi pipeline
    if CACHE_PATH.exists() and not args.no_cache:
        print(f"Učitavam pipeline iz cache: {CACHE_PATH}")
        svc = MLService.load(CACHE_PATH)
    else:
        print(f"Gradim pipeline (max_articles={args.max_articles})...")
        t0 = time.time()
        svc = MLService().build(csv_path=DATA_PATH, max_articles=args.max_articles)
        svc.save(CACHE_PATH)
        print(f"Pipeline spreman za {time.time() - t0:.1f}s")

    print(f"\nPokrećem batch query (k={args.k}) za "
          f"{len(svc.keyword_recommender.keywords_df)} keywordova...")
    t0 = time.time()
    results = svc.batch_keyword_search(k=args.k)
    print(f"Gotovo za {time.time() - t0:.2f}s | {len(results)} redova")

    # --- Sve preporuke ---
    out_all = ROOT / "data" / "keyword_recommendations.csv"
    results.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"\nSve preporuke  → {out_all}")

    # --- Top-1 po keywordu ---
    top1 = (
        results[results["rank"] == 1]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    out_top1 = ROOT / "data" / "keyword_top1.csv"
    top1.to_csv(out_top1, index=False, encoding="utf-8-sig")
    print(f"Top-1 rezultati → {out_top1}  ({len(top1)} redova)")

    # --- Content gapovi ---
    gaps = top1[top1["score"] < args.gap_threshold].reset_index(drop=True)
    out_gaps = ROOT / "data" / "content_gaps.csv"
    gaps.to_csv(out_gaps, index=False, encoding="utf-8-sig")
    print(f"Content gapovi  → {out_gaps}  ({len(gaps)} keywordova ispod {args.gap_threshold})")

    # --- Kratak summary ---
    print("\n--- Prosječni score po seed keywordu ---")
    avg = (
        top1.groupby("seed_keyword")["score"]
        .mean()
        .sort_values(ascending=False)
        .round(4)
    )
    print(avg.to_string())

    if len(gaps):
        print(f"\n--- Content gapovi (score < {args.gap_threshold}) ---")
        print(gaps[["suggested_keyword", "country_sim", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
