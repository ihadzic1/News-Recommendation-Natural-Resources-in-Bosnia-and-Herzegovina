import os, sys, django
os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings'
sys.path.insert(0, '.')
django.setup()

from ml.recommender.nr_recommender import NRRecommender, CACHE_PATH

print("Buildajem iz novog nr_articles.csv...")
rec = NRRecommender().build()
rec.save()
print(f"Gotovo! Sacuvan: {CACHE_PATH}")
print(f"Broj clanaka u modelu: {len(rec.df):,}")
print("Kategorije:")
print(rec.df["category"].value_counts().head(10).to_string())
