from django.core.management.base import BaseCommand

from core.models import Article, ResourceKeyword
from ml.recommender.keywords import _load_keywords_df
from ml.service import DATA_PATH, MLService


def to_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


class Command(BaseCommand):
    help = "Ucitaj CSV, seeduj keywordove i izgradi ML cache."

    def add_arguments(self, parser):
        parser.add_argument("--csv", default=str(DATA_PATH))
        parser.add_argument("--max-articles", type=int, default=4000)
        parser.add_argument("--batch", type=int, default=5000)

    def handle(self, *args, **options):
        service = MLService().build(csv_path=options["csv"], max_articles=options["max_articles"])
        service.save()

        Article.objects.all().delete()
        batch = []
        for idx, r in enumerate(service.df.itertuples(index=False)):
            batch.append(Article(
                title=str(r.title)[:300],
                text=r.text,
                article_class=str(getattr(r, "article_class", "") or "")[:100],
                article_class_name=str(getattr(r, "article_class_name", "") or ""),
                num_of_comments=to_int(r.num_of_comments),
                num_of_shares=to_int(r.num_of_shares),
                link=str(getattr(r, "link", "") or ""),
                picture_path=str(getattr(r, "picture_path", "") or ""),
                corpus_index=idx,
            ))
            if len(batch) >= options["batch"]:
                Article.objects.bulk_create(batch)
                batch = []
        if batch:
            Article.objects.bulk_create(batch)

        kw_df = _load_keywords_df()
        for _, row in kw_df.iterrows():
            country = str(row["country_sim"]).upper()
            region = "US" if country == "US" else "WE"
            ResourceKeyword.objects.update_or_create(
                region=region, phrase=str(row["suggested_keyword"])[:100],
                defaults={"expansion_terms": str(row["bosnian_expansion"])[:300]},
            )

        self.stdout.write(self.style.SUCCESS(
            f"Uvezeno {Article.objects.count()} clanaka, "
            f"{ResourceKeyword.objects.count()} keywordova, ML cache izgradjen."
        ))