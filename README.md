# Bosnian News Recommender — Natural Resources of BiH

A news recommendation system over the klix.ba dataset, with an additional
**natural-resources trend discovery** module. The system links international
interest in Bosnia's natural resources — expressed through the keywords users in
the US and Western Europe search for on Google — with local Bosnian news
articles, by projecting those keywords into the corpus TF-IDF space.

The project follows the CRISP-DM methodology and was developed for the CDA + MLPR
course.

---

## Table of contents

- [Architecture](#architecture)
- [Application pages](#application-pages)
- [How the keyword module works](#how-the-keyword-module-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Required data files](#required-data-files)
- [Running the project](#running-the-project)
- [Keyword pipeline (scraping → translation → database)](#keyword-pipeline-scraping--translation--database)
- [Helper scripts](#helper-scripts)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)

---

## Architecture

The application is a Django web service. All heavy ML computation lives in the
`ml/` package and is loaded as a **singleton** (`MLService`), cached to disk with
`joblib` under `artifacts/`. Django views only assemble template context.

There are two separate ML services:

1. **`MLService`** (`ml/service.py`) — the main pipeline: loads the klix dataset,
   runs preprocessing, builds TF-IDF, three recommenders
   (cosine / euclidean / Jaccard), the keyword recommender, and the
   classification suite. Powers the Home, EDA, Recommender, Natural Resources and
   Classification pages.

2. **`NRRecommender`** (`ml/recommender/nr_recommender.py`) — a separate,
   session-based feed with diversity-aware re-ranking. Powers the Demo page.

The database is **SQLite** (`db.sqlite3`), with two main models: `Article`
(corpus articles) and `ResourceKeyword` (the natural-resources keyword
repository).

---

## Application pages

| Route | Page | Data source |
|-------|------|-------------|
| `/` | Home | `MLService` (metrics summary) |
| `/eda/` | EDA Dashboard | `MLService` + `ml/dashboard/charts.py` |
| `/recommender/` | Recommender | `MLService.recommend()` |
| `/keywords/` | Natural Resources | `ResourceKeyword` (DB) + `MLService.keyword_search()` |
| `/classification/` | Classification | `MLService` classifiers |
| `/demo/` | Demo | `NRRecommender` (session feed) |
| `/admin/` | Django admin | edit `Article` / `ResourceKeyword` |

---

## How the keyword module works

The **Natural Resources** page (`/keywords/`) is the core of the keyword module
and works as follows:

1. **Repository** — all keywords live in the database as `ResourceKeyword`
   records, with fields `region` (US or WE), `phrase` (English keyword) and
   `expansion_terms` (Bosnian expansion terms appended to the query vector).

2. **Search** — when the user selects a keyword, `KeywordRecommender.query()`
   joins the phrase with its Bosnian expansion terms, projects that text into the
   corpus TF-IDF space, and returns the top-k Bosnian articles ranked by cosine
   similarity.

The Bosnian expansion terms are crucial: the English phrase `bosnia iron ore`
alone matches Bosnian articles poorly, but with the expansion
`željezna ruda željezo čelik zenica arcelormittal rudnik` it matches the actual
Klix content far better. The seed → Bosnian expansion mapping is curated manually
in `scripts/translate_keywords.py` (`SEED_TRANSLATIONS`).

The keyword set was collected by combining Google Autocomplete and Google Related
Searches (via SerpApi) across 7 regions (US + 6 Western European: GB, DE, AT, FR,
IT, NL). Related searches carry the real geographic variety — named entities such
as `Adriatic Metals`, `Rio Tinto`, `ArcelorMittal Zenica`,
`Ržr ljubija ad prijedor` — that autocomplete does not produce. Details in the
[Keyword pipeline](#keyword-pipeline-scraping--translation--database) section.

---

## Prerequisites

- **Python 3.13** (the project was tested on 3.13.x)
- **pip** and **venv**
- Git (optional)

Check your version:

```powershell
py -3.13 --version
```

It must print `Python 3.13.x`. If you don't have it, install it from python.org,
or run `py --list` to see which versions you have.

---

## Installation

### 1. Clone / unpack the project

```powershell
cd C:\path\to\News-Recommendation-Natural-Resources-in-Bosnia-and-Herzegovina
```

### 2. Create a virtual environment

**Important:** create the venv with Python 3.13, and **never copy a venv between
folders** — a Windows venv stores absolute paths inside itself. If you move the
project, delete `venv/` and recreate it.

```powershell
py -3.13 -m venv venv
```

### 3. Activate the venv

```powershell
venv\Scripts\Activate.ps1
```

If PowerShell says *"running scripts is disabled"*, run this once in the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

then activate again.

### 4. Confirm the venv points to the right location

```powershell
python --version
where.exe python
```

`python --version` must say `Python 3.13.x`, and the first line of
`where.exe python` must be a path inside `...\venv\Scripts\`, not some other
folder.

### 5. Install dependencies

Use `python -m pip` (not bare `pip`) to avoid launcher issues:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Note:** `openpyxl` is **required** — `translate_keywords.py` reads the `.xlsx`
> file through it. If it's missing you'll get `ImportError: Import openpyxl failed`.

---

## Required data files

The `data/` folder is **not** part of the repository (the data is large /
generated) and must be created manually. The application expects the following
files in `data/`:

| File | Used by | Purpose |
|------|---------|---------|
| `klix_articles.csv` | `MLService` | main article corpus (recommender, EDA, classification, keyword search) |
| `nr_articles.csv` | `NRRecommender` | articles for the Demo feed (natural resources) |
| `keywords_natural_resources_bih.xlsx` | `translate_keywords.py` | raw keyword set (seed, suggested, source, country_sim) |
| `keywords_translated.csv` | `keywords.py` + `load_data` | generated by `translate_keywords.py`; feeds the DB / dashboard |

Create the folder if it doesn't exist:

```powershell
mkdir data
```

### Columns `klix_articles.csv` must contain

`title`, `text`, `article_class`, `article_class_name`, `num_of_comments`,
`num_of_shares`, `link`, `picture_path`

### Columns `keywords_natural_resources_bih.xlsx` must contain

`seed_keyword`, `suggested_keyword`, `source`, `country_sim`

### Files you can safely delete

- **`keywords_natural_resources_bih_old.xlsx`** — an older keyword set, not
  referenced anywhere in the code. Safe to delete.

Everything else in the table above is actively used — keep it.

> **Naming mismatch to be aware of:** `MLService` (and thus the whole app) reads
> `data/klix_articles.csv`, but the batch script `scripts/run_keyword_search.py`
> reads `data/v2-klix_df.csv`. The app works because it uses `klix_articles.csv`;
> the batch script will fail unless you either rename/copy `klix_articles.csv` to
> `v2-klix_df.csv`, or change line 28 in `run_keyword_search.py` to point at
> `klix_articles.csv`.

---

## Running the project

First-time run order (all commands from the project root, with the venv active):

### 1. Database migrations

```powershell
python manage.py migrate
```

### 2. Populate the database (articles + keywords + ML cache)

This command loads the klix CSV, builds the ML cache, and seeds `ResourceKeyword`
from the translated keyword CSV:

```powershell
python manage.py load_data
```

It prints how many articles and keywords were imported. **Prerequisite:** the
translated CSV must exist (`data/keywords_translated.csv`) — see the next section
if you don't have it yet.

### 3. (Optional) Create an admin account

```powershell
python manage.py createsuperuser
```

### 4. (Optional) Build the NR feed for the Demo page

```powershell
python scripts/rebuild_nr.py
```

### 5. Start the server

```powershell
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

> The first request to the app is slow because `MLService` is built and cached to
> `artifacts/pipeline.joblib`. Subsequent requests are fast (loaded from cache).
> If you change the data, delete `artifacts/*.joblib` so the cache regenerates.

---

## Keyword pipeline (scraping → translation → database)

This is how the keyword set flows from the raw Excel file to what appears on the
Natural Resources page. **Run this every time you change the keyword set.**

### Overview

```
keywords_natural_resources_bih.xlsx
        │  (scripts/translate_keywords.py — adds Bosnian expansions)
        ▼
data/keywords_translated.csv
        │  (python manage.py load_data — seeds the DB)
        ▼
ResourceKeyword table in the database
        │
        ▼
Natural Resources page (/keywords/)
```

### Step 1 — Translate keywords to Bosnian

`translate_keywords.py` reads `data/keywords_natural_resources_bih.xlsx`, maps
each seed to its Bosnian expansion terms (`SEED_TRANSLATIONS`), and writes
`data/keywords_translated.csv` with `encoding="utf-8-sig"`.

```powershell
python scripts/translate_keywords.py
```

Watch the output:
- `Učitano N keywordova` — how many it read from the xlsx. If you expect the new
  expanded set, N should be large (e.g. ~589). If it's ~164, the xlsx in `data/`
  is the old one.
- `UPOZORENJE — seed keywordovi bez prijevoda: {...}` — if this appears, the xlsx
  contains seeds not present in `SEED_TRANSLATIONS`, so their expansion terms will
  be empty (weaker TF-IDF match). Add them to the map and re-run.

### Step 2 — Clear the old table and repopulate the database

**Important:** `load_data` uses `update_or_create` for keywords, which means it
**does not delete old keywords** — it only adds/updates. If you previously had
junk or corrupted (mojibake) keywords, they stay in the database. So clear the
table manually first:

```powershell
python manage.py shell
```

then inside the shell:

```python
from core.models import ResourceKeyword
ResourceKeyword.objects.all().delete()
exit()
```

Only then repopulate:

```powershell
python manage.py load_data
```

### Step 3 — Verify it's clean

```powershell
python manage.py shell
```

```python
from core.models import ResourceKeyword
print("Total:", ResourceKeyword.objects.count())
print("Mojibake:", ResourceKeyword.objects.filter(phrase__contains="Ä").count())
print("Junk test:", ResourceKeyword.objects.filter(phrase="Bosnia girls").count())
exit()
```

Expected: total ~279 records (US + WE), mojibake **0**, junk test **0**.

> **Why does US look balanced while WE is smaller than the sum of 6 countries?**
> The model has `unique_together = ("region", "phrase")`, and all 6 Western
> European countries map to a single `WE` region. Since neighbouring European
> countries share a large portion of the same keywords (e.g. `Rio Tinto` is
> searched in DE, AT, IT...), duplicates collapse into a single `WE` record. This
> is expected behaviour, not a bug.

### Note on encoding (mojibake)

If strings like `PeruÄ‡ica`, `VareÅ¡`, `Å¾eljezara` appear instead of
`Perućica`, `Vareš`, `željezara` — that's UTF-8 text read as Windows-1252. Fix:

1. **At the source:** when saving CSV from Colab, use
   `df.to_csv(path, encoding="utf-8-sig")`. When reading, always use
   `pd.read_csv(path, encoding="utf-8")`.
2. **If already corrupted:** apply a fix function before writing to the DB:
   ```python
   def fix_mojibake(s):
       if not isinstance(s, str):
           return s
       try:
           return s.encode("cp1252").decode("utf-8")
       except (UnicodeEncodeError, UnicodeDecodeError):
           return s
   df["suggested_keyword"] = df["suggested_keyword"].apply(fix_mojibake)
   ```
   Order matters: run `fix_mojibake` first, then the junk filter, then dedup — so
   the fixed and previously-corrupted versions of the same term merge into one row.

---

## Helper scripts

All in `scripts/`, run from the project root.

| Script | What it does | Output |
|--------|--------------|--------|
| `translate_keywords.py` | adds Bosnian expansions to keywords | `data/keywords_translated.csv` |
| `run_keyword_search.py` | batch keyword search over the corpus | `data/keyword_recommendations.csv`, `keyword_top1.csv`, `content_gaps.csv` |
| `rebuild_nr.py` | (re)builds the NR feed model for Demo | `artifacts/nr_pipeline.joblib` |
| `test_demo.py` | tests Demo functionality | — |

Example batch keyword search with parameters:

```powershell
python scripts/run_keyword_search.py --k 10 --gap-threshold 0.05
```

> Note: `run_keyword_search.py` currently reads `data/v2-klix_df.csv` (see the
> naming mismatch note above). Point it at `klix_articles.csv` if that's your
> corpus file.

---

## Troubleshooting

**`ImportError: Import openpyxl failed`**
`openpyxl` is missing from the venv. Fix: `python -m pip install openpyxl`.

**`Fatal error in launcher: Unable to create process ... python.exe`**
The venv was copied from another location and points to a stale path. Fix: delete
`venv/`, create a new one (`py -3.13 -m venv venv`), activate it, and use
`python -m pip` instead of `pip`.

**`Activate.ps1 ... running scripts is disabled`**
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

**The dashboard shows old / junk / corrupted keywords**
The database holds the old import. Follow the
[Keyword pipeline](#keyword-pipeline-scraping--translation--database): clear the
`ResourceKeyword` table, then run `load_data` again. Remember that `load_data`
alone does not delete old keywords.

**I changed the data but nothing changed on the pages**
`MLService` is cached in `artifacts/pipeline.joblib`. Delete `artifacts/*.joblib`
so the cache regenerates on the next run.

**`FileNotFoundError: ... klix_articles.csv / nr_articles.csv not found`**
A data file is missing. See [Required data files](#required-data-files) and put
the appropriate CSV in `data/`.

**`translate_keywords.py` says "Učitano 164 keywordova" but I expect more**
`data/keywords_natural_resources_bih.xlsx` contains the old set. Replace it with
the new (expanded) set under the same filename.

---

## Project structure

```
.
├── config/                 # Django project (settings, urls, wsgi/asgi)
│   ├── settings.py         # SQLite DB, DEBUG from .env, ALLOWED_HOSTS
│   └── urls.py
├── core/                   # Main Django app
│   ├── models.py           # Article, ResourceKeyword, ClickEvent
│   ├── views.py            # 7 pages + demo endpoints
│   ├── forms.py            # RecommendForm, KeywordForm
│   ├── urls.py
│   ├── admin.py
│   ├── management/commands/
│   │   └── load_data.py    # import articles + seed keywords + ML cache
│   ├── migrations/
│   ├── templates/core/     # HTML templates (base, home, eda, keywords, ...)
│   └── static/core/        # CSS
├── ml/                     # ML package (independent of Django)
│   ├── service.py          # MLService singleton (main pipeline)
│   ├── data/loader.py      # loading + cleaning + outliers
│   ├── preprocessing/      # cleaner (Bosnian stopwords), features (TF-IDF)
│   ├── models/classifiers.py
│   ├── recommender/
│   │   ├── similarity.py   # cosine / euclidean / Jaccard recommenders
│   │   ├── keywords.py     # KeywordRecommender (keyword TF-IDF projection)
│   │   └── nr_recommender.py # NRRecommender (Demo feed)
│   ├── evaluation/metrics.py
│   └── dashboard/charts.py # chart generation for EDA/classification
├── scripts/                # helper scripts (see above)
├── notebooks/              # Colab notebooks (recommender, keyword scraper)
├── artifacts/              # cached ML models (.joblib, .pkl, .npz)
├── data/                   # DATA — create manually (CSV/XLSX, see above)
├── db.sqlite3              # SQLite database (after migrate)
├── manage.py
├── requirements.txt
└── README.md
```

---

## Tech stack

Django · pandas · scikit-learn · NumPy · Plotly · matplotlib · seaborn ·
joblib · SQLite · TF-IDF / k-NN / Naive Bayes / Decision Tree / Random Forest

CRISP-DM · CDA + MLPR · klix.ba dataset
