# Bosnian News Article Recommender System

Content-based preporučivač bosanskih vijesti sa portala **klix.ba**, izgrađen po **CRISP-DM** metodologiji. Kombinuje koncepte predmeta Data Analytics (CDA) i Machine Learning and Pattern Recognition (MLPR).

## Funkcionalnosti

- **EDA Dashboard** — distribucija kategorija, dužine članaka, angažman (komentari/dijeljenja), nedostajuće vrijednosti, IQR detekcija outliera, heatmapa sličnosti (Plotly + Matplotlib + Seaborn).
- **Preporučivač** — k-NN dohvat (k ∈ {3, 5, 10}) sa tri metrike: kosinusna sličnost i euklidska udaljenost nad TF-IDF vektorima te Jaccard sličnost nad skupovima tokena.
- **Klasifikacija** — predikcija `article_class_name` iz teksta: Multinomial Naive Bayes, Decision Tree, Random Forest (podesivi `max_depth`, `n_estimators`); accuracy/precision/recall/F1 + matrice konfuzije.
- **Evaluacija** — Category Match Rate i prosječni similarity score po metodi i k.
- **Prirodni resursi (dodatni modul)** — repozitorij ključnih riječi (SAD / Zapadna Evropa) u bazi; TF-IDF upit vraća najrelevantnije bh. članke sa relevance scoreom.
- **Colab notebook** — kompletan CRISP-DM tok u `notebooks/klix_crisp_dm.ipynb`.

## Struktura projekta

```
config/                  Django projekat (settings, urls)
core/                    Django aplikacija (modeli, pogledi, forme, šabloni, komande)
ml/
  data/                  učitavanje, nedostajuće vrijednosti, IQR outlieri
  preprocessing/         čišćenje teksta, bosanske stop riječi, TF-IDF, feature engineering
  recommender/           metrike sličnosti, k-NN, keyword discovery
  models/                klasifikatori
  evaluation/            Category Match Rate
  dashboard/             Plotly/Matplotlib/Seaborn grafikoni
  service.py             orkestracija pipelinea + keširanje (joblib)
data/                    dataset (klix_articles.csv)
notebooks/               CRISP-DM Jupyter/Colab notebook
scripts/                 generator sample podataka i notebooka
artifacts/               keširani ML pipeline (generiše se automatski)
```

## Pokretanje

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Dataset
#    Preuzmite Kaggle dataset i snimite CSV kao data/klix_articles.csv:
#    https://www.kaggle.com/datasets/salihseferovic/bosnian-news-articles-dataset-from-klixba
#    (ili zadržite priloženi sintetički sample za demo)

# 2) Baza + import + ML artifacts
python manage.py migrate
python manage.py load_data            # opcionalno: --csv putanja --max-articles 4000

# 3) Server
python manage.py runserver
```

Aplikacija je na http://127.0.0.1:8000/

> Nakon zamjene dataseta ponovo pokrenite `python manage.py load_data` da se TF-IDF, preporučivači i klasifikatori ponovo izgrade (keš je u `artifacts/pipeline.joblib`).

## Konfiguracija

- TF-IDF: `max_features`, `ngram_range` u `MLService.__init__` (`ml/service.py`).
- Klasifikatori: `max_depth`, `n_estimators` u `ClassifierSuite` (`ml/models/classifiers.py`).
- Ključne riječi: Django admin (`/admin`, model `ResourceKeyword`) ili `ml/recommender/keywords.py`.

## Notebook

`notebooks/klix_crisp_dm.ipynb` je samostalan i reproducibilan: u Colabu automatski preuzima dataset preko `kagglehub`, a lokalno koristi `klix_articles.csv` iz radnog direktorija. Pokriva svih 8 sekcija: učitavanje, EDA, pripremu, analizu sličnosti, evaluaciju, klasifikaciju, keyword recommender i završnu diskusiju.

## CRISP-DM mapiranje

| Faza | Implementacija |
|---|---|
| Business Understanding | Početna stranica: cilj, kriteriji uspjeha, metrike kvaliteta |
| Data Understanding | EDA Dashboard + notebook sekcija 2 |
| Data Preparation | `ml/preprocessing` + notebook sekcija 3 |
| Modelling | `ml/recommender`, `ml/models` + notebook sekcije 4 i 6 |
| Evaluation | `ml/evaluation` + stranica Klasifikacija + notebook sekcija 5 |
| Deployment | Django aplikacija |
