import pandas as pd

KEEP_15 = [
    "agriculture", "base_metals", "bauxite", "coal", "forest_wood", "gas",
    "hydropower", "iron_steel", "mining", "oil", "renewables", "salt",
    "solar", "water_res", "wind",
]

NR_MERGE = {
    "heat": "coal",
    "heat_energy": "coal",
    "heat_power": "coal",
    "termoenergetika": "coal",
    "wood": "forest_wood",
    "aluminum": "bauxite",
    "lithium": "mining",
}

klix = pd.read_csv("data/klix_articles.csv")
before = len(klix)
klix = klix[klix["article_class_name"].isin(KEEP_15)].reset_index(drop=True)
klix.to_csv("data/klix_articles.csv", index=False)
print(f"klix_articles.csv: {before} -> {len(klix)} redova")
print(f"  klasa: {klix['article_class_name'].nunique()}")
print(f"  klase: {sorted(klix['article_class_name'].unique())}")

nr = pd.read_csv("data/nr_articles.csv")
nr["category"] = nr["category"].replace(NR_MERGE)
nr.to_csv("data/nr_articles.csv", index=False)
kept = sorted(c for c in nr["category"].dropna().unique())
print(f"\nnr_articles.csv: {len(nr)} redova (spojene pobrkane labele)")
print(f"  kategorije sada: {kept}")
