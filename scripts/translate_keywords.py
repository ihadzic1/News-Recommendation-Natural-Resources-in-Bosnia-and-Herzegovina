"""
Prevodi suggested_keywords iz Excel fajla na bosanski
koristeći ručno kuriranu mapu seed → bosanska ekspanzija.
Output: data/keywords_translated.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEED_TRANSLATIONS = {
    "lithium Bosnia":                  "litij litijum baterije rudnik rudarstvo lopare",
    "Lopare lithium mine":             "litij litijum lopare rudnik rudarstvo eksploatacija",
    "Bosnia mining":                   "rudarstvo rudnik eksploatacija rude minerali kopanje",
    "Bosnia natural resources":        "prirodni resursi sirovine ruda energija šuma voda bogatstvo",
    "bauxite Bosnia":                  "boksit aluminij rudnik rudarstvo eksploatacija",
    "coal mining Bosnia":              "ugalj rudnik ugljenokop termoelektrana koks",
    "hydropower Bosnia":               "hidroelektrana hidroenergija vodna energija brana rijeka",
    "Bosnia forests":                  "šuma šumarstvo drvo sječa pošumljavanje prašuma",
    "Bosnia water resources":          "voda rijeka jezero vodovod resursi izvor",
    "rare earth minerals Balkans":     "rijetki minerali rijetke zemlje sirovine ruda strateški",
    "Bosnia and Herzegovina minerals": "minerali rude sirovine nalazišta geologija zemlja",
    "Bosnia mining investment":        "investicija rudarstvo kapital strani ulagač kompanija",
    "lithium mining Europe":           "litij litijum europa rudarstvo rudnik sirovine",
    "Balkan natural resources":        "balkan prirodni resursi sirovine energija ruda",
    "Bosnia iron ore":                 "željezna ruda željezo čelik zenica arcelormittal rudnik",
    "Zenica steel Bosnia":             "zenica čelik željezara arcelormittal čeličana industrija",
    "Bosnia zinc mining":              "cink olovo rudnik vareš rudarstvo minerali",
    "Bosnia lead mining":              "olovo cink rudnik rudarstvo minerali eksploatacija",
    "Tuzla salt mines":                "sol tuzla rudnik solana soljenje industrija",
    "Bosnia timber industry":          "drvo šuma drvna industrija piljenje izvoz pilana",
    "Bosnia wood export":              "drvo izvoz šuma drvna industrija pilana produkt",
    "Bosnia wind energy":              "vjetar vjetroelektrana obnovljiva energija vjetropark",
    "Bosnia solar energy":             "solarni sunce elektrana paneli obnovljiva energija",
    "Bosnia natural gas":              "plin gas gasovod energija prirodni dovod",
    "Bosnia oil reserves":             "nafta naftno polje bušotina rezerve derivati",
    "Bosnia copper mining":            "bakar rudnik rudarstvo ruda minerali eksploatacija",
    "Bosnia manganese":                "mangan rudnik ruda minerali eksploatacija čelik",
    "Bosnia agriculture land":         "poljoprivreda zemlja obradivo tlo farma kultura",
}


def main():
    input_path  = ROOT / "data" / "keywords_natural_resources_bih.xlsx"
    output_path = ROOT / "data" / "keywords_translated.csv"

    df = pd.read_excel(input_path)
    print(f"Učitano {len(df)} keywordova.")

    missing = set(df["seed_keyword"].unique()) - set(SEED_TRANSLATIONS)
    if missing:
        print(f"UPOZORENJE — seed keywordovi bez prijevoda: {missing}")

    df["bosnian_expansion"] = df["seed_keyword"].map(SEED_TRANSLATIONS).fillna("")
    df["query_bs"] = (df["suggested_keyword"] + " " + df["bosnian_expansion"]).str.strip()

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved → {output_path}")
    print(df[["suggested_keyword", "bosnian_expansion"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
