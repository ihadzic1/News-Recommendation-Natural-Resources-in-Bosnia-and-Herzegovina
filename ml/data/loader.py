"""Dataset loading, missing-value analysis and IQR-based outlier handling."""

import pandas as pd

REQUIRED_COLUMNS = [
    "title", "text", "article_class", "article_class_name",
    "num_of_comments", "num_of_shares", "link", "picture_path",
]


class DataLoader:
    """Loads the klix.ba dataset and applies basic data-quality steps."""

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        """Read the CSV, coerce numeric columns and drop rows without text."""
        df = pd.read_csv(self.csv_path)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df["num_of_comments"] = pd.to_numeric(df["num_of_comments"], errors="coerce").fillna(0)
        df["num_of_shares"] = pd.to_numeric(df["num_of_shares"], errors="coerce").fillna(0)
        df = df.dropna(subset=["text", "title"]).reset_index(drop=True)
        df["article_class"] = df["article_class"].fillna("Nepoznato").astype(str)
        df["article_class_name"] = df["article_class_name"].fillna("Nepoznato").astype(str)
        return df

    @staticmethod
    def missing_value_report(df):
        """Return a DataFrame with missing counts and percentages per column."""
        report = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
        })
        return report.sort_values("missing_count", ascending=False)

    @staticmethod
    def iqr_bounds(series):
        """Compute (lower, upper) IQR fences for a numeric series."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    @classmethod
    def clip_outliers(cls, df, columns=("num_of_comments", "num_of_shares")):
        """Clip values outside the IQR fences. Returns (clipped_df, info_dict)."""
        df = df.copy()
        info = {}
        for col in columns:
            lower, upper = cls.iqr_bounds(df[col])
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            info[col] = {"lower": float(lower), "upper": float(upper), "n_outliers": int(outliers)}
            df[col + "_clipped"] = df[col].clip(lower=max(lower, 0), upper=upper)
        return df, info
