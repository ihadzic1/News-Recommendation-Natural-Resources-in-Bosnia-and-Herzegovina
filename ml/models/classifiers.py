"""Classification of article_class_name from article text using TF-IDF features."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


class ClassifierSuite:
    """Trains Multinomial Naive Bayes, Decision Tree and Random Forest on
    TF-IDF features and produces a comparison of evaluation metrics."""

    def __init__(self, max_depth=30, n_estimators=200, test_size=0.2, random_state=42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        self.labels = []

    def _models(self):
        return {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=self.max_depth, random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }

    def run(self, X, y):
        """Train/test split, fit all models and collect metrics.
        Returns a dict: model name -> metrics and confusion matrix."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        self.labels = sorted(set(y))
        self.results = {}
        for name, model in self._models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=self.labels).tolist(),
            }
        return self.results

    def comparison_table(self):
        """Return results as a list of row dicts sorted by F1 score."""
        rows = [
            {
                "model": name,
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            }
            for name, m in self.results.items()
        ]
        return sorted(rows, key=lambda r: r["f1"], reverse=True)
