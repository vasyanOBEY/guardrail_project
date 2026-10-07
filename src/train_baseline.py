import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline


def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        "[PROMPT] "
        + df["prompt"].fillna("").astype(str)
        + " [RESPONSE] "
        + df["response"].fillna("").astype(str)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--valid", default="data/valid.csv")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    valid = pd.read_csv(args.valid)

    required = {"id", "prompt", "response", "label"}
    for name, df in [("train", train), ("valid", valid)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv: missing columns: {sorted(missing)}")

    x_train = build_text(train)
    x_valid = build_text(valid)
    y_train = train["label"].astype(int)
    y_valid = valid["label"].astype(int)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=100_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training TF-IDF + Logistic Regression...")
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_valid)[:, 1]

    best_threshold = 0.5
    best_f1 = -1.0
    for i in range(10, 91):
        threshold = i / 100
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_valid, pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    pred = (proba >= best_threshold).astype(int)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"F1:        {f1_score(y_valid, pred):.4f}")
    print(f"Precision: {precision_score(y_valid, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_valid, pred, zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_valid, pred))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = valid[["id"]].copy()
    scores["label_true"] = y_valid.to_numpy()
    scores["ml_score"] = proba
    scores["prediction"] = pred
    scores.to_csv(outdir / "valid_scores.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
        },
        outdir / "tfidf_logreg.joblib",
    )

    print(f"Saved: {outdir / 'valid_scores.csv'}")
    print(f"Saved: {outdir / 'tfidf_logreg.joblib'}")


if __name__ == "__main__":
    main()
