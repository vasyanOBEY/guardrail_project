import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def read_table(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".parquet":
        return pd.read_parquet(path_obj)
    if path_obj.suffix.lower() == ".csv":
        return pd.read_csv(path_obj)
    raise ValueError(f"Unsupported file format: {path_obj.suffix}")


def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        "[PROMPT] "
        + df["prompt"].fillna("").astype(str)
        + " [RESPONSE] "
        + df["response"].fillna("").astype(str)
    )


def validate(df: pd.DataFrame, name: str) -> None:
    required = {"id", "prompt", "response", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    unique_scores = np.unique(scores)
    candidates = [0.0, 1.0]
    candidates.extend(unique_scores.tolist())
    if len(unique_scores) > 1:
        candidates.extend(((unique_scores[:-1] + unique_scores[1:]) / 2).tolist())

    best_threshold = 0.5
    best_key = (-1.0, -1.0, float("-inf"))

    for threshold in sorted(set(candidates)):
        pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        key = (f1, precision, -abs(threshold - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)

    return best_threshold


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/split_train.parquet")
    parser.add_argument("--threshold", default="data/split_threshold.parquet")
    parser.add_argument("--holdout", default="data/split_holdout.parquet")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    train = read_table(args.train)
    threshold_df = read_table(args.threshold)
    holdout = read_table(args.holdout)

    for name, df in [
        ("train", train),
        ("threshold", threshold_df),
        ("holdout", holdout),
    ]:
        validate(df, name)

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
    model.fit(build_text(train), train["label"].astype(int))

    threshold_scores = model.predict_proba(build_text(threshold_df))[:, 1]
    holdout_scores = model.predict_proba(build_text(holdout))[:, 1]

    y_threshold = threshold_df["label"].astype(int).to_numpy()
    y_holdout = holdout["label"].astype(int).to_numpy()

    best_threshold = choose_threshold(y_threshold, threshold_scores)

    report = {
        "best_threshold": best_threshold,
        "threshold": metrics(y_threshold, threshold_scores, best_threshold),
        "holdout": metrics(y_holdout, holdout_scores, best_threshold),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"id": threshold_df["id"], "score": threshold_scores}
    ).to_csv(outdir / "threshold_id_score.csv", index=False)

    pd.DataFrame(
        {"id": holdout["id"], "score": holdout_scores}
    ).to_csv(outdir / "holdout_id_score.csv", index=False)

    pd.DataFrame(
        {
            "id": threshold_df["id"],
            "label": y_threshold,
            "score": threshold_scores,
            "prediction": (threshold_scores >= best_threshold).astype(int),
        }
    ).to_csv(outdir / "threshold_detailed.csv", index=False)

    pd.DataFrame(
        {
            "id": holdout["id"],
            "label": y_holdout,
            "score": holdout_scores,
            "prediction": (holdout_scores >= best_threshold).astype(int),
        }
    ).to_csv(outdir / "holdout_detailed.csv", index=False)

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {"model": model, "threshold": best_threshold},
        outdir / "tfidf_logreg.joblib",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved results to: {outdir}")


if __name__ == "__main__":
    main()
