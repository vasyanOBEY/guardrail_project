from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline


REQUIRED_COLUMNS = {"id", "prompt", "response", "label"}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def resolve_split(explicit_path: str | None, stem: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    for suffix in (".csv", ".parquet", ".pq"):
        candidate = Path("data") / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find data/{stem}.csv or data/{stem}.parquet. "
        f"Pass the path explicitly with a command-line argument."
    )


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")

    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["prompt"] = out["prompt"].fillna("").astype(str)
    out["response"] = out["response"].fillna("").astype(str)
    out["label"] = out["label"].astype(int)

    if not out["label"].isin([0, 1]).all():
        raise ValueError(f"{name}: label must contain only 0/1")
    if out["id"].duplicated().any():
        raise ValueError(f"{name}: duplicate ids found")

    return out


def build_text(df: pd.DataFrame) -> pd.Series:
    return "[PROMPT] " + df["prompt"] + " [RESPONSE] " + df["response"]


def make_word_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def make_word_char_model(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=150_000,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_candidates(x_train: pd.Series, y_train: pd.Series) -> tuple[pd.DataFrame, dict]:
    min_class = int(y_train.value_counts().min())
    n_splits = min(3, min_class)
    if n_splits < 2:
        raise ValueError("Not enough examples in one of the classes for cross-validation.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall"}

    rows: list[dict] = []
    candidates: list[tuple[str, float, Pipeline]] = []

    for c_value in (0.5, 1.0, 2.0, 4.0):
        candidates.append(("word_tfidf", c_value, make_word_model(c_value)))
        candidates.append(("word_char_tfidf", c_value, make_word_char_model(c_value)))

    for model_name, c_value, model in candidates:
        scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": model_name,
                "C": c_value,
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_f1_std": float(scores["test_f1"].std()),
                "cv_precision_mean": float(scores["test_precision"].mean()),
                "cv_recall_mean": float(scores["test_recall"].mean()),
                "cv_folds": n_splits,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["cv_f1_mean", "cv_precision_mean", "cv_recall_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = results.iloc[0].to_dict()
    return results, best


def build_model(model_name: str, c_value: float) -> Pipeline:
    if model_name == "word_tfidf":
        return make_word_model(c_value)
    if model_name == "word_char_tfidf":
        return make_word_char_model(c_value)
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and select an ML verifier using CV on train only. "
            "Outputs raw id,score files for threshold and holdout; "
            "threshold selection is intentionally left to the evaluator."
        )
    )
    parser.add_argument("--train", default=None)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    train_path = resolve_split(args.train, "split_train")
    threshold_path = resolve_split(args.threshold, "split_threshold")
    holdout_path = resolve_split(args.holdout, "split_holdout")

    train = validate(read_table(train_path), "train")
    threshold = validate(read_table(threshold_path), "threshold")
    holdout = validate(read_table(holdout_path), "holdout")

    x_train = build_text(train)
    y_train = train["label"]

    cv_results, best = evaluate_candidates(x_train, y_train)

    best_model = build_model(str(best["model"]), float(best["C"]))
    best_model.fit(x_train, y_train)

    threshold_scores = best_model.predict_proba(build_text(threshold))[:, 1]
    holdout_scores = best_model.predict_proba(build_text(holdout))[:, 1]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": threshold["id"], "score": threshold_scores}).to_csv(
        outdir / "verifier_threshold_scores.csv", index=False
    )
    pd.DataFrame({"id": holdout["id"], "score": holdout_scores}).to_csv(
        outdir / "verifier_holdout_scores.csv", index=False
    )
    cv_results.to_csv(outdir / "verifier_cv_results.csv", index=False)

    selection = {
        "selected_model": str(best["model"]),
        "selected_C": float(best["C"]),
        "selection_rule": "highest mean CV F1 on train; precision then recall as tie-breakers",
        "cv_f1_mean": float(best["cv_f1_mean"]),
        "cv_f1_std": float(best["cv_f1_std"]),
        "cv_precision_mean": float(best["cv_precision_mean"]),
        "cv_recall_mean": float(best["cv_recall_mean"]),
        "cv_folds": int(best["cv_folds"]),
        "train_rows": int(len(train)),
        "threshold_rows": int(len(threshold)),
        "holdout_rows": int(len(holdout)),
        "note": (
            "No threshold is selected here. Send the two id,score files to the "
            "shared evaluator so threshold choice is identical for the whole team."
        ),
    }

    with open(outdir / "verifier_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {"model": best_model, "selection": selection},
        outdir / "verifier.joblib",
    )

    print("Done.")
    print(f"Selected model: {selection['selected_model']}  C={selection['selected_C']}")
    print(f"CV F1: {selection['cv_f1_mean']:.4f} +/- {selection['cv_f1_std']:.4f}")
    print(f"Scores: {outdir / 'verifier_threshold_scores.csv'}")
    print(f"Scores: {outdir / 'verifier_holdout_scores.csv'}")
    print("Threshold is NOT chosen in this script.")


if __name__ == "__main__":
    main()
