from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from train_verifier_cv import (
    RANDOM_STATE,
    build_representation as build_structural_representation,
    make_model,
    model_scores,
    read_table,
    resolve_split,
    validate,
    validate_split_separation,
)
from run_semantic_verifier import (
    build_representation as build_semantic_representation,
    make_text_model,
)


SEMANTIC_C = 0.25
ALPHAS = (0.0, 0.15, 0.25, 0.35)


def load_semantic_features(outdir: Path, split: str) -> pd.DataFrame:
    path = outdir / f"semantic_{split}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/run_semantic_verifier.py once first."
        )
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path}: missing id column")
    df["id"] = df["id"].astype(str)
    return df


def subset_semantic(
    semantic: pd.DataFrame,
    ids: pd.Series,
) -> pd.DataFrame:
    wanted = {str(x) for x in ids}
    part = semantic[semantic["id"].astype(str).isin(wanted)].copy()
    missing = wanted - set(part["id"].astype(str))
    if missing:
        raise ValueError(
            f"Semantic features missing {len(missing)} ids"
        )
    return part


def blend_scores(
    structural_scores,
    semantic_scores,
    alpha: float,
):
    return (
        (1.0 - alpha) * structural_scores
        + alpha * semantic_scores
    )


def evaluate(train: pd.DataFrame, train_sem: pd.DataFrame, repeats: int):
    cv = RepeatedStratifiedKFold(
        n_splits=3,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )

    rows = []
    metrics = {
        alpha: {"f1": [], "precision": [], "recall": []}
        for alpha in ALPHAS
    }

    for train_idx, valid_idx in cv.split(train, train["label"]):
        fold_train = train.iloc[train_idx].copy()
        fold_valid = train.iloc[valid_idx].copy()

        fold_train_sem = subset_semantic(train_sem, fold_train["id"])
        fold_valid_sem = subset_semantic(train_sem, fold_valid["id"])

        structural_model = make_model("word", 0.5, "logreg")
        structural_model.fit(
            build_structural_representation(
                fold_train,
                "response_support",
            ),
            fold_train["label"],
        )
        structural_scores = model_scores(
            structural_model,
            build_structural_representation(
                fold_valid,
                "response_support",
            ),
        )

        semantic_model = make_text_model(SEMANTIC_C)
        semantic_model.fit(
            build_semantic_representation(
                fold_train,
                fold_train_sem,
            ),
            fold_train["label"],
        )
        semantic_scores = semantic_model.predict_proba(
            build_semantic_representation(
                fold_valid,
                fold_valid_sem,
            )
        )[:, 1]

        y_true = fold_valid["label"].to_numpy()

        for alpha in ALPHAS:
            blended = blend_scores(
                structural_scores,
                semantic_scores,
                alpha,
            )
            pred = (blended >= 0.5).astype(int)
            metrics[alpha]["f1"].append(
                f1_score(y_true, pred, zero_division=0)
            )
            metrics[alpha]["precision"].append(
                precision_score(y_true, pred, zero_division=0)
            )
            metrics[alpha]["recall"].append(
                recall_score(y_true, pred, zero_division=0)
            )

    for alpha in ALPHAS:
        f1_values = pd.Series(metrics[alpha]["f1"])
        rows.append(
            {
                "alpha_semantic": alpha,
                "alpha_structural": 1.0 - alpha,
                "cv_f1_mean": float(f1_values.mean()),
                "cv_f1_std": float(f1_values.std(ddof=0)),
                "cv_precision_mean": float(
                    pd.Series(metrics[alpha]["precision"]).mean()
                ),
                "cv_recall_mean": float(
                    pd.Series(metrics[alpha]["recall"]).mean()
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        [
            "cv_f1_mean",
            "cv_f1_std",
            "cv_precision_mean",
            "cv_recall_mean",
        ],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return results, results.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blend the current structural verifier with already-computed "
            "semantic NLI features. No NLI inference is repeated."
        )
    )
    parser.add_argument("--train", default=None)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--cv-repeats", type=int, default=3)
    args = parser.parse_args()

    outdir = Path(args.outdir)

    train = validate(
        read_table(resolve_split(args.train, "split_train")),
        "train",
    )
    threshold = validate(
        read_table(resolve_split(args.threshold, "split_threshold")),
        "threshold",
    )
    holdout = validate(
        read_table(resolve_split(args.holdout, "split_holdout")),
        "holdout",
    )
    validate_split_separation(train, threshold, holdout)

    train_sem = load_semantic_features(outdir, "train")
    threshold_sem = load_semantic_features(outdir, "threshold")
    holdout_sem = load_semantic_features(outdir, "holdout")

    print("Cross-validation: structural + semantic blend...", flush=True)
    results, best = evaluate(
        train,
        train_sem,
        repeats=args.cv_repeats,
    )
    results.to_csv(
        outdir / "ensemble_cv_results.csv",
        index=False,
    )

    alpha = float(best["alpha_semantic"])

    structural_model = make_model("word", 0.5, "logreg")
    structural_model.fit(
        build_structural_representation(
            train,
            "response_support",
        ),
        train["label"],
    )

    semantic_model = make_text_model(SEMANTIC_C)
    semantic_model.fit(
        build_semantic_representation(train, train_sem),
        train["label"],
    )

    threshold_structural = model_scores(
        structural_model,
        build_structural_representation(
            threshold,
            "response_support",
        ),
    )
    holdout_structural = model_scores(
        structural_model,
        build_structural_representation(
            holdout,
            "response_support",
        ),
    )

    threshold_semantic = semantic_model.predict_proba(
        build_semantic_representation(
            threshold,
            threshold_sem,
        )
    )[:, 1]
    holdout_semantic = semantic_model.predict_proba(
        build_semantic_representation(
            holdout,
            holdout_sem,
        )
    )[:, 1]

    threshold_scores = blend_scores(
        threshold_structural,
        threshold_semantic,
        alpha,
    )
    holdout_scores = blend_scores(
        holdout_structural,
        holdout_semantic,
        alpha,
    )

    pd.DataFrame(
        {"id": threshold["id"], "score": threshold_scores}
    ).to_csv(
        outdir / "ensemble_verifier_threshold_scores.csv",
        index=False,
    )
    pd.DataFrame(
        {"id": holdout["id"], "score": holdout_scores}
    ).to_csv(
        outdir / "ensemble_verifier_holdout_scores.csv",
        index=False,
    )

    summary = {
        "selected_alpha_semantic": alpha,
        "selected_alpha_structural": 1.0 - alpha,
        "semantic_C": SEMANTIC_C,
        "cv_f1_mean": float(best["cv_f1_mean"]),
        "cv_f1_std": float(best["cv_f1_std"]),
        "cv_precision_mean": float(best["cv_precision_mean"]),
        "cv_recall_mean": float(best["cv_recall_mean"]),
        "note": (
            "Selection uses train-only repeated CV. "
            "alpha=0.0 is the current structural baseline. "
            "Threshold and holdout labels are never used for selection."
        ),
    }
    with open(
        outdir / "ensemble_selection.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    baseline = results[
        results["alpha_semantic"] == 0.0
    ].iloc[0]

    print("Done.")
    print(
        f"Baseline CV F1: {baseline['cv_f1_mean']:.4f} "
        f"+/- {baseline['cv_f1_std']:.4f}"
    )
    print(
        f"Best blend CV F1: {best['cv_f1_mean']:.4f} "
        f"+/- {best['cv_f1_std']:.4f}"
    )
    print(
        f"Selected semantic weight: {alpha:.2f} "
        f"(structural={1.0 - alpha:.2f})"
    )
    print(
        "Scores: "
        + str(outdir / "ensemble_verifier_threshold_scores.csv")
    )
    print(
        "Scores: "
        + str(outdir / "ensemble_verifier_holdout_scores.csv")
    )
    print("Threshold is NOT chosen in this script.")


if __name__ == "__main__":
    main()
