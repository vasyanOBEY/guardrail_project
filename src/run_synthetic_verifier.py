from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from train_verifier_cv import (
    RANDOM_STATE,
    SPECIAL_TOKEN_RE,
    NUMBER_RE,
    TOOL_CALL_MARKER_RE,
    STATUS_TERMS,
    build_representation,
    make_model,
    model_scores,
    read_table,
    resolve_split,
    validate,
    validate_split_separation,
)


STATUS_FLIPS = {
    "active": "inactive",
    "inactive": "active",
    "approved": "rejected",
    "rejected": "approved",
    "pending": "completed",
    "completed": "pending",
    "failed": "successful",
    "successful": "failed",
    "success": "failed",
    "confirmed": "cancelled",
    "cancelled": "confirmed",
    "canceled": "confirmed",
    "open": "closed",
    "closed": "open",
    "активен": "неактивен",
    "неактивен": "активен",
    "одобрено": "отклонено",
    "отклонено": "одобрено",
    "ожидает": "завершено",
    "ожидание": "завершено",
    "выполнено": "ожидание",
    "завершено": "ожидание",
    "ошибка": "успешно",
    "успешно": "ошибка",
    "подтверждено": "отменено",
    "отменено": "подтверждено",
    "открыт": "закрыт",
    "закрыт": "открыт",
}


def mutate_digits(token: str) -> str:
    chars = list(token)
    for i, ch in enumerate(chars):
        if ch.isdigit():
            chars[i] = str((int(ch) + 1) % 10)
            return "".join(chars)
    return token


def replace_once_case_insensitive(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, count=1, flags=re.IGNORECASE)


def synthetic_variants(prompt: str, response: str) -> list[str]:
    """
    Create a few deterministic hallucination-like responses from a clean example.

    We only mutate values that are actually grounded in the prompt, so replacing
    them creates an unsupported/contradictory answer by construction.
    """
    variants: list[str] = []
    prompt_lower = prompt.lower()

    # 1) Change one structured ID/number/date-like token that the response copied
    # from the prompt.
    structured = []
    for regex in (SPECIAL_TOKEN_RE, NUMBER_RE):
        for token in regex.findall(response):
            if token.lower() in prompt_lower and any(ch.isdigit() for ch in token):
                structured.append(token)
    for token in structured[:2]:
        mutated = mutate_digits(token)
        if mutated != token and mutated.lower() not in prompt_lower:
            variants.append(response.replace(token, mutated, 1))

    # 2) Flip one grounded status.
    response_words = re.findall(r"[A-Za-zА-Яа-яЁё]+", response)
    for word in response_words:
        lower = word.lower()
        if (
            lower in STATUS_TERMS
            and lower in STATUS_FLIPS
            and lower in prompt_lower
        ):
            replacement = STATUS_FLIPS[lower]
            if replacement not in prompt_lower:
                variants.append(
                    replace_once_case_insensitive(response, word, replacement)
                )
                break

    # 3) Corrupt one actual tool-call name into an unavailable sibling name.
    tool_match = TOOL_CALL_MARKER_RE.search(response)
    if tool_match:
        tool_name = tool_match.group(1)
        fake_tool = tool_name + "_alt"
        if fake_tool.lower() not in prompt_lower:
            start, end = tool_match.span(1)
            variants.append(response[:start] + fake_tool + response[end:])

    # Deduplicate and cap. Tiny train set means a small amount of targeted
    # synthetic data is safer than flooding the classifier with artificial text.
    unique: list[str] = []
    seen = {response}
    for value in variants:
        if value not in seen:
            seen.add(value)
            unique.append(value)
        if len(unique) >= 2:
            break
    return unique


def augment_training_fold(train_fold: pd.DataFrame) -> pd.DataFrame:
    synthetic_rows: list[dict] = []

    # Only clean examples are mutated. We never manufacture "clean" labels.
    clean = train_fold[train_fold["label"] == 0]
    for _, row in clean.iterrows():
        for idx, response in enumerate(
            synthetic_variants(row["prompt"], row["response"])
        ):
            synthetic_rows.append(
                {
                    "id": f"synthetic::{row['id']}::{idx}",
                    "prompt": row["prompt"],
                    "response": response,
                    "label": 1,
                }
            )

    if not synthetic_rows:
        return train_fold.copy()

    return pd.concat(
        [train_fold, pd.DataFrame(synthetic_rows)],
        ignore_index=True,
    )


def repeated_cv(train: pd.DataFrame, use_synthetic: bool, repeats: int) -> dict:
    cv = RepeatedStratifiedKFold(
        n_splits=3,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )

    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    synthetic_counts: list[int] = []

    for train_idx, valid_idx in cv.split(train, train["label"]):
        fold_train = train.iloc[train_idx].copy()
        fold_valid = train.iloc[valid_idx].copy()

        fit_df = augment_training_fold(fold_train) if use_synthetic else fold_train
        synthetic_counts.append(len(fit_df) - len(fold_train))

        model = make_model("word", 0.5, "logreg")
        model.fit(
            build_representation(fit_df, "response_support"),
            fit_df["label"],
        )

        # Important: validation is always the untouched original data.
        scores = model_scores(
            model,
            build_representation(fold_valid, "response_support"),
        )
        pred = [int(score >= 0.5) for score in scores]

        y_true = fold_valid["label"].tolist()
        f1_values.append(f1_score(y_true, pred, zero_division=0))
        precision_values.append(
            precision_score(y_true, pred, zero_division=0)
        )
        recall_values.append(recall_score(y_true, pred, zero_division=0))

    series = pd.Series(f1_values)
    return {
        "name": "synthetic_structural" if use_synthetic else "baseline_structural",
        "cv_f1_mean": float(series.mean()),
        "cv_f1_std": float(series.std(ddof=0)),
        "cv_precision_mean": float(pd.Series(precision_values).mean()),
        "cv_recall_mean": float(pd.Series(recall_values).mean()),
        "synthetic_rows_mean": float(pd.Series(synthetic_counts).mean()),
        "synthetic_rows_min": int(min(synthetic_counts)),
        "synthetic_rows_max": int(max(synthetic_counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only synthetic augmentation experiment for the structural verifier. "
            "Synthetic samples are generated inside each CV training fold only."
        )
    )
    parser.add_argument("--train", default=None)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--cv-repeats", type=int, default=3)
    args = parser.parse_args()

    train = validate(read_table(resolve_split(args.train, "split_train")), "train")
    threshold = validate(
        read_table(resolve_split(args.threshold, "split_threshold")),
        "threshold",
    )
    holdout = validate(
        read_table(resolve_split(args.holdout, "split_holdout")),
        "holdout",
    )
    validate_split_separation(train, threshold, holdout)

    print("Cross-validation: baseline...", flush=True)
    baseline = repeated_cv(train, use_synthetic=False, repeats=args.cv_repeats)

    print("Cross-validation: synthetic augmentation...", flush=True)
    synthetic = repeated_cv(train, use_synthetic=True, repeats=args.cv_repeats)

    results = pd.DataFrame([baseline, synthetic]).sort_values(
        ["cv_f1_mean", "cv_f1_std", "cv_precision_mean", "cv_recall_mean"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)

    best = results.iloc[0].to_dict()
    use_synthetic = str(best["name"]) == "synthetic_structural"

    final_train = augment_training_fold(train) if use_synthetic else train.copy()
    model = make_model("word", 0.5, "logreg")
    model.fit(
        build_representation(final_train, "response_support"),
        final_train["label"],
    )

    threshold_scores = model_scores(
        model,
        build_representation(threshold, "response_support"),
    )
    holdout_scores = model_scores(
        model,
        build_representation(holdout, "response_support"),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results.to_csv(outdir / "synthetic_cv_results.csv", index=False)
    pd.DataFrame(
        {"id": threshold["id"], "score": threshold_scores}
    ).to_csv(outdir / "synthetic_verifier_threshold_scores.csv", index=False)
    pd.DataFrame(
        {"id": holdout["id"], "score": holdout_scores}
    ).to_csv(outdir / "synthetic_verifier_holdout_scores.csv", index=False)

    summary = {
        "selected_candidate": str(best["name"]),
        "uses_synthetic": use_synthetic,
        "cv_f1_mean": float(best["cv_f1_mean"]),
        "cv_f1_std": float(best["cv_f1_std"]),
        "baseline": baseline,
        "synthetic": synthetic,
        "final_original_rows": int(len(train)),
        "final_synthetic_rows": int(len(final_train) - len(train)),
        "note": (
            "Synthetic data is generated only from clean examples inside each "
            "training fold. Validation, threshold and holdout examples are never "
            "used to generate training examples."
        ),
    }
    with open(
        outdir / "synthetic_selection.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(
        f"Baseline CV F1: {baseline['cv_f1_mean']:.4f} "
        f"+/- {baseline['cv_f1_std']:.4f}"
    )
    print(
        f"Synthetic CV F1: {synthetic['cv_f1_mean']:.4f} "
        f"+/- {synthetic['cv_f1_std']:.4f}"
    )
    print(
        f"Selected: {summary['selected_candidate']} "
        f"(uses_synthetic={summary['uses_synthetic']})"
    )
    print(
        f"Repeated-CV F1: {summary['cv_f1_mean']:.4f} "
        f"+/- {summary['cv_f1_std']:.4f}"
    )
    print(
        f"Final synthetic rows: {summary['final_synthetic_rows']}"
    )
    print(
        "Scores: "
        + str(outdir / "synthetic_verifier_threshold_scores.csv")
    )
    print(
        "Scores: "
        + str(outdir / "synthetic_verifier_holdout_scores.csv")
    )
    print("Threshold is NOT chosen in this script.")


if __name__ == "__main__":
    main()
