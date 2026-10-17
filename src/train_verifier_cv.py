from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline


REQUIRED_COLUMNS = {"id", "prompt", "response", "label"}
RANDOM_STATE = 42
TAIL_CHARS = 8_000

SPECIAL_TOKEN_RE = re.compile(
    r"\b(?:[A-Za-zА-Яа-яЁё_]*\d[A-Za-zА-Яа-яЁё0-9_.:-]*|"
    r"[A-Za-zА-Яа-яЁё0-9_.:-]*\d[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_.:-]*)\b"
)
NUMBER_RE = re.compile(r"(?<![\w])\d{2,}(?:[.,:/-]\d+)*(?![\w])")
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{4,}")
TOOL_CALL_RE = re.compile(r"TOOL_CALL\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
TOOL_RESULT_MARKER_RE = re.compile(r"TOOL_RESULT", re.IGNORECASE)
MESSAGE_SPLIT_RE = re.compile(
    r"(?=^(?:SYSTEM|USER|ASSISTANT|ASSISTANT_TOOL_CALL|USER_TOOL_CALL|TOOL_RESULT)\b)",
    re.IGNORECASE | re.MULTILINE,
)

STATUS_TERMS = {
    "active",
    "inactive",
    "approved",
    "rejected",
    "pending",
    "completed",
    "failed",
    "success",
    "successful",
    "cancelled",
    "canceled",
    "confirmed",
    "processing",
    "open",
    "closed",
    "активен",
    "активна",
    "неактивен",
    "одобрено",
    "отклонено",
    "ожидает",
    "ожидание",
    "выполнено",
    "завершено",
    "ошибка",
    "успешно",
    "отменено",
    "подтверждено",
    "подтверждён",
    "обрабатывается",
    "открыт",
    "закрыт",
}


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
        "Pass the path explicitly with a command-line argument."
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
    if out["prompt"].str.strip().eq("").any():
        raise ValueError(f"{name}: empty prompt found")
    if out["response"].str.strip().eq("").any():
        raise ValueError(f"{name}: empty response found")

    return out


def validate_split_separation(
    train: pd.DataFrame,
    threshold: pd.DataFrame,
    holdout: pd.DataFrame,
) -> None:
    splits = {"train": train, "threshold": threshold, "holdout": holdout}
    names = list(splits)

    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left = splits[left_name]
            right = splits[right_name]

            id_overlap = set(left["id"]) & set(right["id"])
            if id_overlap:
                raise ValueError(
                    f"Leakage: {left_name}/{right_name} share {len(id_overlap)} ids"
                )

            left_pairs = set(zip(left["prompt"], left["response"]))
            right_pairs = set(zip(right["prompt"], right["response"]))
            pair_overlap = left_pairs & right_pairs
            if pair_overlap:
                raise ValueError(
                    f"Leakage: {left_name}/{right_name} share "
                    f"{len(pair_overlap)} exact prompt/response pairs"
                )


def _normalize_token(token: str) -> str:
    return token.lower().strip(" \t\n\r.,;:!?()[]{}<>\"'")


def _extract_statuses(text: str) -> set[str]:
    words = {_normalize_token(x) for x in WORD_RE.findall(text)}
    return {x for x in words if x in STATUS_TERMS}


def _chunk_prompt(prompt: str, max_chunk_chars: int = 2200) -> list[str]:
    parts = [x.strip() for x in MESSAGE_SPLIT_RE.split(prompt) if x.strip()]
    if len(parts) <= 1:
        parts = [x.strip() for x in re.split(r"\n\s*\n", prompt) if x.strip()]
    if not parts:
        return [prompt]

    chunks: list[str] = []
    for part in parts:
        if len(part) <= max_chunk_chars:
            chunks.append(part)
            continue

        step = max_chunk_chars - 250
        for start in range(0, len(part), step):
            piece = part[start : start + max_chunk_chars].strip()
            if piece:
                chunks.append(piece)
            if start + max_chunk_chars >= len(part):
                break
    return chunks


def retrieve_context(
    prompt: str,
    response: str,
    max_chunks: int = 4,
    max_chars: int = 8000,
) -> str:
    """
    Select a few prompt fragments that are most relevant to the response.

    Retrieval is label-free: it uses only lexical overlap, structured values
    (numbers/IDs/dates/times), TOOL_RESULT presence and a tiny recency bonus.
    """
    chunks = _chunk_prompt(prompt)
    if not chunks:
        return prompt[-max_chars:]

    response_words = {x.lower() for x in WORD_RE.findall(response)}
    structured = {
        _normalize_token(x)
        for regex in (SPECIAL_TOKEN_RE, NUMBER_RE, DATE_RE, TIME_RE)
        for x in regex.findall(response)
        if _normalize_token(x)
    }

    scored: list[tuple[float, int, str]] = []
    total = max(1, len(chunks) - 1)

    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        chunk_words = {x.lower() for x in WORD_RE.findall(chunk)}
        lexical_overlap = (
            len(response_words & chunk_words) / max(1, len(response_words))
        )
        structured_hits = sum(1 for token in structured if token in chunk_lower)

        score = lexical_overlap
        score += 0.35 * structured_hits
        if TOOL_RESULT_MARKER_RE.search(chunk):
            score += 0.10
        score += 0.05 * (idx / total)

        scored.append((score, idx, chunk))

    best = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[:max_chunks]

    # Keep the newest message too. It is often the immediate context for the
    # response and costs little compared with feeding the whole prompt.
    newest_idx = len(chunks) - 1
    if all(idx != newest_idx for _, idx, _ in best):
        best[-1] = (scored[newest_idx][0], newest_idx, chunks[newest_idx])

    selected = sorted(best, key=lambda x: x[1])
    text = "\n\n".join(chunk for _, _, chunk in selected)
    return text[-max_chars:]


def support_signature(prompt: str, response: str) -> str:
    """
    Build relational features that compare a response with its own context.

    No labels/explanations are used here. Every feature is available at hidden-test
    inference time, so this does not create leakage.
    """
    prompt_lower = prompt.lower()

    special_tokens = [_normalize_token(x) for x in SPECIAL_TOKEN_RE.findall(response)]
    number_tokens = [_normalize_token(x) for x in NUMBER_RE.findall(response)]
    date_tokens = [_normalize_token(x) for x in DATE_RE.findall(response)]
    time_tokens = [_normalize_token(x) for x in TIME_RE.findall(response)]
    tool_calls = [x.lower() for x in TOOL_CALL_RE.findall(response)]

    unseen_specials = [x for x in special_tokens if x and x not in prompt_lower]
    unseen_numbers = [x for x in number_tokens if x and x not in prompt_lower]
    unseen_dates = [x for x in date_tokens if x and x not in prompt_lower]
    unseen_times = [x for x in time_tokens if x and x not in prompt_lower]
    unseen_tools = [x for x in tool_calls if x and x not in prompt_lower]

    prompt_statuses = _extract_statuses(prompt)
    response_statuses = _extract_statuses(response)
    unseen_statuses = sorted(response_statuses - prompt_statuses)
    status_changed = bool(
        prompt_statuses and response_statuses and not (prompt_statuses & response_statuses)
    )

    response_words = {x.lower() for x in WORD_RE.findall(response)}
    prompt_words = {x.lower() for x in WORD_RE.findall(prompt)}
    overlap_ratio = (
        len(response_words & prompt_words) / len(response_words)
        if response_words
        else 1.0
    )
    overlap_bucket = min(9, int(overlap_ratio * 10))

    has_tool_result = bool(TOOL_RESULT_MARKER_RE.search(prompt))

    features = [
        "SUPPORT_SIGNATURE",
        f"WORD_OVERLAP_{overlap_bucket}",
        f"HAS_TOOL_RESULT_{int(has_tool_result)}",
        f"UNSEEN_SPECIAL_COUNT_{min(len(unseen_specials), 5)}",
        f"UNSEEN_NUMBER_COUNT_{min(len(unseen_numbers), 5)}",
        f"UNSEEN_DATE_COUNT_{min(len(unseen_dates), 3)}",
        f"UNSEEN_TIME_COUNT_{min(len(unseen_times), 3)}",
        f"UNSEEN_STATUS_COUNT_{min(len(unseen_statuses), 3)}",
        f"UNSEEN_TOOL_COUNT_{min(len(unseen_tools), 3)}",
        f"STATUS_CHANGED_{int(status_changed)}",
    ]

    if has_tool_result and (
        unseen_specials or unseen_numbers or unseen_dates or unseen_statuses
    ):
        features.append("UNSUPPORTED_AFTER_TOOL_RESULT")

    # Keep only a few concrete values so one long answer cannot explode the
    # feature space. Counts above remain available for all examples.
    features.extend(f"UNSEEN_SPECIAL_{x}" for x in unseen_specials[:8])
    features.extend(f"UNSEEN_DATE_{x}" for x in unseen_dates[:4])
    features.extend(f"UNSEEN_STATUS_{x}" for x in unseen_statuses[:4])
    features.extend(f"UNSEEN_TOOL_{x}" for x in unseen_tools[:4])
    return " ".join(features)


def build_representation(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "full":
        return "[PROMPT] " + df["prompt"] + " [RESPONSE] " + df["response"]

    if mode == "response":
        return "[RESPONSE] " + df["response"]

    if mode == "tail8k":
        return (
            "[PROMPT_TAIL] "
            + df["prompt"].str[-TAIL_CHARS:]
            + " [RESPONSE] "
            + df["response"]
        )

    if mode == "response_support":
        values = []
        for prompt, response in zip(df["prompt"], df["response"]):
            values.append(
                "[RESPONSE] "
                + response
                + " [RELATIONAL_FEATURES] "
                + support_signature(prompt, response)
            )
        return pd.Series(values, index=df.index)

    if mode == "tail8k_support":
        values = []
        for prompt, response in zip(df["prompt"], df["response"]):
            values.append(
                "[PROMPT_TAIL] "
                + prompt[-TAIL_CHARS:]
                + " [RESPONSE] "
                + response
                + " [RELATIONAL_FEATURES] "
                + support_signature(prompt, response)
            )
        return pd.Series(values, index=df.index)

    if mode == "retrieved":
        values = []
        for prompt, response in zip(df["prompt"], df["response"]):
            values.append(
                "[RETRIEVED_CONTEXT] "
                + retrieve_context(prompt, response)
                + " [RESPONSE] "
                + response
            )
        return pd.Series(values, index=df.index)

    if mode == "retrieved_support":
        values = []
        for prompt, response in zip(df["prompt"], df["response"]):
            values.append(
                "[RETRIEVED_CONTEXT] "
                + retrieve_context(prompt, response)
                + " [RESPONSE] "
                + response
                + " [RELATIONAL_FEATURES] "
                + support_signature(prompt, response)
            )
        return pd.Series(values, index=df.index)

    raise ValueError(f"Unknown representation: {mode}")


def make_model(feature_kind: str, c_value: float) -> Pipeline:
    if feature_kind == "word":
        features = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
            max_features=50_000,
        )
    elif feature_kind == "word_char":
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
                        max_features=40_000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        sublinear_tf=True,
                        max_features=60_000,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown feature kind: {feature_kind}")

    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def candidate_specs() -> list[dict]:
    specs = [
        {
            "name": "legacy_full_word",
            "representation": "full",
            "feature_kind": "word",
            "C": 1.0,
        }
    ]

    # Response-only is a useful control: it tells us whether the dataset has
    # strong answer-style signal. We do not assume it is the final architecture.
    for c_value in (0.5, 1.0, 2.0):
        specs.append(
            {
                "name": "response_word",
                "representation": "response",
                "feature_kind": "word",
                "C": c_value,
            }
        )
        specs.append(
            {
                "name": "response_word_char",
                "representation": "response",
                "feature_kind": "word_char",
                "C": c_value,
            }
        )
        specs.append(
            {
                "name": "tail8k_word",
                "representation": "tail8k",
                "feature_kind": "word",
                "C": c_value,
            }
        )
        specs.append(
            {
                "name": "response_support_word_char",
                "representation": "response_support",
                "feature_kind": "word_char",
                "C": c_value,
            }
        )
        specs.append(
            {
                "name": "tail8k_support_word_char",
                "representation": "tail8k_support",
                "feature_kind": "word_char",
                "C": c_value,
            }
        )
        specs.append(
            {
                "name": "retrieved_word_char",
                "representation": "retrieved",
                "feature_kind": "word_char",
                "C": c_value,
            }
        )
        specs.append(
            {
                "name": "retrieved_support_word_char",
                "representation": "retrieved_support",
                "feature_kind": "word_char",
                "C": c_value,
            }
        )

    return specs


def evaluate_candidates(
    train: pd.DataFrame,
    repeats: int,
) -> tuple[pd.DataFrame, dict]:
    y_train = train["label"]
    min_class = int(y_train.value_counts().min())
    n_splits = min(3, min_class)
    if n_splits < 2:
        raise ValueError("Not enough examples in one class for cross-validation.")

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall"}

    representation_cache: dict[str, pd.Series] = {}
    rows: list[dict] = []

    for spec in candidate_specs():
        representation = spec["representation"]
        if representation not in representation_cache:
            representation_cache[representation] = build_representation(
                train, representation
            )

        x_train = representation_cache[representation]
        model = make_model(spec["feature_kind"], float(spec["C"]))

        started = time.perf_counter()
        scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        wall_seconds = time.perf_counter() - started

        rows.append(
            {
                **spec,
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_f1_std": float(scores["test_f1"].std()),
                "cv_precision_mean": float(scores["test_precision"].mean()),
                "cv_recall_mean": float(scores["test_recall"].mean()),
                "fit_time_mean_sec": float(scores["fit_time"].mean()),
                "score_time_mean_sec": float(scores["score_time"].mean()),
                "wall_time_sec": float(wall_seconds),
                "cv_folds_total": int(n_splits * repeats),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["cv_f1_mean", "cv_f1_std", "cv_precision_mean", "cv_recall_mean"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)

    best = results.iloc[0].to_dict()
    return results, best


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def split_summary(df: pd.DataFrame) -> dict:
    prompt_len = df["prompt"].str.len()
    response_len = df["response"].str.len()
    return {
        "rows": int(len(df)),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum()),
        "prompt_chars_median": float(prompt_len.median()),
        "prompt_chars_max": int(prompt_len.max()),
        "response_chars_median": float(response_len.median()),
        "response_chars_max": int(response_len.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train/select an ML verifier using train-only CV. "
            "Outputs raw id,score files. Final threshold belongs to the shared evaluator."
        )
    )
    parser.add_argument("--train", default=None)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=3,
        help="Repeated 3-fold CV repeats on train only. Default: 3.",
    )
    args = parser.parse_args()

    if args.cv_repeats < 1:
        raise ValueError("--cv-repeats must be >= 1")

    train_path = resolve_split(args.train, "split_train")
    threshold_path = resolve_split(args.threshold, "split_threshold")
    holdout_path = resolve_split(args.holdout, "split_holdout")

    train = validate(read_table(train_path), "train")
    threshold = validate(read_table(threshold_path), "threshold")
    holdout = validate(read_table(holdout_path), "holdout")
    validate_split_separation(train, threshold, holdout)

    cv_results, best = evaluate_candidates(train, repeats=args.cv_repeats)

    best_representation = str(best["representation"])
    best_feature_kind = str(best["feature_kind"])
    best_c = float(best["C"])

    best_model = make_model(best_feature_kind, best_c)
    best_model.fit(build_representation(train, best_representation), train["label"])

    threshold_scores = best_model.predict_proba(
        build_representation(threshold, best_representation)
    )[:, 1]
    holdout_scores = best_model.predict_proba(
        build_representation(holdout, best_representation)
    )[:, 1]

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
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "selected_candidate": str(best["name"]),
        "selected_representation": best_representation,
        "selected_feature_kind": best_feature_kind,
        "selected_C": best_c,
        "selection_rule": (
            "highest mean train-only repeated-CV F1; then lower F1 std; "
            "then precision; then recall"
        ),
        "cv_f1_mean": float(best["cv_f1_mean"]),
        "cv_f1_std": float(best["cv_f1_std"]),
        "cv_precision_mean": float(best["cv_precision_mean"]),
        "cv_recall_mean": float(best["cv_recall_mean"]),
        "cv_folds_total": int(best["cv_folds_total"]),
        "cv_repeats": int(args.cv_repeats),
        "train": split_summary(train),
        "threshold": split_summary(threshold),
        "holdout": split_summary(holdout),
        "note": (
            "No threshold is selected here. threshold/holdout labels are not used "
            "for model selection. Send the two id,score files to the shared evaluator."
        ),
    }

    with open(outdir / "verifier_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "model": best_model,
            "representation": best_representation,
            "tail_chars": TAIL_CHARS,
            "selection": selection,
        },
        outdir / "verifier.joblib",
    )

    print("Done.")
    print(
        "Selected: "
        f"{selection['selected_candidate']} "
        f"(representation={best_representation}, features={best_feature_kind}, C={best_c})"
    )
    print(
        f"Repeated-CV F1: {selection['cv_f1_mean']:.4f} "
        f"+/- {selection['cv_f1_std']:.4f}"
    )
    print(f"Scores: {outdir / 'verifier_threshold_scores.csv'}")
    print(f"Scores: {outdir / 'verifier_holdout_scores.csv'}")
    print("Threshold is NOT chosen in this script.")


if __name__ == "__main__":
    main()
