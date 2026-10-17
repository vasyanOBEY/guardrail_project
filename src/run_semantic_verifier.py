from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_verifier_cv import (
    RANDOM_STATE,
    read_table,
    resolve_split,
    support_signature,
    validate,
    validate_split_separation,
)


DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
STRUCTURED_RE = re.compile(
    r"\b(?:[A-Za-zА-Яа-яЁё_]*\d[A-Za-zА-Яа-яЁё0-9_.:#/-]*|"
    r"\d{2,}(?:[.,:/-]\d+)*)\b"
)
TURN_MARKER_RE = re.compile(
    r"(?=⟦(?:SYSTEM|USER|ASSISTANT|TOOL)[^⟧]*⟧)",
    re.IGNORECASE,
)
TOOL_CALL_RE = re.compile(
    r"→\s*TOOL_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s*:",
    re.IGNORECASE,
)


def chunk_prompt(prompt: str, max_chars: int = 1400) -> list[str]:
    parts = [x.strip() for x in TURN_MARKER_RE.split(prompt) if x.strip()]
    if len(parts) <= 1:
        parts = [x.strip() for x in re.split(r"\n\s*\n", prompt) if x.strip()]
    if not parts:
        return [prompt[-max_chars:]]

    chunks: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            chunks.append(part)
            continue

        step = max_chars - 200
        for start in range(0, len(part), step):
            piece = part[start : start + max_chars].strip()
            if piece:
                chunks.append(piece)
            if start + max_chars >= len(part):
                break
    return chunks


def extract_claims(response: str, max_claims: int) -> list[str]:
    text = re.sub(r"^\s*⟦ASSISTANT[^⟧]*⟧\s*", "", response).strip()
    claims: list[str] = []

    # Natural-language text before the first tool call.
    match = TOOL_CALL_RE.search(text)
    natural = text[: match.start()].strip() if match else text
    if natural:
        for piece in re.split(r"(?<=[.!?])\s+|\n+", natural):
            piece = piece.strip(" \t-•")
            if len(piece) >= 20 and not piece.endswith("?"):
                claims.append(piece)

    # Tool-call summaries/reasons often contain factual claims. Pull string
    # values from JSON without pretending NLI can reason reliably over raw JSON.
    matches = list(TOOL_CALL_RE.finditer(text))
    for i, tool_match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[tool_match.end() : end]
        obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not obj_match:
            continue
        try:
            payload = json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            continue

        preferred_keys = ("summary", "reason", "description", "message")
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, str) and len(value.strip()) >= 20:
                claims.append(value.strip())

    # Deduplicate while keeping order.
    unique: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        normalized = re.sub(r"\s+", " ", claim.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(claim)
        if len(unique) >= max_claims:
            break

    return unique


def rank_chunks(prompt: str, claim: str, k: int) -> list[str]:
    chunks = chunk_prompt(prompt)
    claim_words = {x.lower() for x in WORD_RE.findall(claim)}
    structured = {x.lower() for x in STRUCTURED_RE.findall(claim)}

    scored: list[tuple[float, int, str]] = []
    total = max(1, len(chunks) - 1)
    for idx, chunk in enumerate(chunks):
        lower = chunk.lower()
        chunk_words = {x.lower() for x in WORD_RE.findall(chunk)}
        overlap = len(claim_words & chunk_words) / max(1, len(claim_words))
        structured_hits = sum(1 for token in structured if token in lower)

        score = overlap + 0.35 * structured_hits
        if "tool_result" in lower or "tool_response" in lower:
            score += 0.08
        if "[available tools]" in lower:
            score += 0.04
        score += 0.04 * (idx / total)
        scored.append((score, idx, chunk))

    best = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[:k]
    return [chunk for _, _, chunk in best]


def get_label_ids(model) -> tuple[int, int, int]:
    label2id = {
        str(name).lower(): int(idx)
        for name, idx in model.config.label2id.items()
    }
    try:
        return (
            label2id["entailment"],
            label2id["neutral"],
            label2id["contradiction"],
        )
    except KeyError as exc:
        raise ValueError(
            "NLI model must expose entailment/neutral/contradiction labels. "
            f"Got: {model.config.label2id}"
        ) from exc


@torch.inference_mode()
def infer_pairs(
    model,
    tokenizer,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int,
    device: torch.device,
) -> list[tuple[float, float, float]]:
    entail_id, neutral_id, contradiction_id = get_label_ids(model)
    results: list[tuple[float, float, float]] = []

    for start in range(0, len(premises), batch_size):
        batch_premises = premises[start : start + batch_size]
        batch_hypotheses = hypotheses[start : start + batch_size]
        encoded = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu()

        for row in probs:
            results.append(
                (
                    float(row[entail_id]),
                    float(row[neutral_id]),
                    float(row[contradiction_id]),
                )
            )
    return results


def semantic_features(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int,
    device: torch.device,
    max_claims: int,
    chunks_per_claim: int,
) -> pd.DataFrame:
    premises: list[str] = []
    hypotheses: list[str] = []
    pair_meta: list[tuple[int, int]] = []
    claims_by_row: dict[int, list[str]] = {}

    for row_pos, (_, row) in enumerate(df.iterrows()):
        claims = extract_claims(row["response"], max_claims=max_claims)
        claims_by_row[row_pos] = claims
        for claim_pos, claim in enumerate(claims):
            chunks = rank_chunks(
                row["prompt"],
                claim,
                k=chunks_per_claim,
            )
            for chunk in chunks:
                premises.append(chunk)
                hypotheses.append(claim)
                pair_meta.append((row_pos, claim_pos))

    scores = infer_pairs(
        model=model,
        tokenizer=tokenizer,
        premises=premises,
        hypotheses=hypotheses,
        batch_size=batch_size,
        device=device,
    ) if premises else []

    per_claim: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for meta, score in zip(pair_meta, scores):
        per_claim.setdefault(meta, []).append(score)

    rows: list[dict] = []
    for row_pos in range(len(df)):
        claims = claims_by_row[row_pos]
        unsupported_values: list[float] = []
        contradiction_values: list[float] = []
        support_values: list[float] = []
        neutral_values: list[float] = []

        for claim_pos in range(len(claims)):
            pair_scores = per_claim.get((row_pos, claim_pos), [])
            if not pair_scores:
                continue
            support = max(x[0] for x in pair_scores)
            neutral = max(x[1] for x in pair_scores)
            contradiction = max(x[2] for x in pair_scores)
            support_values.append(support)
            neutral_values.append(neutral)
            contradiction_values.append(contradiction)
            unsupported_values.append(1.0 - support)

        if unsupported_values:
            max_unsupported = max(unsupported_values)
            mean_unsupported = sum(unsupported_values) / len(unsupported_values)
            max_contradiction = max(contradiction_values)
            mean_support = sum(support_values) / len(support_values)
            max_neutral = max(neutral_values)
            low_support_claims = sum(x < 0.35 for x in support_values)
        else:
            max_unsupported = 0.0
            mean_unsupported = 0.0
            max_contradiction = 0.0
            mean_support = 1.0
            max_neutral = 0.0
            low_support_claims = 0

        rows.append(
            {
                "id": str(df.iloc[row_pos]["id"]),
                "claim_count": len(claims),
                "nli_max_unsupported": max_unsupported,
                "nli_mean_unsupported": mean_unsupported,
                "nli_max_contradiction": max_contradiction,
                "nli_mean_support": mean_support,
                "nli_max_neutral": max_neutral,
                "nli_low_support_claims": low_support_claims,
            }
        )

    return pd.DataFrame(rows)


def bucket(value: float) -> int:
    return min(9, max(0, int(math.floor(float(value) * 10))))


def semantic_tokens(row: pd.Series) -> str:
    if int(row["claim_count"]) == 0:
        return "NLI_NO_CLAIMS"

    return " ".join(
        [
            f"NLI_CLAIM_COUNT_{min(int(row['claim_count']), 6)}",
            f"NLI_MAX_UNSUPPORTED_{bucket(row['nli_max_unsupported'])}",
            f"NLI_MEAN_UNSUPPORTED_{bucket(row['nli_mean_unsupported'])}",
            f"NLI_MAX_CONTRADICTION_{bucket(row['nli_max_contradiction'])}",
            f"NLI_MEAN_SUPPORT_{bucket(row['nli_mean_support'])}",
            f"NLI_MAX_NEUTRAL_{bucket(row['nli_max_neutral'])}",
            f"NLI_LOW_SUPPORT_COUNT_{min(int(row['nli_low_support_claims']), 6)}",
        ]
    )


def build_representation(
    df: pd.DataFrame,
    semantic: pd.DataFrame | None,
) -> pd.Series:
    by_id = (
        semantic.set_index("id")
        if semantic is not None
        else None
    )
    values: list[str] = []

    for _, row in df.iterrows():
        value = (
            "[RESPONSE] "
            + row["response"]
            + " [RELATIONAL_FEATURES] "
            + support_signature(row["prompt"], row["response"])
        )
        if by_id is not None:
            semantic_row = by_id.loc[str(row["id"])]
            value += " [SEMANTIC_FEATURES] " + semantic_tokens(semantic_row)
        values.append(value)

    return pd.Series(values, index=df.index)


def make_text_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    sublinear_tf=True,
                    max_features=50_000,
                ),
            ),
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


def evaluate_candidates(
    train: pd.DataFrame,
    semantic: pd.DataFrame,
    repeats: int,
) -> tuple[pd.DataFrame, dict]:
    cv = RepeatedStratifiedKFold(
        n_splits=3,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall"}
    candidates = [
        ("baseline_structural", None, 0.5),
        ("structural_nli", semantic, 0.25),
        ("structural_nli", semantic, 0.5),
        ("structural_nli", semantic, 1.0),
    ]

    rows: list[dict] = []
    for name, semantic_features_df, c_value in candidates:
        x = build_representation(train, semantic_features_df)
        model = make_text_model(c_value)
        started = time.perf_counter()
        result = cross_validate(
            model,
            x,
            train["label"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        rows.append(
            {
                "name": name,
                "C": c_value,
                "uses_nli": semantic_features_df is not None,
                "cv_f1_mean": float(result["test_f1"].mean()),
                "cv_f1_std": float(result["test_f1"].std()),
                "cv_precision_mean": float(result["test_precision"].mean()),
                "cv_recall_mean": float(result["test_recall"].mean()),
                "wall_time_sec": float(time.perf_counter() - started),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["cv_f1_mean", "cv_f1_std", "cv_precision_mean", "cv_recall_mean"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return results, results.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental multilingual NLI verifier. It never tunes on threshold/"
            "holdout labels; those splits are used only to emit raw scores."
        )
    )
    parser.add_argument("--train", default=None)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-claims", type=int, default=6)
    parser.add_argument("--chunks-per-claim", type=int, default=3)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NLI model: {args.model}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_sem = semantic_features(
        train,
        model,
        tokenizer,
        args.batch_size,
        device,
        args.max_claims,
        args.chunks_per_claim,
    )
    threshold_sem = semantic_features(
        threshold,
        model,
        tokenizer,
        args.batch_size,
        device,
        args.max_claims,
        args.chunks_per_claim,
    )
    holdout_sem = semantic_features(
        holdout,
        model,
        tokenizer,
        args.batch_size,
        device,
        args.max_claims,
        args.chunks_per_claim,
    )

    train_sem.to_csv(outdir / "semantic_train_features.csv", index=False)
    threshold_sem.to_csv(outdir / "semantic_threshold_features.csv", index=False)
    holdout_sem.to_csv(outdir / "semantic_holdout_features.csv", index=False)

    cv_results, best = evaluate_candidates(
        train,
        train_sem,
        repeats=args.cv_repeats,
    )
    cv_results.to_csv(outdir / "semantic_cv_results.csv", index=False)

    use_nli = bool(best["uses_nli"])
    best_c = float(best["C"])
    final_model = make_text_model(best_c)
    final_model.fit(
        build_representation(train, train_sem if use_nli else None),
        train["label"],
    )

    threshold_scores = final_model.predict_proba(
        build_representation(threshold, threshold_sem if use_nli else None)
    )[:, 1]
    holdout_scores = final_model.predict_proba(
        build_representation(holdout, holdout_sem if use_nli else None)
    )[:, 1]

    pd.DataFrame(
        {"id": threshold["id"], "score": threshold_scores}
    ).to_csv(outdir / "semantic_verifier_threshold_scores.csv", index=False)
    pd.DataFrame(
        {"id": holdout["id"], "score": holdout_scores}
    ).to_csv(outdir / "semantic_verifier_holdout_scores.csv", index=False)

    baseline = cv_results[cv_results["name"] == "baseline_structural"].iloc[0]
    nli_best = (
        cv_results[cv_results["name"] == "structural_nli"]
        .sort_values(
            ["cv_f1_mean", "cv_f1_std"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    selection = {
        "nli_model": args.model,
        "device": str(device),
        "selected_candidate": str(best["name"]),
        "selected_C": best_c,
        "uses_nli": use_nli,
        "cv_f1_mean": float(best["cv_f1_mean"]),
        "cv_f1_std": float(best["cv_f1_std"]),
        "baseline_cv_f1_mean": float(baseline["cv_f1_mean"]),
        "baseline_cv_f1_std": float(baseline["cv_f1_std"]),
        "semantic_best_cv_f1_mean": float(nli_best["cv_f1_mean"]),
        "semantic_best_cv_f1_std": float(nli_best["cv_f1_std"]),
        "max_claims": args.max_claims,
        "chunks_per_claim": args.chunks_per_claim,
        "note": (
            "Model/candidate selection uses train-only repeated CV. "
            "Threshold and holdout labels are not used for selection."
        ),
    }
    with open(
        outdir / "semantic_selection.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(
        f"Baseline CV F1: {float(baseline['cv_f1_mean']):.4f} "
        f"+/- {float(baseline['cv_f1_std']):.4f}"
    )
    print(
        f"Semantic best: {float(nli_best['cv_f1_mean']):.4f} "
        f"+/- {float(nli_best['cv_f1_std']):.4f} "
        f"(C={float(nli_best['C'])})"
    )
    print(
        f"Selected: {best['name']} (C={best_c}, uses_nli={use_nli})"
    )
    print(
        f"Repeated-CV F1: {float(best['cv_f1_mean']):.4f} "
        f"+/- {float(best['cv_f1_std']):.4f}"
    )
    print(
        "Scores: "
        + str(outdir / "semantic_verifier_threshold_scores.csv")
    )
    print(
        "Scores: "
        + str(outdir / "semantic_verifier_holdout_scores.csv")
    )
    print("Threshold is NOT chosen in this script.")


if __name__ == "__main__":
    main()
