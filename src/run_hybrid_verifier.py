from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from train_verifier_cv import (
    RANDOM_STATE,
    _available_tool_names,
    _required_schema_keys,
    _response_tool_calls,
    build_representation,
    make_model,
    model_scores,
    read_table,
    resolve_split,
    validate,
    validate_split_separation,
)


RULE_SCORE_FLOOR = 0.99

TOOL_CALL_MARKER_RE = re.compile(
    r"TOOL_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s*:",
    re.IGNORECASE,
)
TURN_HEADER_RE = re.compile(
    r"^\s*⟦ASSISTANT[^⟧]*⟧\s*",
    re.IGNORECASE,
)

IDENTIFIER_KEYS = {
    "id",
    "zip",
    "zip_code",
    "dob",
    "date",
    "date_of_birth",
    "email",
    "phone_number",
    "flight_number",
    "reservation_id",
    "customer_id",
    "user_id",
    "line_id",
    "bill_id",
    "plan_id",
    "device_id",
    "account_id",
    "order_id",
    "item_id",
    "new_item_id",
    "payment_id",
    "payment_method_id",
}


def _conversation_history(prompt: str) -> str:
    marker = "[AVAILABLE TOOLS]"
    start = prompt.find(marker)
    if start < 0:
        return prompt

    tail = prompt[start + len(marker) :]
    match = re.search(
        r"⟦(?:ASSISTANT|USER|SYSTEM)[^⟧]*⟧",
        tail,
        re.IGNORECASE,
    )
    return tail[match.start() :] if match else tail


def _natural_response_text(response: str) -> str:
    text = TURN_HEADER_RE.sub("", response).strip()
    kept: list[str] = []
    for line in text.splitlines():
        if "→ TOOL_CALL" in line or "← TOOL_RESPONSE" in line:
            continue
        if line.strip():
            kept.append(line.strip())
    return "\n".join(kept).strip()


def _parse_tool_payloads(response: str) -> list[tuple[str, dict | None]]:
    matches = list(TOOL_CALL_MARKER_RE.finditer(response))
    calls: list[tuple[str, dict | None]] = []

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        raw = response[match.end() : end]
        raw = raw.split("← TOOL_RESPONSE", 1)[0].strip()

        payload: dict | None = None
        object_match = re.search(
            r"\{.*?\}(?=\s*(?:→|←|$))",
            raw,
            re.DOTALL,
        )
        if object_match:
            try:
                parsed = json.loads(object_match.group(0))
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = None

        calls.append((match.group(1).lower(), payload))

    return calls


def _flatten_identifier_values(value) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).lower()
            is_identifier_key = (
                key_lower in IDENTIFIER_KEYS
                or key_lower.endswith("_id")
                or key_lower.endswith("_ids")
            )

            if is_identifier_key:
                if isinstance(child, (str, int, float)):
                    found.append((key_lower, str(child)))
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, (str, int, float)):
                            found.append((key_lower, str(item)))

            if isinstance(child, (dict, list)):
                found.extend(_flatten_identifier_values(child))

    elif isinstance(value, list):
        for child in value:
            found.extend(_flatten_identifier_values(child))

    return found


def _value_is_grounded(history: str, value: str) -> bool:
    value = value.strip()
    if not value:
        return True

    pattern = (
        r"(?<![A-Za-z0-9_])"
        + re.escape(value)
        + r"(?![A-Za-z0-9_])"
    )
    return re.search(pattern, history, re.IGNORECASE) is not None


def _asks_for_known_field(history: str, response: str) -> list[str]:
    history_lower = history.lower()
    response_lower = response.lower()

    request_words = (
        "пожалуйста",
        "уточните",
        "предоставьте",
        "необходим",
        "нужен",
        "нужно",
        "provide",
        "need",
        "required",
    )

    specs = [
        (
            "DOB",
            ("дата рождения", "date of birth", "dob"),
            ('"dob"', '"date_of_birth"'),
        ),
        (
            "EMAIL",
            ("email", "электронн"),
            ('"email"',),
        ),
        (
            "USER_ID",
            ("id пользователя", "user id", "user_id"),
            ('"user_id"',),
        ),
        (
            "PHONE",
            ("номер телефона", "phone number"),
            ('"phone_number"',),
        ),
        (
            "ZIP",
            ("почтовый индекс", "zip code", "zip"),
            ('"zip"', '"zip_code"'),
        ),
        (
            "RESERVATION",
            ("номер бронирования", "reservation id"),
            ('"reservation_id"',),
        ),
    ]

    results: list[str] = []
    for name, ask_terms, history_keys in specs:
        asks = (
            any(term in response_lower for term in ask_terms)
            and any(word in response_lower for word in request_words)
        )
        known = any(key in history_lower for key in history_keys)
        if asks and known:
            results.append(name)

    return results


def hard_rule_details(prompt: str, response: str) -> dict[str, int]:
    prompt_lower = prompt.lower()
    history = _conversation_history(prompt)
    calls = _parse_tool_payloads(response)

    available = _available_tool_names(prompt)
    required_by_tool = _required_schema_keys(prompt)
    raw_calls = _response_tool_calls(response)

    unavailable = 0
    previously_failed = 0
    missing_required = 0
    placeholder = 0

    for tool_name, raw_args in raw_calls:
        if available and tool_name not in available:
            unavailable += 1

        failed_pattern = re.compile(
            rf"Tool\s+['\"]{re.escape(tool_name)}['\"]\s+not\s+found",
            re.IGNORECASE,
        )
        if failed_pattern.search(prompt):
            previously_failed += 1

        present_keys = {
            x.lower()
            for x in re.findall(
                r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:',
                raw_args,
            )
        }
        missing_required += len(
            required_by_tool.get(tool_name, set()) - present_keys
        )

        if re.search(
            r"placeholder|\bunknown\b|\bTODO\b|dummy[_ -]?id|fake[_ -]?id",
            raw_args,
            re.IGNORECASE,
        ):
            placeholder += 1

    one_tool_policy = (
        "only make one tool call at a time" in prompt_lower
        or "only make one tool call" in prompt_lower
    )
    exclusive_policy = (
        "if you make a tool call, you should not respond to the user simultaneously"
        in prompt_lower
        or "you cannot do both at the same time" in prompt_lower
    )

    natural_text = _natural_response_text(response)
    multiple_tool_violation = int(
        one_tool_policy and len(calls) > 1
    )
    text_and_tool_violation = int(
        exclusive_policy
        and len(calls) > 0
        and len(natural_text) >= 8
    )

    unsupported_identifier_args = 0
    unchanged_replacements = 0

    for _, payload in calls:
        if not isinstance(payload, dict):
            continue

        for _, value in _flatten_identifier_values(payload):
            if not _value_is_grounded(history, value):
                unsupported_identifier_args += 1

        item_ids = payload.get("item_ids")
        new_item_ids = payload.get("new_item_ids")
        if isinstance(item_ids, list) and isinstance(new_item_ids, list):
            unchanged_replacements += sum(
                str(old) == str(new)
                for old, new in zip(item_ids, new_item_ids)
            )

    known_fields = _asks_for_known_field(
        history,
        response,
    )

    return {
        "unavailable_tool": unavailable,
        "previously_failed_tool": previously_failed,
        "missing_required_tool_key": missing_required,
        "placeholder_tool_arg": placeholder,
        "multiple_tool_policy_violation": multiple_tool_violation,
        "text_and_tool_policy_violation": text_and_tool_violation,
        "unsupported_identifier_arg": unsupported_identifier_args,
        "unchanged_replacement": unchanged_replacements,
        "asks_for_known_field": len(known_fields),
    }


def has_hard_violation(prompt: str, response: str) -> bool:
    details = hard_rule_details(prompt, response)
    return any(value > 0 for value in details.values())


def apply_rule_floor(
    scores,
    df: pd.DataFrame,
) -> np.ndarray:
    output = np.asarray(scores, dtype=float).copy()
    flags = np.asarray(
        [
            has_hard_violation(prompt, response)
            for prompt, response in zip(
                df["prompt"],
                df["response"],
            )
        ],
        dtype=bool,
    )
    output[flags] = np.maximum(
        output[flags],
        RULE_SCORE_FLOOR,
    )
    return output


def evaluate(
    train: pd.DataFrame,
    repeats: int,
) -> tuple[pd.DataFrame, dict]:
    cv = RepeatedStratifiedKFold(
        n_splits=3,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )

    metrics = {
        "baseline_structural": {
            "f1": [],
            "precision": [],
            "recall": [],
        },
        "hybrid_structural_rules": {
            "f1": [],
            "precision": [],
            "recall": [],
        },
    }

    for train_idx, valid_idx in cv.split(
        train,
        train["label"],
    ):
        fold_train = train.iloc[train_idx].copy()
        fold_valid = train.iloc[valid_idx].copy()

        model = make_model("word", 0.5, "logreg")
        model.fit(
            build_representation(
                fold_train,
                "response_support",
            ),
            fold_train["label"],
        )

        base_scores = model_scores(
            model,
            build_representation(
                fold_valid,
                "response_support",
            ),
        )
        hybrid_scores = apply_rule_floor(
            base_scores,
            fold_valid,
        )

        y_true = fold_valid["label"].to_numpy()

        for name, scores in (
            ("baseline_structural", base_scores),
            ("hybrid_structural_rules", hybrid_scores),
        ):
            pred = (np.asarray(scores) >= 0.5).astype(int)
            metrics[name]["f1"].append(
                f1_score(
                    y_true,
                    pred,
                    zero_division=0,
                )
            )
            metrics[name]["precision"].append(
                precision_score(
                    y_true,
                    pred,
                    zero_division=0,
                )
            )
            metrics[name]["recall"].append(
                recall_score(
                    y_true,
                    pred,
                    zero_division=0,
                )
            )

    rows: list[dict] = []
    for name, values in metrics.items():
        f1_values = np.asarray(
            values["f1"],
            dtype=float,
        )
        rows.append(
            {
                "name": name,
                "cv_f1_mean": float(f1_values.mean()),
                "cv_f1_std": float(f1_values.std()),
                "cv_precision_mean": float(
                    np.mean(values["precision"])
                ),
                "cv_recall_mean": float(
                    np.mean(values["recall"])
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
            "Compare the current structural verifier with a conservative "
            "hard-rule overlay for explicit policy/tool violations."
        )
    )
    parser.add_argument("--train", default=None)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--cv-repeats", type=int, default=3)
    args = parser.parse_args()

    train = validate(
        read_table(resolve_split(args.train, "split_train")),
        "train",
    )
    threshold = validate(
        read_table(
            resolve_split(
                args.threshold,
                "split_threshold",
            )
        ),
        "threshold",
    )
    holdout = validate(
        read_table(
            resolve_split(
                args.holdout,
                "split_holdout",
            )
        ),
        "holdout",
    )
    validate_split_separation(
        train,
        threshold,
        holdout,
    )

    print(
        "Cross-validation: structural vs hybrid rules...",
        flush=True,
    )
    results, best = evaluate(
        train,
        repeats=args.cv_repeats,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )
    results.to_csv(
        outdir / "hybrid_cv_results.csv",
        index=False,
    )

    use_rules = (
        str(best["name"])
        == "hybrid_structural_rules"
    )

    model = make_model(
        "word",
        0.5,
        "logreg",
    )
    model.fit(
        build_representation(
            train,
            "response_support",
        ),
        train["label"],
    )

    threshold_scores = model_scores(
        model,
        build_representation(
            threshold,
            "response_support",
        ),
    )
    holdout_scores = model_scores(
        model,
        build_representation(
            holdout,
            "response_support",
        ),
    )

    if use_rules:
        threshold_scores = apply_rule_floor(
            threshold_scores,
            threshold,
        )
        holdout_scores = apply_rule_floor(
            holdout_scores,
            holdout,
        )

    pd.DataFrame(
        {
            "id": threshold["id"],
            "score": threshold_scores,
        }
    ).to_csv(
        outdir
        / "hybrid_verifier_threshold_scores.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "id": holdout["id"],
            "score": holdout_scores,
        }
    ).to_csv(
        outdir
        / "hybrid_verifier_holdout_scores.csv",
        index=False,
    )

    rule_train = pd.DataFrame(
        [
            {
                "id": row["id"],
                "label": int(row["label"]),
                "hard_violation": int(
                    has_hard_violation(
                        row["prompt"],
                        row["response"],
                    )
                ),
                **hard_rule_details(
                    row["prompt"],
                    row["response"],
                ),
            }
            for _, row in train.iterrows()
        ]
    )
    rule_train.to_csv(
        outdir / "hybrid_rule_audit_train.csv",
        index=False,
    )

    summary = {
        "selected_candidate": str(best["name"]),
        "uses_hard_rules": use_rules,
        "rule_score_floor": RULE_SCORE_FLOOR,
        "cv_f1_mean": float(best["cv_f1_mean"]),
        "cv_f1_std": float(best["cv_f1_std"]),
        "cv_precision_mean": float(
            best["cv_precision_mean"]
        ),
        "cv_recall_mean": float(
            best["cv_recall_mean"]
        ),
        "hard_rule_train_positive_count": int(
            rule_train["hard_violation"].sum()
        ),
        "hard_rule_train_false_positive_count": int(
            (
                (rule_train["hard_violation"] == 1)
                & (rule_train["label"] == 0)
            ).sum()
        ),
        "note": (
            "Rules only encode explicit structural violations visible at "
            "inference time. No explanation, threshold label or holdout "
            "label is used for selection."
        ),
    }

    with open(
        outdir / "hybrid_selection.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    baseline = results[
        results["name"] == "baseline_structural"
    ].iloc[0]
    hybrid = results[
        results["name"] == "hybrid_structural_rules"
    ].iloc[0]

    print("Done.")
    print(
        f"Baseline CV F1: "
        f"{baseline['cv_f1_mean']:.4f} "
        f"+/- {baseline['cv_f1_std']:.4f}"
    )
    print(
        f"Hybrid CV F1: "
        f"{hybrid['cv_f1_mean']:.4f} "
        f"+/- {hybrid['cv_f1_std']:.4f}"
    )
    print(
        f"Selected: {summary['selected_candidate']} "
        f"(uses_hard_rules={use_rules})"
    )
    print(
        f"Repeated-CV F1: "
        f"{summary['cv_f1_mean']:.4f} "
        f"+/- {summary['cv_f1_std']:.4f}"
    )
    print(
        "Hard-rule train false positives: "
        f"{summary['hard_rule_train_false_positive_count']}"
    )
    print(
        "Scores: "
        + str(
            outdir
            / "hybrid_verifier_threshold_scores.csv"
        )
    )
    print(
        "Scores: "
        + str(
            outdir
            / "hybrid_verifier_holdout_scores.csv"
        )
    )
    print("Threshold is NOT chosen in this script.")


if __name__ == "__main__":
    main()
