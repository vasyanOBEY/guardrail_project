# ML verifier audit

Date: 2026-10-16

## Current state

Primary ML script: `src/train_verifier_cv.py`.

The repository also still contains two older baseline scripts:

- `src/train_baseline.py`
- `src/run_baseline_splits.py`

Those legacy scripts choose a threshold themselves. They are useful as historical baselines, but they must **not** be used as the final ML workflow because threshold selection belongs to the shared evaluator.

## What is already correct

- Model input is based on `prompt + response`; `explanation` is not used as a model feature.
- Model/hyperparameter selection is performed on train only.
- Threshold and holdout are kept separate from training.
- Raw score files are produced as `id,score`.
- Competition data and generated outputs are ignored by Git so they are not accidentally pushed to the public repository.
- IDs are preserved.

## Audit findings

| Finding | Severity | Fix |
| --- | --- | --- |
| Legacy scripts still select threshold and can be run by mistake | High | Treat them as historical baselines only. Use `train_verifier_cv.py` for participant 3. |
| Prompts are extremely long compared with responses, so full-prompt TF-IDF can be dominated by policy boilerplate | High | Compare bounded context, response-only control, and contextual support features instead of relying only on full prompt. |
| The train split is tiny, so one 3-fold CV split is unstable | High | Use repeated stratified 3-fold CV on train only and report mean + std. |
| Previous verifier compared only full-text word vs word+char TF-IDF | High | Add representation experiments: response, prompt tail + response, and response + prompt-support signature. |
| No explicit cross-split duplicate/leakage check | High | Fail fast on overlapping IDs or exact prompt/response pairs. |
| Experiment output did not record timing or Git commit | Medium | Save per-candidate timing and current commit in result artifacts. |
| No final `scripts/predict.py` / Docker integration yet | High for final submission, not blocking current ML experiments | Participant 4 integrates the selected verifier after model choice stabilizes. |
| Current error-type helper from the shared evaluation is coarse and should not be treated as ground truth taxonomy | Medium | Use it for debugging only; later expand categories for tool/ID/number/policy failures if useful. |

## Data observations used for the first experiment

The fixed splits are small and balanced. The prompt is often tens of thousands of characters while the final response is usually only a few hundred characters. This makes a full bag-of-words representation especially vulnerable to irrelevant policy/context boilerplate.

The first improvement therefore targets **input representation and CV stability**, not threshold tuning.

## Three highest-priority directions

### 1. Context-focused linear verifier — implement first

Compare these train-only CV candidates:

- legacy full-prompt word TF-IDF baseline;
- response-only word TF-IDF control;
- response-only word + character TF-IDF;
- last 8k prompt characters + response;
- response + cheap relational support signature derived from the same prompt.

The relational signature checks whether response numbers, structured IDs and tool-call names are absent from the prompt. It uses no labels or explanations and is available at hidden-test inference time.

Selection rule:

1. highest mean repeated-CV F1;
2. lower F1 standard deviation;
3. higher precision;
4. higher recall.

### 2. Structured verifier features + rules complement

After the first experiment, add more explicit features for:

- tool name availability;
- tool argument mismatch;
- repeated failed actions;
- IDs/numbers/dates/statuses not supported by context;
- policy precondition violations.

Then compare ML-only, rules-only and combined scores on the shared evaluator.

### 3. Semantic support verifier

Only after the lightweight verifier is understood:

- retrieve relevant prompt fragments;
- evaluate open-source NLI / cross-encoder / small instruction model;
- compare gain against latency and memory;
- keep it only if the improvement is robust.

## Important evaluation rule

Participant 3 outputs raw scores only. The shared evaluator chooses the final threshold on the threshold split and reports holdout metrics. Holdout must never be used to choose architecture, features, hyperparameters or threshold.
