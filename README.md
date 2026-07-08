# Shadow-to-Think

Shadow-to-Think is a token-level large-small model collaboration prototype.

Core idea:

> The student drafts a short continuation. The teacher generates a shadow continuation from the same prefix. The first meaningful trajectory divergence identifies a high-impact token position; the teacher token at that position becomes a candidate intervention. Only verified interventions are distilled into local student-side trigger and selector modules.

This repository contains a complete research prototype:

- student draft generation;
- teacher shadow continuation through either local Transformers or vLLM/OpenAI-compatible serving;
- first meaningful divergence detection;
- candidate set construction: `StudentTopK ∪ {TeacherToken}`;
- counterfactual student rollout verifier;
- feature-based selector baseline;
- local trigger training;
- hidden-state selector training;
- generation-time decoding with shadow mode and student-only local mode;
- smoke tests that run without model downloads.

## Install

```bash
cd shadow-to-think
pip install -e .
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

For an experiment environment with all optional scripts:

```bash
pip install -r requirements.txt
```

`vLLM` is optional. Install it only on the machine that will serve the teacher model.

## Repository Hygiene

The codebase is intended to be committed without local experiment artifacts.
The default `.gitignore` excludes generated outputs such as `runs/`,
`checkpoints/`, model weights, caches, and downloaded benchmark data. Keep only
small hand-written examples in the repository; publish large datasets,
checkpoints, and full evaluation outputs separately.

## Smoke tests

Old MVP selector smoke test:

```bash
python scripts/smoke_test.py
```

v1.0 trigger + hidden selector smoke test:

```bash
python scripts/smoke_test_v1.py
```

These smoke tests use synthetic data and do not download student or teacher
models.

## Dataset format

JSONL, one example per line:

```json
{"id":"ex1","prompt":"Solve: 2 + 2 = ?","answer":"4"}
```

Supported prompt keys: `prompt`, `question`, `input`, `problem`, `query`.

Supported gold keys: `answer`, `gold`, `target`, `label`, `final_answer`.

## 1. Start vLLM teacher serving

On a GPU machine with vLLM installed:

```bash
TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
VLLM_API_KEY=EMPTY \
bash scripts/start_vllm_teacher.sh
```

The project uses vLLM's OpenAI-compatible `/v1/completions` endpoint. The teacher is asked to continue the same text prefix; it does not output logits or explanations.

## 2. Collect shadow correction data

Using local Transformers teacher:

```bash
python scripts/collect_shadow_data.py \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --teacher_backend transformers \
  --dataset_path data/train.jsonl \
  --output_path runs/qwen_shadow/train_shadow.jsonl \
  --draft_len 32 \
  --shadow_len 32 \
  --student_topk 16 \
  --rollout_len 96 \
  --max_samples 1000
```

Using vLLM teacher:

```bash
python scripts/collect_shadow_data.py \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --teacher_backend vllm \
  --teacher_base_url http://localhost:8000/v1 \
  --teacher_api_key EMPTY \
  --dataset_path data/train.jsonl \
  --output_path runs/qwen_shadow/train_shadow.jsonl \
  --draft_len 32 \
  --shadow_len 32 \
  --student_topk 16 \
  --rollout_len 96
```

Important data-quality behavior:

- verification is run on the candidate-conditioned generated suffix, not the full prompt;
- if all counterfactual rollouts fail, the record is kept as `abstain_no_verified_candidate` but does not receive a selector label;
- output is overwritten by default; use `--append` only when you intentionally want to append.

## 3. Train local trigger

The local trigger learns where the student should slow down and check.

```bash
python scripts/train_trigger.py \
  --train_path runs/qwen_shadow/train_shadow.jsonl \
  --output_path checkpoints/qwen_shadow_trigger.pt \
  --epochs 3 \
  --hidden_size 64 \
  --lr 1e-3
```

Trigger labels are produced from offline shadow discovery:

- positive: first meaningful divergence position;
- negative: nearby non-divergent student positions.

At inference time, the trigger uses only student-side features such as entropy, top-1/top-2 margin, token type, and relative position.

## 4. Train hidden-state selector

The hidden selector scores candidates using:

```text
student final hidden state h_t
candidate token embedding e_i
candidate scalar features
```

Train it with the real student model:

```bash
python scripts/train_hidden_selector.py \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_path runs/qwen_shadow/train_shadow.jsonl \
  --output_path checkpoints/qwen_shadow_hidden_selector.pt \
  --epochs 3 \
  --selector_dim 256 \
  --lr 1e-3
```

The smoke test uses `--use_precomputed` so it can run without model downloads.

## 5. Generation-time decoding

### Teacher-assisted shadow decoding

```bash
python scripts/eval_shadow_decode.py \
  --mode shadow \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_backend vllm \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --teacher_base_url http://localhost:8000/v1 \
  --trigger_ckpt checkpoints/qwen_shadow_trigger.pt \
  --selector_ckpt checkpoints/qwen_shadow_hidden_selector.pt \
  --dataset_path data/test.jsonl \
  --output_path runs/qwen_shadow/decode_shadow.jsonl \
  --max_new_tokens 256 \
  --draft_len 32 \
  --shadow_len 32
```

This mode calls the teacher on risky chunks, finds the first meaningful divergence, and uses the hidden selector to choose a corrected token from `StudentTopK ∪ {TeacherToken}`.

### Student-only local decoding

```bash
python scripts/eval_shadow_decode.py \
  --mode local \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --trigger_ckpt checkpoints/qwen_shadow_trigger.pt \
  --selector_ckpt checkpoints/qwen_shadow_hidden_selector.pt \
  --dataset_path data/test.jsonl \
  --output_path runs/qwen_shadow/decode_local.jsonl \
  --max_new_tokens 256
```

This mode does not call the teacher. It uses the distilled trigger and selector to rerank student top-K candidates during generation.

## 6. Optional direct next-token LoRA distillation

```bash
python scripts/train_student_next_token_lora.py \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_path runs/qwen_shadow/train_shadow.jsonl \
  --output_dir checkpoints/qwen_shadow_student_lora \
  --use_lora \
  --epochs 1 \
  --lr 2e-5
```

This script now uses saved `prefix_at_div_ids` when available, avoiding text re-tokenization drift.

## Main files

```text
src/shadow_to_think/teacher_backend.py        # Transformers + vLLM teacher backend
src/shadow_to_think/collector.py             # shadow data collection + bug-fixed verifier labels
src/shadow_to_think/trigger_model.py         # local trigger
src/shadow_to_think/train_trigger.py         # trigger trainer
src/shadow_to_think/hidden_selector_model.py # hidden-state selector
src/shadow_to_think/train_hidden_selector.py # hidden selector trainer
src/shadow_to_think/decode_controller.py     # generation-time decoding controller
scripts/eval_shadow_decode.py                # full decoding entrypoint
```

## What remains research-grade, not production-grade

- first meaningful divergence is still heuristic token alignment;
- v1.0 assumes same tokenizer student/teacher for clean token-level alignment;
- verifier is simple exact/last-number matching;
- local trigger labels are heuristic positives/nearby negatives;
- no RL training is implemented.

The current code is enough to run a complete prototype loop:

```text
collect shadow data → train trigger → train hidden selector → teacher-assisted decoding → student-only local decoding
```
