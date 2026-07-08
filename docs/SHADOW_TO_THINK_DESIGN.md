# Shadow-to-Think v1.0 完整工程设计文档

## 0. 一句话定义

Shadow-to-Think 是一个 token 级大小模型协同与蒸馏框架：小模型先生成短 draft，大模型从同一 prefix 做 shadow continuation；两条轨迹的第一个有意义分叉点被视为高影响 token 位置，大模型在该位置自然输出的 token 被作为候选干预；系统只接受经过 rollout/verifier 支持的干预，并把有效干预蒸馏成小模型自己的 local trigger 和 hidden-state selector。

---

## 1. 动机

### 1.1 为什么不是 RelayLLM 式接管

RelayLLM 让小模型通过 `<call>n</call>` 主动求助，大模型接管后续 n 个 token。这个机制推理时不需要逐 token teacher-student KL，成本比较友好。但它的监督粒度较粗：系统知道某个位置需要 help，却不精确知道哪个 token 开始导致轨迹偏离。因此它更像 span-level delegation，不适合作为 token-level 蒸馏信号。

### 1.2 为什么不是原始 Select-to-Think

Select-to-Think 的优点是粒度细：在关键 token 处，大模型从小模型 top-K 中选一个更合理的 token。但它的问题是关键 token 发现成本高。如果用 teacher-student token distribution divergence / KL 找关键点，就需要大量 teacher forward。并且原始 S2T 默认正确 token 已经在 student top-K 中，如果 student top-K 全是错误方向，就难以修复。

### 1.3 为什么使用 shadow continuation

大模型最自然的接口不是输出 logits，也不是做 judge，而是从 prefix 继续生成。Shadow-to-Think 让大模型只做一次自然续写，得到两个信息：

1. location：teacher shadow 与 student draft 第一次发生有意义分叉的位置；
2. intervention：teacher 在该位置自然输出的 token。

因此，大模型的一次 normal generation 同时提供关键 token 定位和候选修正。

---

## 2. 核心流程

```text
prompt / prefix
   ↓
student draft: d_1 ... d_L
   ↓
teacher shadow continuation: t_1 ... t_L
   ↓
first meaningful divergence i
   ↓
student token d_i, teacher token t_i
   ↓
CandidateSet = StudentTopK_i ∪ {TeacherToken_i}
   ↓
counterfactual student rollout for each candidate
   ↓
verifier selects best candidate or abstains
   ↓
train local trigger + hidden-state selector
```

### 2.1 Student Draft

小模型从当前 prefix 生成一个短 chunk。v1 默认 `draft_len=32`。

### 2.2 Teacher Shadow Continuation

teacher 从同一 prefix 生成一个短 continuation。teacher 不看 student draft，不输出解释，不输出 logits。

### 2.3 First Meaningful Divergence

比较 student draft 与 teacher shadow。v1 使用轻量规则过滤：

- 跳过标点、空格、纯 stop-like token；
- 跳过少量近义连接词，如 `therefore/thus/hence/so`；
- 保留真正可能改变轨迹的 token mismatch。

### 2.4 Candidate Construction

在分叉 prefix 上重新计算 student next-token top-K：

```text
C = StudentTopK ∪ {TeacherToken}
```

这比 S2T 更宽，因为 teacher token 不必出现在 student top-K；又比 RelayLLM 更克制，因为 teacher 只提供 token-level proposal。

### 2.5 Verified Label

对每个候选 token 做 student rollout：

```text
prefix_at_div + candidate_token → student continues
```

verifier 只检查 candidate-conditioned generated suffix，不检查完整 prompt，避免 prompt 中数字造成假阳性。如果所有候选都失败，样本会被标记为 `abstain_no_verified_candidate`，不会强行生成 selector label。

---

## 3. 已实现模块

### 3.1 vLLM Teacher Serving

文件：

```text
scripts/start_vllm_teacher.sh
src/shadow_to_think/teacher_backend.py
```

启动：

```bash
TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8000 VLLM_API_KEY=EMPTY bash scripts/start_vllm_teacher.sh
```

调用：

```bash
python scripts/collect_shadow_data.py \
  --teacher_backend vllm \
  --teacher_base_url http://localhost:8000/v1 \
  --teacher_api_key EMPTY
```

实现方式：使用 vLLM OpenAI-compatible `/v1/completions` endpoint。teacher 返回 continuation text，然后用 student tokenizer 编码为 token ids。

### 3.2 Local Trigger Training

文件：

```text
src/shadow_to_think/trigger_model.py
src/shadow_to_think/train_trigger.py
scripts/train_trigger.py
```

训练目标：

```text
student-side features → risky token position?
```

特征包括：

- student entropy；
- top-1/top-2 margin；
- top-1 logprob/prob；
- relative position；
- token text length；
- digit / constraint / reasoning connector flags。

训练标签：

- positive：first meaningful divergence position；
- negative：附近非分叉位置。

### 3.3 Hidden-State Selector

文件：

```text
src/shadow_to_think/hidden_selector_model.py
src/shadow_to_think/train_hidden_selector.py
scripts/train_hidden_selector.py
```

模型输入：

```text
prefix hidden state h_t
candidate token embedding e_i
candidate scalar features f_i
```

模型输出：

```text
score(candidate_i)
```

训练目标：candidate-level cross entropy，label 来自 verified rollout。

### 3.4 Generation-Time Selector Decoding

文件：

```text
src/shadow_to_think/decode_controller.py
scripts/eval_shadow_decode.py
```

支持两种模式：

#### shadow mode

```text
student chunk
→ optional local trigger
→ teacher shadow continuation
→ first meaningful divergence
→ candidate set
→ hidden selector chooses token
→ inject selected token
→ student continues
```

#### local mode

```text
student next-token stats
→ local trigger
→ hidden selector reranks student top-K
→ selected token
→ student continues
```

local mode 不调用 teacher，用于验证蒸馏后的 student-only 能力。

---

## 4. 完整命令流程

### 4.1 安装

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### 4.2 smoke test

```bash
python scripts/smoke_test.py
python scripts/smoke_test_v1.py
```

### 4.3 启动 vLLM teacher

```bash
TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8000 VLLM_API_KEY=EMPTY bash scripts/start_vllm_teacher.sh
```

### 4.4 收集 shadow data

```bash
python scripts/collect_shadow_data.py \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --teacher_backend vllm \
  --teacher_base_url http://localhost:8000/v1 \
  --dataset_path data/train.jsonl \
  --output_path runs/qwen_shadow/train_shadow.jsonl
```

### 4.5 训练 trigger

```bash
python scripts/train_trigger.py \
  --train_path runs/qwen_shadow/train_shadow.jsonl \
  --output_path checkpoints/qwen_shadow_trigger.pt
```

### 4.6 训练 hidden selector

```bash
python scripts/train_hidden_selector.py \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_path runs/qwen_shadow/train_shadow.jsonl \
  --output_path checkpoints/qwen_shadow_hidden_selector.pt
```

### 4.7 teacher-assisted decoding

```bash
python scripts/eval_shadow_decode.py \
  --mode shadow \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_backend vllm \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --trigger_ckpt checkpoints/qwen_shadow_trigger.pt \
  --selector_ckpt checkpoints/qwen_shadow_hidden_selector.pt \
  --dataset_path data/test.jsonl \
  --output_path runs/qwen_shadow/decode_shadow.jsonl
```

### 4.8 student-only decoding

```bash
python scripts/eval_shadow_decode.py \
  --mode local \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --trigger_ckpt checkpoints/qwen_shadow_trigger.pt \
  --selector_ckpt checkpoints/qwen_shadow_hidden_selector.pt \
  --dataset_path data/test.jsonl \
  --output_path runs/qwen_shadow/decode_local.jsonl
```

---

## 5. 关键 bug 修复

### 5.1 verifier 不再验证完整 prompt

旧版会在 prompt + generation 上抽答案，prompt 中的数字可能导致假阳性。v1 改为只验证候选 token 之后的 generated suffix。

### 5.2 全候选失败时不再强行产生 label

旧版如果所有 rollout score 都是 0，也会选择第一个候选作为 label。v1 改为 `abstain_no_verified_candidate`，不参与 selector training。

### 5.3 输出默认 overwrite

旧版重复运行 collection 会悄悄 append。v1 默认覆盖，只有传 `--append` 才追加。

### 5.4 LoRA 蒸馏优先使用 token ids

旧版用 `prefix_at_div_text` 重新 tokenize，可能出现 token drift。v1 保存并优先使用 `prefix_at_div_ids`。

---

## 6. 当前边界

这个工程是完整 v1.0 原型，但还不是 production system：

1. 默认同 tokenizer student/teacher；
2. first meaningful divergence 仍是 heuristic；
3. verifier 仍是简单 exact/last-number；
4. local trigger 的负样本是附近位置，未来可加入 hard negatives；
5. 没有做 RL 或多 teacher；
6. vLLM 路径依赖远端 completion text 再编码，严格 token 对齐不如本地 logits 直接。

---

## 7. 推荐实验指标

性能：

```text
student greedy accuracy
teacher-assisted shadow accuracy
student-only local accuracy
```

成本：

```text
teacher calls / question
teacher generated tokens / question
interventions / question
```

蒸馏效果：

```text
student-only local - student greedy
```

可信性：

```text
accepted teacher intervention success rate
rejected teacher intervention success rate
abstain rate
harmful teacher-token rate
```

---

## 8. 方法定位

Shadow-to-Think v1.0 可以视为：

```text
S2T 的 token-level selection
+
RelayLLM 的低频 teacher invocation 思想
+
shadow continuation 的自然 key-token discovery
+
verified corrective distillation
```

它的核心不同点是：

```text
teacher generation 同时提供 location 和 intervention，
verified rollout 决定是否蒸馏，
local trigger/selector 负责最终 student-only 推理。
```
