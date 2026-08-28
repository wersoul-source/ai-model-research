# Topic 10: Building an LLM from Scratch — What Kampun Needs to Know

> **Author:** คำปัน (Kampun) | BOI Family
> **Date:** 2026-08-10 | **Session:** S10-Topic10
> **Bro's Scope:** "ในส่วนที่ตัวเองคิดว่าต้องได้ใช้งานหากว่าเริ่มสร้าง โมเดล" — self-directed research on what's actually needed to build a model
> **Philosophy:** ไม่ใช่แค่รู้ theory — ต้องรู้ cost, tooling, failure modes, และ hands-on practice จริง

---

## Table of Contents
1. [Training Cost Estimation](#1-training-cost-estimation--ต้นทุนจริง)
2. [Data Preparation Pipeline](#2-data-preparation-pipeline)
3. [Cloud GPU Platforms & Pricing](#3-cloud-gpu-platforms--pricing)
4. [Distributed Training Setup](#4-distributed-training-setup)
5. [Hyperparameter Tuning Essentials](#5-hyperparameter-tuning-essentials)
6. [Common Training Failures & Fixes](#6-common-training-failures--fixes)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Model Serving & Deployment](#8-model-serving--deployment)
9. [Fine-Tuning vs Pre-training](#9-fine-tuning-vs-pre-training)
10. [Hands-on Small Projects](#10-hands-on-small-projects)
11. [The Full Picture: From Zero to Production](#11-the-full-picture-from-zero-to-production)

---

## 1. Training Cost Estimation — ต้นทุนจริง

> Source: Galileo AI, DeployBase, SpendArk, YuSMP Group (2025-2026)

### 1.1 The Hard Truth: Pre-training Costs

| Model Size | Pre-training Cost | Time | Hardware |
|-----------|------------------|------|----------|
| 700M (learning project) | **~$50** | Hours | 1× RTX 4090 |
| 7B (Chinchilla-optimal) | **$30K-$50K** | Days-Weeks | 8× A100 spot |
| 7B (production, 1T tokens) | **$100K-$500K** | Weeks | 8× H100 |
| 13B | **$200K-$1M** | Weeks | 8-16× H100 |
| 70B | **$1M-$5M** | Months | 32-64× H100 |
| 70B (Llama 3 class, 15T tokens) | **$1M-$2M** | Months | Multi-node H100 |
| GPT-3 (175B, 2020) | **$4-5M** | Months | 10,000× V100 |
| GPT-4 | **$78M-$100M+** | Months | 25,000× A100 |
| Gemini Ultra 1.0 | **$192M** | Months | TPU v5e |
| DeepSeek V3 (671B) | **$5.6M** (claimed) | 2 months | 2,048× H800 |

**Key insight:** DeepSeek V3's $5.6M figure excluded infrastructure, experimentation, and failed runs. Real cost is higher.

### 1.2 Fine-tuning Costs (The Realistic Path)

| Method | 7B Model | 70B Model | What You Get |
|--------|----------|-----------|-------------|
| **QLoRA** (4-bit) | **$40-$140** | **$50-$500** | Adapter weights, 0.1-3% params |
| **LoRA** (rank 8-64) | **$100-$500** | **$500-$2,000** | Adapter weights |
| **Full SFT** | **$1,000-$3,000** | **$5,000-$20,000** | All weights updated |
| **DPO/RLHF** | **$300-$1,500** | **$2,000-$10,000** | Aligned model |

**Break-even rule:** If you spend >$25K/month on frontier API calls, fine-tuning pays back in 3-6 months. Below $5K/month, just use the API.

### 1.3 Cost Formula

```
Training Cost = GPU-hours × Price-per-GPU-hour

GPU-hours = (Tokens × Model Size) / Throughput

Example: 7B model, 1T tokens on 8× H100
= (1T × 7B) / (8 × ~1,500 tokens/sec/GPU)
= ~70 GPU-days per GPU
= 70/8 = 8.75 days total
= 8.75 × 24 × $2.20 (H100 spot) = ~$462
```

### 1.4 What Drives Cost

1. **Model size** — 10x params = ~10x cost
2. **Dataset size** — 2x tokens = ~2x cost
3. **GPU type** — H100 2-3x faster than A100
4. **Cloud provider** — Hyperscaler 2-4x more expensive than neocloud
5. **Failed runs** — Budget 2-3x for experimentation
6. **Data preparation** — Can exceed compute costs (28x for RLHF annotation)

---

## 2. Data Preparation Pipeline

> Source: Nebius, Latitude, Omar Nahdi, GenAI Institute (2024-2026)

### 2.1 Why Data Is 80% of Quality

> "A simple model trained on excellent, diverse, clean data beats a complex model on noisy data every time." — Amit Ray

**80% of LLM quality comes from data quality.** Spend more time on data than on the model itself.

### 2.2 The Complete Pipeline

```
Raw Text Corpus
     │
     ▼
┌─────────────────────────┐
│ 1. Quality Filtering    │  ← Remove short docs, gibberish, boilerplate
│    - Language detection  │
│    - Min length filter   │
│    - Max repetition      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. Deduplication        │  ← Exact + fuzzy + semantic
│    - MinHash + LSH      │
│    - FAISS for semantic  │
│    - GPU-accelerated CC  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 3. Text Cleaning        │  ← Normalize, fix encoding
│    - Unicode normalization│
│    - ASCII filtering     │
│    - Remove PII          │
│    - Fix formatting      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 4. Tokenization         │  ← Convert text to token IDs
│    - BPE / WordPiece     │
│    - SentencePiece       │
│    - Build vocab         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 5. Sequence Packing     │  ← Pack into fixed-length sequences
│    - Pad to context len  │
│    - Attention masks     │
│    - Minimize waste      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 6. Train/Val Split      │  ← 95/5 or 99/1 split
│    - No data leakage     │
│    - Stratified by domain│
└────────────┬────────────┘
             │
             ▼
Training-Ready Parquet Files
```

### 2.3 Tokenization Methods

| Algorithm | Used By | How It Works | Best For |
|-----------|---------|-------------|----------|
| **BPE** | GPT, GPT-2, RoBERTa | Merge most frequent pairs | General purpose |
| **WordPiece** | BERT, DistilBERT | Mark subwords with "##" | Encoder models |
| **Unigram** | ALBERT, T5, XLNet | Prune vocab by loss | Multilingual |
| **SentencePiece** | Llama, Gemma | Language-agnostic | Any language |

### 2.4 Deduplication at Scale

**Why it matters:** Deduplication prevents memorization, reduces overfitting, and improves diversity.

```
NVIDIA NeMo Curator Pipeline:
1. Exact dedup: Hash-based (fast, catches exact copies)
2. Fuzzy dedup: MinHash + LSH (catches near-duplicates)
3. Semantic dedup: Embedding-based FAISS (catches paraphrases)
4. GPU-accelerated connected components (scales to trillions of tokens)
```

### 2.5 Data Sources for Pre-training

| Dataset | Size | Source | License |
|---------|------|--------|---------|
| **FineWeb** | 15T tokens | Web crawl | Open |
| **FineWeb-Edu** | 1.3T tokens | Educational web | Open |
| **SlimPajama** | 627B tokens | Red Pajama subset | Open |
| **The Stack v2** | 675B tokens | Code (GitHub) | Open |
| **DCLM** | 3T tokens | Datacomp-LM | Open |
| **Red Pajama** | 1.2T tokens | Llama-style web | Open |

---

## 3. Cloud GPU Platforms & Pricing

> Source: CloudMart, GPUCloudCost, GPUPerHour, Spheron (2026)

### 3.1 GPU Pricing Comparison (July 2026)

| GPU | Hyperscaler (AWS/GCP) | Neocloud On-Demand | Neocloud Spot | Best For |
|-----|----------------------|-------------------|---------------|----------|
| **RTX 4090** (24GB) | N/A | $0.74-$1.00 | $0.44-$0.60 | Inference, small fine-tune |
| **A100 40GB** | $3.20-$5.12 | $1.20-$1.80 | $0.80-$1.40 | Training <30B |
| **A100 80GB** | $3.20-$5.12 | $1.64-$2.70 | $1.14-$1.80 | Training, fine-tuning |
| **H100 80GB** | $6.88-$12.29 | $1.99-$4.20 | $1.49-$2.46 | Training >30B |
| **H200 141GB** | $5.50-$8.00 | $2.40-$5.00 | $1.80-$3.40 | Large models, long context |
| **B200** (Blackwell) | $14.24 | $4.00-$6.00 | $2.12+ | Next-gen training |
| **L40S** (48GB) | $2.25-$3.00 | $0.79-$1.04 | $0.54-$0.98 | Inference, medium training |

### 3.2 Platform Recommendations

| Situation | Best Platform | Why |
|-----------|--------------|-----|
| **1-4 GPUs, small team** | Vast.ai, RunPod Community | Cheapest spot rates |
| **8-64 GPUs, serious training** | Lambda, CoreWeave, Crusoe | Reserved capacity, NVLink |
| **256+ GPUs, frontier model** | Hyperscaler SuperPODs | Scale, networking |
| **Budget-constrained** | Vast.ai marketplace | P2P, cheapest possible |
| **No interruption tolerance** | Lambda Reserved, CoreWeave | SLA-backed |
| **AMD GPUs** | vLLM + MI300X | 10-25% cheaper than NVIDIA |

### 3.3 Cost Optimization Rules

1. **Price in tokens-per-dollar, not dollars-per-hour** — A $4.50/hr B200 at 2.3x throughput = cheaper than $2.20/hr H100
2. **Use spot instances for training** — 40-65% savings if you implement checkpoint/resume
3. **Right-size your GPU** — Don't use H100 for 7B model fine-tuning (RTX 4090 is enough)
4. **Batch wisely** — Larger batch = fewer steps = less overhead
5. **Use mixed precision** — BF16/FP8 = 2x throughput vs FP32

---

## 4. Distributed Training Setup

> Source: DeepSpeed, Megatron-LM, FSDP documentation (2025-2026)

### 4.1 Why Distributed Training?

A 7B model needs ~28GB VRAM just for weights (FP16). Add gradients, optimizer states, and activations → 100GB+ for training. No single GPU has that much.

### 4.2 DeepSpeed ZeRO Stages

| Stage | What's Sharded | VRAM Savings | Communication | Use Case |
|-------|---------------|-------------|---------------|----------|
| **ZeRO-1** | Optimizer states | ~4x | Low | Small models |
| **ZeRO-2** | + Gradients | ~8x | Medium | Medium models |
| **ZeRO-3** | + Parameters | ~Nx (N=gpus) | High | Large models |
| **ZeRO-Infinity** | + NVMe offload | Massive | Very High | Very large models |

### 4.3 FSDP (Fully Sharded Data Parallel)

PyTorch's native alternative to DeepSpeed ZeRO-3:

```python
# FSDP setup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

**When to use FSDP vs DeepSpeed:**
- FSDP: PyTorch-native, simpler setup, good for most cases
- DeepSpeed: More optimization options, better for very large models

### 4.4 Multi-Node Training (70B+ Models)

```
Node 1 (8× H100)  ←──NVLink──→  Node 2 (8× H100)
       │                              │
       └──────InfiniBand──────────────┘
                      │
              Node 3 (8× H100)  ←──NVLink──→  Node 4 (8× H100)
```

**Key components:**
- **NVLink:** 900 GB/s within node (GPU↔GPU)
- **InfiniBand:** 400 Gbps between nodes (Node↔Node)
- **Tensor Parallelism:** Split model across GPUs within node
- **Pipeline Parallelism:** Split model layers across nodes
- **Data Parallelism:** Different data on each node

### 4.5 Mixed Precision Training

| Precision | Memory | Speed | Quality | When to Use |
|-----------|--------|-------|---------|-------------|
| **FP32** | 4x | 1x | Baseline | Debugging only |
| **FP16** | 2x | 2x | Good (with loss scaling) | Legacy systems |
| **BF16** | 2x | 2x | Excellent | **Default choice** |
| **FP8** | 1x | 3-4x | Good (with care) | H100/B200 training |

**Critical:** Always use loss scaling with FP16. BF16 doesn't need it (wider dynamic range).

---

## 5. Hyperparameter Tuning Essentials

> Source: DeepChecks, Latitude, Sebastian Raschka (2025-2026)

### 5.1 The Big 5 Hyperparameters

| Hyperparameter | Pre-training Range | Fine-tuning Range | Why It Matters |
|---------------|-------------------|-------------------|----------------|
| **Learning Rate** | 1e-4 to 3e-4 | 1e-5 to 2e-4 | Too high = divergence, too low = slow convergence |
| **Batch Size** | 1M-4M tokens | 16-32 samples | Larger = more stable, needs more VRAM |
| **Warmup Steps** | 2,000-5,000 | 10% of total steps | Prevents early instability |
| **Weight Decay** | 0.1 | 0.01-0.1 | Regularization |
| **Gradient Clipping** | max_norm=1.0 | max_norm=1.0 | Prevents explosion |

### 5.2 Learning Rate Schedule: Cosine Decay

The dominant schedule in 2026 — used by LLaMA, GPT, Gemma, Mistral, Qwen:

```
Learning Rate
    │
    │    ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
    │   ╱                                  ╲
    │  ╱                                    ╲
    │ ╱                                      ╲
    │╱                                        ╲___
    └──────────────────────────────────────────── Step
    warmup   peak    cosine decay to min_lr (10% of peak)
```

**Key parameters:**
- `peak_lr`: 1e-4 to 3e-4 (pre-training), 1e-5 to 2e-4 (fine-tuning)
- `min_lr`: 10% of peak (e.g., 1e-5 if peak is 1e-4)
- `warmup_steps`: 10% of total steps
- `total_steps`: Based on token budget

### 5.3 Batch Size Scaling Rule

**Linear scaling rule:** If you multiply batch size by `k`, multiply peak LR by `k`.

```
Batch size 256 → LR = 1e-4
Batch size 512 → LR = 2e-4
Batch size 1024 → LR = 4e-4
```

**Inverse warmup scaling:** If you double batch size, halve warmup steps (same total samples seen).

### 5.4 LoRA-Specific Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **Learning Rate** | 2e-4 | Higher than full fine-tune |
| **Rank (r)** | 8-64 | Higher = more capacity, more VRAM |
| **Alpha** | 16-64 | Typically 2× rank |
| **Target Modules** | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj | All linear layers |
| **Dropout** | 0.05-0.1 | Regularization |

### 5.5 The Warmup-Stable-Decay Pattern

For LoRA fine-tuning (recommended by Modal engineers):

```
Phase 1 (80% of training): High LR (2e-4) — fast learning
Phase 2 (10% of training): Sharp decay to 10% — settle into solution
Phase 3 (10% of training): Low LR — fine adjustments
```

---

## 6. Common Training Failures & Fixes

> Source: Neel Mishra, Rohan Paul, CLOVA, ICLR 2025 (2024-2026)

### 6.1 Failure Mode Taxonomy

| Failure Mode | Symptom | Severity | Recovery |
|-------------|---------|----------|----------|
| **NaN Loss** | Loss = NaN/Inf, training halts | Critical | Checkpoint rollback |
| **Gradient Explosion** | Grad norms > 1000, params diverge | Critical | Gradient clipping + LR reduction |
| **Loss Spike** | Sudden 2-10x jump in loss | High | May recover or need rollback |
| **Slow Divergence** | Gradual loss increase over 1000s steps | Medium | LR schedule adjustment |
| **Loss Plateau** | Loss stalls at high value | Medium | Data quality check, LR increase |
| **Catastrophic Forgetting** | Model loses pre-trained knowledge | High | Lower LR, include general data |

### 6.2 NaN Loss — The Silent Killer

**Common causes:**
```
1. log(0) → -Inf → NaN on next op
2. 0 / 0 → NaN directly
3. exp(large) → Inf in FP16 (>65504)
4. sqrt(negative) from numerical drift
5. Softmax overflow: exp(88) ≈ 1.65×10³⁸
6. Corrupted data (NaN in input)
7. Uninitialized parameters
```

**FP16 trap:** Dynamic range is only [6e-8, 65504]. Values outside become zero (underflow) or Inf (overflow).

**Fixes:**
- Use BF16 instead of FP16 (wider dynamic range)
- Always use loss scaling with FP16
- Add epsilon to log/division operations
- Check data for NaN before training
- Use `torch.autograd.set_detect_anomaly(True)` for debugging

### 6.3 Gradient Explosion

**Symptoms:** Gradient norms > 1000, loss becomes NaN, parameter values diverge.

**Root cause:** Per-layer Jacobian norms > 1.0 → gradients multiply exponentially across layers.

**Fixes:**
```python
# Gradient clipping (the #1 defense)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Lower learning rate
optimizer = AdamW(model.parameters(), lr=1e-4)  # was 3e-4

# Warmup schedule
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000)

# Pre-norm architecture (Peri-LN)
# Apply normalization BEFORE and AFTER each module
```

### 6.4 Loss Spikes

Google's PaLM documented ~20 loss spikes during 540B training. Each required manual intervention or automatic rollback.

**Detection:**
```python
# Spike detection threshold
if current_loss > max(2 * ema_loss, ema_loss + 4 * std_loss):
    # Potential spike detected
    # Option 1: Skip this batch
    # Option 2: Rollback to last checkpoint
```

**SPAM Optimizer (ICLR 2025):** Spike-Aware Adam with Momentum Reset — detects gradient spikes (up to 1000x larger than normal) and resets momentum to prevent corrupted updates.

### 6.5 Peri-LN: The New Standard

CLOVA's Peri-LN (ICML 2025) applies normalization BEFORE and AFTER each attention/MLP module:

```
Pre-LN:  x → Norm → Attention → + x
Peri-LN: x → Norm → Attention → Norm → + x
```

**Result:** Variance grows linearly (not exponentially), eliminating gradient explosion at scale.

### 6.6 Debugging Checklist

```
□ Is data clean? (No NaN, no duplicates, correct encoding)
□ Is LR appropriate? (Start with 1e-4, adjust based on loss curve)
□ Is warmup sufficient? (10% of total steps minimum)
□ Is gradient clipping enabled? (max_norm=1.0)
□ Is mixed precision correct? (BF16 preferred, FP16 needs loss scaling)
□ Are there corrupted batches? (Monitor per-batch loss)
□ Is the model architecture stable? (Pre-norm, RMSNorm, no bias)
```

---

## 7. Evaluation Metrics

> Source: LLM Trust, TokenCalculator, DataLearner, MachineLearningPlus (2025-2026)

### 7.1 The Three Layers of Evaluation

| Layer | What It Answers | Tools |
|-------|----------------|-------|
| **Benchmarks** | "How does this model rank?" | MMLU, HumanEval, MATH |
| **Metrics** | "How well does it work on MY data?" | Perplexity, BLEU, ROUGE |
| **Judgment** | "Is it actually good?" | Human eval, LLM-as-Judge |

### 7.2 Key Benchmarks (2026)

| Benchmark | What It Measures | Scale | Saturation? |
|-----------|-----------------|-------|-------------|
| **MMLU** | 57 academic subjects | 0-100% | Yes (90%+ for top models) |
| **MMLU-Pro** | Harder version of MMLU | 0-100% | No |
| **HumanEval** | Python code generation | 0-100% | Mostly (99%+ top models) |
| **MATH** | Competition math | 0-100% | No (hard problems remain) |
| **GSM8K** | Grade school math | 0-100% | Yes (95%+ frontier) |
| **GPQA** | Graduate-level science | 0-100% | No (hard, expert-level) |
| **SWE-bench** | Real GitHub issues | 0-100% | No (very hard) |
| **MT-Bench** | Multi-turn conversation | 1-10 | Partially |
| **Arena Elo** | Human preference (Chatbot Arena) | Elo rating | No |

### 7.3 Metrics You Can Compute

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **Perplexity** | How surprised is the model? | Language modeling quality |
| **BLEU** | Text overlap (precision) | Translation |
| **ROUGE** | Text overlap (recall) | Summarization |
| **BERTScore** | Semantic similarity | Any generation task |
| **pass@k** | Does code pass tests? | Code generation |
| **F1 Score** | Balance precision/recall | Classification |

### 7.4 Benchmark Scores: Frontier Models (2026)

| Model | MMLU | HumanEval | MATH | GPQA |
|-------|------|-----------|------|------|
| o3 | 91.4% | 92.8% | 96.7% | — |
| Claude Opus 4.6 | 91.2% | 91.5% | 85.2% | 87.0% |
| Gemini 3 Pro | — | — | — | 91.9% |
| GPT-4o | 88.7% | 87.1% | 76.6% | 53.6% |
| DeepSeek V3 | 88.5% | 89.4% | 75.7% | 52.7% |
| Qwen3-32B | ~82% | ~85% | ~80% | — |

### 7.5 Evaluation Pitfalls

1. **Benchmark contamination** — Models trained on benchmark questions inflate scores
2. **Benchmark saturation** — MMLU no longer differentiates top models (all 88-91%)
3. **Metric gaming** — Optimizing for benchmark ≠ improving real capability
4. **Size comparison trap** — Don't compare 7B model to 70B model on same benchmark
5. **Prompt sensitivity** — Different prompts → different scores

**Rule:** Don't choose a model based on benchmark scores alone. Run your own evaluation on data that matches your use case.

---

## 8. Model Serving & Deployment

> Source: Local-LLM, AICompetence, DeployBase, Hivenet (2026)

### 8.1 The Serving Landscape

```
                    ┌─────────────────┐
                    │  Your Application│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
         │  Local   │   │   API   │   │Production│
         │  Dev     │   │  Server │   │  Cluster │
         └────┬────┘   └────┬────┘   └────┬────┘
              │              │              │
         Ollama         vLLM/SGLang    TensorRT-LLM
         llama.cpp      TGI            Triton
```

### 8.2 Inference Engine Comparison

| Engine | Language | Best For | Throughput | Ease of Use |
|--------|----------|----------|-----------|-------------|
| **Ollama** | Go | Local dev, prototypes | Low | ★★★★★ |
| **llama.cpp** | C++ | CPU, edge, GGUF | Medium | ★★★★ |
| **vLLM** | Python | Production GPU serving | High | ★★★★ |
| **SGLang** | Python | Structured/agentic serving | High | ★★★ |
| **TGI** | Rust | HuggingFace ecosystem | High | ★★★ |
| **TensorRT-LLM** | C++/Python | Max NVIDIA performance | Highest | ★★ |

### 8.3 Decision Matrix

| Your Situation | Use |
|---------------|-----|
| "I just need to run a model locally" | **Ollama** |
| "I need efficient CPU/edge inference" | **llama.cpp** |
| "I need a production GPU API server" | **vLLM** |
| "I need structured/agentic serving" | **SGLang** |
| "NVIDIA only, max throughput, stable workload" | **TensorRT-LLM** |
| "Already on HuggingFace ecosystem" | **TGI** |

### 8.4 vLLM: The Default Choice

vLLM has become the standard for production LLM serving in 2026:

**Key features:**
- **PagedAttention:** Manages KV cache like virtual memory → near-zero waste
- **Continuous batching:** Requests join batch automatically → GPU always busy
- **OpenAI-compatible API:** Drop-in replacement for OpenAI endpoints
- **Multi-GPU:** Tensor/pipeline/data parallelism
- **Quantization:** AWQ, GPTQ, FP8, GGUF support

```bash
# Quick start
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2

# Docker
docker run --gpus all -p 8000:8000 vllm/vllm-openai \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### 8.5 Ollama for Local Development

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Run models
ollama run llama3.1:8b
ollama run qwen3:14b
ollama run gemma4:latest

# API
curl http://localhost:11434/api/generate \
  -d '{"model": "llama3.1:8b", "prompt": "Hello!"}'
```

### 8.6 Serving Cost Comparison

| Setup | Monthly Cost | Throughput | Users |
|-------|-------------|-----------|-------|
| Ollama (1× RTX 4090) | ~$500 (cloud) | ~50 tok/s | 1-5 |
| vLLM (1× A100) | ~$1,200 | ~200 tok/s | 10-50 |
| vLLM (4× H100) | ~$8,000 | ~1,000 tok/s | 100-500 |
| TensorRT-LLM (8× H100) | ~$16,000 | ~2,000 tok/s | 500+ |

---

## 9. Fine-Tuning vs Pre-training

> Source: MLJourney, CoreWeave, Stratagem Systems, is4.ai (2025-2026)

### 9.1 The Fundamental Difference

| Aspect | Pre-training | Fine-tuning |
|--------|-------------|-------------|
| **Purpose** | Learn general language understanding | Adapt to specific task/domain |
| **Data** | Trillions of tokens (web-scale) | Thousands to millions of examples |
| **Compute** | 1000s of GPUs, months | 1-4 GPUs, hours to days |
| **Cost** | $10K-$100M+ | $40-$50K |
| **Output** | Base model (general) | Specialized model |
| **When** | Only if no suitable base model exists | **99% of the time** |

### 9.2 When to Pre-train from Scratch

**Only pre-train when:**
1. No adequate base model exists for your language/domain
2. You need fundamentally new capabilities
3. You have $100K+ budget and months of time
4. You're building a foundation model for many downstream tasks

**Examples:**
- Underrepresented languages (not enough pre-trained models)
- Highly specialized scientific domains
- Building a new base model family

### 9.3 When to Fine-tune (Almost Always)

**Fine-tune when:**
1. A good base model exists (Llama, Qwen, Gemma, Mistral)
2. You need task-specific behavior
3. You need consistent style/tone
4. You have domain-specific data
5. Budget is limited

**Methods:**
- **QLoRA:** Cheapest, 4-bit quantized base, train adapters
- **LoRA:** Good balance, train low-rank matrices
- **Full SFT:** Most expensive, update all weights
- **DPO/RLHF:** For alignment (safety, helpfulness)

### 9.4 RAG vs Fine-tuning vs Long Context

| Approach | Best For | Cost | Maintenance |
|----------|---------|------|-------------|
| **RAG** | Dynamic, frequently updated info | Low setup, medium ongoing | Easy (update vector DB) |
| **Fine-tuning** | Consistent behavior, style/tone | Medium setup, low ongoing | Hard (retrain) |
| **Long Context** | Small doc sets, low query volume | Zero setup, high per-query | None |
| **Hybrid** | Production systems with budget | High | Complex |

### 9.5 The Decision Tree

```
Do you need AI for a specific task?
├── YES → Does a good base model exist?
│   ├── YES → Can prompting solve it?
│   │   ├── YES → Just use prompting (free)
│   │   └── NO → Can RAG solve it?
│   │       ├── YES → Use RAG ($500-$2K/month)
│   │       └── NO → Fine-tune ($40-$50K)
│   └── NO → Do you have $100K+ and months?
│       ├── YES → Pre-train from scratch
│       └── NO → Find a closer base model
└── NO → You don't need AI for this
```

---

## 10. Hands-on Small Projects

> Source: Karpathy, FareedKhan, CodersEra, DataCamp, Amit Ray (2025-2026)

### 10.1 The 4-Week Learning Roadmap

**Week 1: Foundations (Tokenization + Data)**
- Implement CharTokenizer from scratch
- Run tiktoken demos — understand how GPT-2 sees text
- Implement BPE from scratch (the merge algorithm)
- Build data pipeline: clean → filter → tokenize → batch

**Week 2: Build NanoGPT**
- Implement Transformer from scratch in PyTorch
- Multi-head self-attention
- Feed-forward network with GELU
- Positional embeddings
- Train on Shakespeare — watch loss fall

**Week 3: Fine-tune Existing Models**
- Fine-tune DistilGPT-2 on custom domain with HuggingFace
- Apply LoRA for larger models on limited hardware
- Evaluate with perplexity and human inspection

**Week 4: Deploy and Iterate**
- Install Ollama, run Llama 3B locally
- Build a simple chat interface
- Experiment with different prompts
- Measure latency and throughput

### 10.2 Recommended Projects by Level

**Beginner (No GPU Required):**

| Project | What You Learn | Time |
|---------|---------------|------|
| Run Ollama + chat with local model | How LLMs work in practice | 1 hour |
| Build a RAG system with LangChain | Retrieval + generation pipeline | 4 hours |
| Fine-tune a model with OpenAI API | How fine-tuning works | 2 hours |
| Build a language tutor with Langflow | Visual AI workflow design | 3 hours |

**Intermediate (1× GPU):**

| Project | What You Learn | Time |
|---------|---------------|------|
| Train NanoGPT on Shakespeare | Transformer internals | 4 hours |
| Fine-tune Llama 3B with LoRA | Parameter-efficient training | 2 hours |
| Build a code generation model | Domain-specific fine-tuning | 8 hours |
| Implement BPE tokenizer from scratch | Tokenization fundamentals | 3 hours |

**Advanced (Multi-GPU):**

| Project | What You Learn | Time |
|---------|---------------|------|
| Train 700M model from scratch | Full pre-training pipeline | 1-2 days |
| Fine-tune 13B with QLoRA on A100 | Large model fine-tuning | 4-8 hours |
| Build production serving with vLLM | Deployment at scale | 1 day |
| Train a reasoning model with GRPO | Alignment techniques | 2-3 days |

### 10.3 The $50 Learning Project

Train a 700M parameter model to GPT-2 quality:

```bash
# Using Karpathy's nanochat
git clone https://github.com/karpathy/nanochat
cd nanochat
pip install -r requirements.txt

# Train on single GPU (3 hours on 8× H100)
bash speedrun.sh

# Or smaller scale
python train.py --model_size 700M --dataset openwebtext
```

**What you get:** A deployable model + complete understanding of the training stack.

### 10.4 Key Resources

| Resource | What It Teaches | Link |
|----------|----------------|------|
| **Karpathy's nanoGPT** | Transformer from scratch | github.com/karpathy/nanoGPT |
| **Karpathy's nanochat** | Train + deploy a chat model | github.com/karpathy/nanochat |
| **train-llm-from-scratch** | End-to-end training pipeline | github.com/FareedKhan-dev/train-llm-from-scratch |
| **LLM From Scratch (workshop)** | Hands-on, no ML experience needed | github.com/cyyself/tiny-llm |
| **HuggingFace Course** | Fine-tuning with PEFT | huggingface.co/learn |

---

## 11. The Full Picture: From Zero to Production

### 11.1 Kampun's Decision Framework

```
STEP 1: Can I solve this with prompting?
├── YES → Do that. It's free.
└── NO ↓

STEP 2: Can I solve this with RAG?
├── YES → Build RAG system. $500-$2K/month.
└── NO ↓

STEP 3: Should I fine-tune?
├── YES → Choose method:
│   ├── QLoRA ($40-$500) — most cost-effective
│   ├── LoRA ($100-$2K) — good balance
│   └── Full SFT ($1K-$20K) — when you need it
└── NO ↓

STEP 4: Should I pre-train?
├── Do you have $100K+? → YES → Pre-train
└── NO → Find a better base model
```

### 11.2 The Complete Model Building Stack

```
┌─────────────────────────────────────────┐
│            APPLICATION LAYER            │
│  Chat interface, API, mobile app        │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│            SERVING LAYER                │
│  vLLM, Ollama, TensorRT-LLM            │
│  Continuous batching, KV cache         │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│            MODEL LAYER                  │
│  Pre-trained weights + LoRA adapters    │
│  Quantized (GGUF/AWQ/GPTQ)             │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│            TRAINING LAYER               │
│  PyTorch + DeepSpeed/FSDP               │
│  Mixed precision, gradient clipping     │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│            DATA LAYER                   │
│  Clean, deduplicated, tokenized         │
│  Parquet format, streaming              │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│            INFRASTRUCTURE LAYER         │
│  Cloud GPUs (Lambda, RunPod, CoreWeave) │
│  Storage (S3, distributed FS)           │
└─────────────────────────────────────────┘
```

### 11.3 Kampun's Top 10 Takeaways

1. **Data is 80% of quality** — spend more time on data than on model architecture
2. **Fine-tune, don't pre-train** — 99% of teams should fine-tune existing models
3. **LoRA/QLoRA changes everything** — fine-tune 70B on a single A100
4. **Cosine decay is the standard** — use it for learning rate scheduling
5. **BF16 > FP16** — wider dynamic range, no loss scaling needed
6. **Gradient clipping saves lives** — max_norm=1.0, always
7. **vLLM for production, Ollama for dev** — the 2026 serving stack
8. **Price in tokens-per-dollar** — not dollars-per-hour
9. **Checkpoint everything** — training can fail at any point
10. **Start small, scale up** — 13M first, then scale to GPU limit

---

## References

1. Galileo AI. "How Much Does LLM Training Cost?" galileo.ai/blog/llm-model-training-cost. 2026.
2. DeployBase. "AI Training Cost." deploybase.ai/articles/ai-training-cost. Aug 2025.
3. SpendArk. "How Much Does It Cost to Train an LLM in 2026?" spendark.com/blog/cost-to-train-llm-2026. Jun 2026.
4. YuSMP Group. "LLM Fine-Tuning Cost Benchmark 2026." yusmpgroup.com/blog/llm-fine-tuning-cost-benchmark-2026. Jul 2026.
5. Nebius. "Data preparation for LLMs." nebius.com/blog/posts/data-preparation. Jun 2024.
6. Latitude. "Ultimate Guide to Preprocessing Pipelines for LLMs." latitude.so/blog. Mar 2025.
7. Omar Nahdi. "Building an LLM Pretraining Data Pipeline." omarnahdi.dev. Mar 2026.
8. CloudMart. "GPU Cloud Pricing Comparison 2026." cloudmart.dev/gpu. 2026.
9. GPUCloudCost. "GPU pricing in 2026." gpucloudcost.com. Jul 2026.
10. DeepSpeed. "ZeRO & DeepSpeed." deepspeed.ai. 2025.
11. DeepChecks. "Hyperparameter Optimization For LLMs." deepchecks.com. Oct 2025.
12. Latitude. "Fine-Tuning LLMs: Hyperparameter Best Practices." latitude.so/blog. Feb 2026.
13. Neel Mishra. "Debugging Training: NaN, Divergence." neelmishra.github.io. 2025.
14. CLOVA. "Stable training: Preventing divergence with Peri-LN." clova.ai/en/tech-blog. Jun 2025.
15. ICLR 2025. "SPAM: Spike-Aware Adam with Momentum Reset." iclr.cc/virtual/2025. 2025.
16. LLM Trust. "Understanding LLM Benchmarks." llmtrust.com/blog. Mar 2026.
17. Local-LLM. "vLLM vs TGI vs TensorRT-LLM." local-llm.net. Jun 2026.
18. AICompetence. "LLM Inference Stack Comparison." aicompetence.org. Jul 2026.
19. MLJourney. "LLM Training vs Fine-Tuning." mljourney.com. Dec 2025.
20. Karpathy. "nanochat." github.com/karpathy/nanochat. 2025.
21. FareedKhan. "train-llm-from-scratch." github.com/FareedKhan-dev. 2025.
22. CodersEra. "Self-Training a Small LLM From Scratch." codersera.com. May 2026.

---

**All reports in this series:**
- Topic 1: Software & Hardware Requirements — `AI-Model-Report-Topic1-Software-Hardware.md`
- Topic 2: Model Architecture Components — `AI-Model-Report-Topic2-Model-Architecture.md`
- Topic 3: Frameworks, Research Papers, Steps — `AI-Model-Report-Topic3-Frameworks-Papers-Steps.md`
- Topic 4: Foundation & Supplementary Knowledge — `AI-Model-Report-Topic4-Knowledge-Requirements.md`
- Topic 5: Model Types, Formats, Performance — `AI-Model-Report-Topic5-Model-Types-Formats.md`
- Topic 6: Model Computation Methods — `AI-Model-Report-Topic6-Computation-Methods.md`
- Topic 7: Model Assembly Principles — `AI-Model-Report-Topic7-Assembly-Principles.md`
- Topic 8: Production Pipeline — `AI-Model-Report-Topic8-Production-Pipeline.md`
- Topic 9: Model Architecture Specs — `AI-Model-Report-Topic9-Model-Architecture-Specs.md`
- **Topic 10: Building an LLM — What Kampun Needs to Know — `AI-Model-Report-Topic10-Building-LLM-Guide.md`** (This Report)
