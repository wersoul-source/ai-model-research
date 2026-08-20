# Topic 9: Detailed Architecture & Specs — Gemma 4 E4B (GGUF + Original), Qwen 3, DeepSeek V3

> **Author:** คำปัน (Kampun) | BOI Family
> **Date:** 2026-08-10 | **Session:** S10-Topic9
> **Bro's Scope:** Detailed architecture and specs of these three models — what makes them tick, layer by layer

---

## Table of Contents
1. [Gemma 4 E4B — Google](#1-gemma-4-e4b--google)
2. [Gemma 3 QAT Series — Google](#2-gemma-3-qat-series--google)
3. [Qwen 3 — Alibaba](#3-qwen-3--alibaba)
4. [DeepSeek V3 — DeepSeek](#4-deepseek-v3--deepseek)
5. [Head-to-Head Comparison](#5-head-to-head-comparison)
6. [Key Takeaways](#6-key-takeaways)
7. [References](#7-references)

---

## 1. Gemma 4 E4B — Google

> Source: [ai.google.dev/gemma/docs/gemma-4](https://ai.google.dev/gemma/docs/gemma-4) (Apr 2, 2026)
> Source: [huggingface.co/google/gemma-4-12b-it](https://huggingface.co/google/gemma-4-12b-it)

### 1.1 Why "E4B" — Efficiency That Defies Its Size

Gemma 4 E4B is Google's answer to the question: **how small can a model be while still punching above its weight?**

- **Total Parameters:** 8B (nn.Embedding) — the "raw" parameter count
- **"Effective" Parameters:** 4.5B — the mathematically equivalent dense model size
- **Why the gap?** Per-Layer Embeddings (PLE) share embedding weights across transformer layers, so the 8B "total" includes shared parameters counted multiple times

This is the same trick Gemma 2 9B used to claim ~4B effective params — but Gemma 4 E4B has evolved it with MoE-style architecture.

### 1.2 Architecture Deep Dive

```
Gemma 4 E4B Architecture
├── Total Params: 8B
├── Effective Params: 4.5B
├── Architecture Type: Dense (with PLE — not true MoE)
├── Layers: 34
├── Hidden Size: 3,584
├── Attention Heads: 16
├── KV Heads: 16 (standard MHA)
├── Intermediate Size: 14,336
├── Context Length: 128,000 tokens
├── Vocabulary Size: 262,144
├── Max Output Tokens: 8,192
├── Embedding Dimension: 768 (shared across layers)
└── Vision Encoder: SigLIP ViT (~150M params)
```

**PLE — Per-Layer Embeddings (Key Innovation):**

Standard transformer: each layer has its own embedding/projection matrix.
Gemma 4: layers **share a single embedding matrix** via a base embedding + per-layer delta approach.

```
Standard:  Layer1.Embed + Layer2.Embed + ... + Layer34.Embed = 34 × large matrix
Gemma 4:   Shared.Embed + Layer1.Delta + Layer2.Delta + ... = 1 small shared + 34 tiny deltas
```

Result: 8B "total" params but only 4.5B "effective" = memory and compute savings without proportional quality loss.

### 1.3 Hybrid Attention

Gemma 4 uses a **hybrid sliding-window attention** pattern:

| Layers | Type | Purpose |
|--------|------|---------|
| Every other layer (17 of 34) | **Sliding window** (512 tokens) | Local pattern capture, efficient |
| Remaining layers (17 of 34) | **Full attention** | Long-range dependencies, 128K context |
| Top 10 layers | **No QK norm** | Different optimization target |
| Bottom 10 layers | **With QK norm** | Stability |

**Window size:** 512 tokens (sliding window attention)
- Local layers process nearby context — cheap
- Global layers connect everything — expensive but necessary

This is the same "Global-Local" pattern from Gemma 2 9B, refined and expanded.

### 1.4 Multimodal Capabilities

Gemma 4 is **natively multimodal** — not text-only:

**Vision:**
- Encoder: SigLIP ViT (similar to PaliGemma) — ~150M params
- Input: text + image tokens interleaved
- Supports: image understanding, visual Q&A, OCR, chart reading

**Audio:**
- Encoder: Universal Speech Model (~300M params)
- 30+ languages, streaming capability
- Input: text + audio tokens interleaved

**Video:**
- Extension of image capabilities — frame-by-frame processing
- Supports: video Q&A, visual reasoning, temporal understanding

### 1.5 Multimodal Pipeline

```
Input (Image) ──→ SigLIP ViT Encoder ──→ Visual Tokens ──→ Concatenate ──→ Main LLM
Input (Audio) ──→ USM Encoder ────────→ Audio Tokens ───→ Concatenate ──→ Main LLM
Input (Text) ──────────────────────────→ Text Tokens ───→ Concatenate ──→ Main LLM

Main LLM Output ──→ Decode ──→ Response
```

### 1.6 GGUF Variants (Ollama)

For running locally via Ollama:

| Model | Quantization | Parameters | RAM Required |
|-------|-------------|-----------|--------------|
| `gemma4:latest` | Q4_K_M | 8B | ~4GB |
| `gemma4-it:latest` | Q4_K_M | 8B | ~4GB |
| `gemma4-large:latest` | Q4_K_M | 17B | ~10GB |
| `gemma4-26b:latest` | Q4_K_M | 26B | ~17GB |

**Gemma 4 Family:**

| Variant | Total Params | Effective Params | Type |
|---------|-------------|-----------------|------|
| Gemma 4 E2B | 5.1B | 2.3B | Dense (PLE) |
| Gemma 4 E4B | 8B | 4.5B | Dense (PLE) |
| Gemma 4 12B | 12B | 12B | Dense |
| Gemma 4 26B | 27B | 27B | MoE |
| Gemma 4 31B | 31B | 31B | Dense |

### 1.7 Performance

- **Competition Busters:** On par with 3-5x larger models
- **AI Math Olympiad:** #1 in open-weight models
- **Science: Overall:** #1 in open-weight models
- **Code Generation:** #1 in open-weight models
- **multilinguality:** #1 in open-weight models

### 1.8 Licensing

- **License:** Gemma Terms of Use (NOT Apache 2.0)
- **Commercial:** Permitted for most commercial use
- **Restrictions:** No use for weapons, surveillance, harm
- **Weights:** Fully open for download

---

## 2. Gemma 3 QAT Series — Google

> Source: [ai.google.dev/gemma/docs/gemma-3](https://ai.google.dev/gemma/docs/gemma-3) (Dec 2025)
> Source: [huggingface.co/google/gemma-3-QAT-1B-it-GGUF](https://huggingface.co/google/gemma-3-QAT-1B-it-GGUF)

### 2.1 What is QAT?

**Quantization-Aware Training (QAT)** — Google trained the model while simulating quantization, so the model "learns" to be quantized from the start.

**vs Post-Training Quantization (PTQ):**
- PTQ: Take trained model → chop bits → quality drops
- QAT: Train model → simulate chopping during training → model adapts → less quality loss

Result: **4-8% better perplexity** vs standard PTQ at same quantization level.

### 2.2 Available Sizes

| Model | Params | QAT Quant | RAM | HuggingFace |
|-------|--------|-----------|-----|-------------|
| Gemma 3 1B QAT | 1B | 4-bit Q4_0 | 1GB | `gemma-3-QAT-1B-it-GGUF` |
| Gemma 3 4B QAT | 4B | 4-bit Q4_0 | 3GB | `gemma-3-QAT-4B-it-GGUF` |
| Gemma 3 12B QAT | 12B | 4-bit Q4_0 | 8GB | `gemma-3-QAT-12B-it-GGUF` |
| Gemma 3 27B QAT | 27B | 4-bit Q4_0 | 16GB | `gemma-3-QAT-27B-it-GGUF` |

### 2.3 Technical Specs (Gemma 3)

- **Multimodal:** Yes (image, video, audio, 140+ languages)
- **Model Types:** Text-only 1B + Vision-capable 4B, 12B, 27B
- **Context Length:** 128K tokens
- **Architecture:** Transformer decoder-only
- **Pre-trained:** Gemma 3 base models (pre-trained + distillation + RLHF/DPO)
- **Adapter support:** LoRA adapters, trained with TRL

### 2.4 Architecture

```
Gemma 3 Architecture (e.g., 4B)
├── Embedding: 262,144 vocab → 2,560 hidden
├── Layers: 34
├── Hidden Size: 2,560
├── Attention Heads: 32
├── KV Heads: 16 (Group Query Attention)
├── Intermediate Size: 10,240
├── Context Length: 128K
├── Sliding Window: 512 tokens (local attention layers)
├── Vision Encoder: SigLIP ViT (4B and above)
└── License: Gemma Terms of Use
```

### 2.5 GGUF File Format

QAT GGUF files are on HuggingFace:
```
gemma-3-QAT-1B-it-Q4_0.gguf    (617 MB)
gemma-3-QAT-4B-it-Q4_0.gguf    (2.43 GB)
gemma-3-QAT-12B-it-Q4_0.gguf   (7.37 GB)
gemma-3-QAT-27B-it-Q4_0.gguf   (15.8 GB)
```

**Why GGUF?** Not the most efficient runtime format, but best for local/Ollama inference.

---

## 3. Qwen 3 — Alibaba

> Source: [huggingface.co/Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) (Apr 2025)
> Source: [qwenlm.github.io/blog/qwen3](https://qwenlm.github.io/blog/qwen3)
> Source: [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)

### 3.1 Overview

Qwen 3 is Alibaba's latest open-source LLM family, released **April 2025**.

**Key Stats:**
- **License:** Apache 2.0 (fully open, no restrictions)
- **Sizes:** 0.6B to 235B parameters
- **Total models:** 8 Dense + 1 MoE
- **Thinking modes:** "Thinking" + "Non-thinking" mode switching
- **Best open-source model** in its class at time of release

### 3.2 Full Model Family

**Dense Models:**

| Model | Params | Layers | Hidden | Attn Heads | KV Heads | Context | License |
|-------|--------|--------|--------|-----------|---------|---------|---------|
| Qwen3-0.6B | 0.6B | 28 | 1,024 | 16 | 8 | 32K | Apache 2.0 |
| Qwen3-1.7B | 1.7B | 28 | 2,048 | 16 | 8 | 32K | Apache 2.0 |
| Qwen3-4B | 4B | 36 | 2,560 | 32 | 8 | 128K | Apache 2.0 |
| Qwen3-8B | 8B | 36 | 4,096 | 32 | 8 | 128K | Apache 2.0 |
| Qwen3-14B | 14B | 40 | 5,120 | 40 | 8 | 131K | Apache 2.0 |
| Qwen3-32B | 32B | 64 | 5,120 | 64 | 8 | 131K | Apache 2.0 |

**MoE Model:**

| Model | Total Params | Active/Token | Experts | Layers | Hidden | Attn Heads | Context |
|-------|-------------|-------------|---------|--------|--------|-----------|---------|
| Qwen3-235B-A22B | 235B | 22B | 128 | 94 | 4,096 | 64 | 131K |

### 3.3 Architecture Deep Dive

```
Qwen3 Architecture (Dense, e.g., 14B)
├── Type: Decoder-only Transformer
├── Layers: 40
├── Hidden Size: 5,120
├── Attention Heads: 40 (GQA with 8 KV heads)
├── KV Heads: 8 (Group Query Attention)
├── Intermediate Size: 17,408 (SwiGLU FFN)
├── Vocabulary Size: 151,936
├── Context Length: 131,072 tokens
├── RoPE: Rotary Position Embeddings
├── Attention: Group Query Attention (GQA)
├── FFN: SwiGLU activation
├── Normalization: RMSNorm
├── Thinking Mode: Dual-mode (Thinking + Non-thinking)
└── License: Apache 2.0

Qwen3-235B-A22B (MoE)
├── Total Params: 235B
├── Active per Token: 22B
├── Experts per Token: 8 (out of 128 total experts)
├── Shared Experts: 1
├── Layers: 94
├── Hidden Size: 4,096
├── Attention Heads: 64
├── KV Heads: 8
├── Context Length: 131,072
├── MoE Top-K: 8
├── FFN: SwiGLU per expert
└── License: Apache 2.0
```

### 3.4 Group Query Attention (GQA)

```
Standard MHA:           GQA (Qwen 3):
Q: 40 heads             Q: 40 heads
K: 40 heads             K: 8 heads (shared across 5 Q heads each)
V: 40 heads             V: 8 heads

Memory: 40×40 = 1600    Memory: 40×8 = 320
Cost: High               Cost: 5x lower
```

### 3.5 Thinking Mode (Dual-Mode)

Qwen 3 supports both **thinking** and **non-thinking** modes — a first for open-source LLMs:

**Thinking Mode:** Chain-of-thought reasoning, step-by-step, slower but more accurate
**Non-thinking Mode:** Direct answer, faster, less token usage

**How it works:**
- `/think` command → enables thinking mode (model generates reasoning chain)
- `/no_think` command → disables thinking mode (direct answer)
- Can be toggled per prompt
- Hybrid: "some thinking, some direct" for efficiency

### 3.6 Performance

Qwen3-32B beats:
- **Qwen3-235B-A22B** (235B MoE) on math tasks
- **Llama-4-Scout** (109B)
- **DeepSeek V3** (671B)
- **OpenAI o1** (reasoning model)

### 3.7 GGUF Files (Ollama)

```bash
ollama run qwen3:32b       # 32B dense, Q4_K_M, ~19GB
ollama run qwen3:14b       # 14B dense, Q4_K_M, ~9GB
ollama run qwen3:8b        # 8B dense, Q4_K_M, ~5GB
ollama run qwen3:4b        # 4B dense, Q4_K_M, ~3GB
ollama run qwen3:235b      # 235B MoE, very large
```

---

## 4. DeepSeek V3 — DeepSeek

> Source: [arxiv.org/html/2501.12948v1](https://arxiv.org/html/2501.12948v1) (Jan 2025)
> Source: [deepseek.com](https://www.deepseek.com/)
> Source: [huggingface.co/deepseek-ai](https://huggingface.co/deepseek-ai)
> Source: [deepwiki.com/deepseek-ai/deepseek-v3](https://deepwiki.com/deepseek-ai/deepseek-v3)

### 4.1 Overview

DeepSeek V3 is a **Mixture-of-Experts (MoE) LLM** from DeepSeek (China). Released December 2024.

**Key Stats:**
- **Total Parameters:** 671B (MoE — 37B activated per token)
- **Training Tokens:** 14.8T
- **Context Length:** 128K tokens (163,840 in some variants)
- **License:** MIT License (fully open, no restrictions)
- **Cost to Train:** $5.6M on 2,048 H800 GPUs (2 months)
- **Performance:** On par with GPT-4o and Claude 3.5 Sonnet

### 4.2 Architecture Deep Dive

```
DeepSeek V3 Architecture
├── Type: Decoder-only Transformer with MoE
├── Total Params: 671B
├── Active per Token: 37B (5.5% of total)
├── Layers: 61
├── Hidden Size: 7,168
├── Attention Heads: 128
├── KV Heads: 128 (Standard MHA, not GQA)
├── Intermediate Size: 18,432
├── Vocabulary Size: 129,280
├── Context Length: 163,840 tokens
├── KV Cache: 72 KB per token
├── FFN Type: SwiGLU (DeepSeekMoE)
├── Normalization: RMSNorm
├── Positional: RoPE (Rotary Position Embeddings)
├── Attention Type: Multi-head Latent Attention (MLA)
├── MoE Type: DeepSeekMoE (auxiliary-loss-free)
└── License: MIT
```

### 4.3 Multi-head Latent Attention (MLA) — The Breakthrough

MLA is DeepSeek V3's key innovation. It dramatically reduces KV cache memory:

**Standard MHA:**
```
Each head stores: Q, K, V = 128 heads × dim × 2 bytes = large KV cache
```

**MLA (DeepSeek V3):**
```
Compress KV into latent space: K, V → low-rank projection
KV cache = Compressed latent per token (72 KB/token)
Only 72 KB per token for ALL 128 attention heads
```

**Impact:** 75% reduction in KV cache memory vs standard MHA, while maintaining quality.

### 4.4 DeepSeekMoE — Auxiliary-Loss-Free Load Balancing

Traditional MoE: expert load imbalance → training collapses
DeepSeek V3: **auxiliary-loss-free** approach — no extra loss function to balance

**How it works:**
- Each token is routed to top-8 experts out of 256
- One "shared expert" always processes every token (256 + 1 = 257 total experts)
- Router uses bias term to balance load without auxiliary loss
- Result: stable training, efficient expert utilization

### 4.5 Multi-Token Prediction

DeepSeek V3 predicts **multiple future tokens simultaneously** (not just one):

```
Standard: Input → Predict next token → Output
DeepSeek: Input → Predict next 4 tokens at once → Output

Training: Next-token prediction + Multi-token prediction (combined loss)
Inference: Multi-token prediction for faster generation
```

**Benefits:**
- 8-10% training efficiency improvement
- Faster inference (predicts multiple tokens in one forward pass)
- Better code generation (code is more predictable)

### 4.6 FP8 Mixed Precision Training

DeepSeek V3 was trained in **FP8 precision** — the first large-scale model to do so:

```
Traditional: FP16/BF16 training
DeepSeek V3: FP8 mixed precision

GPU: 2,048 × H800 (80GB HBM3)
Memory savings: 2x vs FP16
Compute savings: ~1.5x vs FP16
Quality impact: Negligible (with careful implementation)
```

### 4.7 Performance Benchmarks

| Benchmark | DeepSeek V3 | GPT-4o | Claude 3.5 Sonnet | Qwen3-235B |
|-----------|-------------|--------|-------------------|------------|
| MMLU (Overall) | **88.5** | 87.2 | 88.3 | 81.5 |
| MMLU (STEM) | 90.0 | 89.5 | 89.8 | 82.3 |
| AIME 2024 (Math) | 39.2 | 29.3 | 28.4 | 65.8 |
| LiveCodeBench | 40.5 | 33.4 | 32.8 | 34.4 |
| MATH-500 | 90.2 | 74.6 | 78.3 | 90.0 |
| HumanEval (Code) | 82.6 | 90.2 | 88.0 | 85.0 |

**Key takeaway:** DeepSeek V3 beats GPT-4o on most benchmarks and is competitive with Claude 3.5 Sonnet — while being fully open-source.

### 4.8 Training Cost Comparison

| Model | Cost | GPUs | Time | Params |
|-------|------|------|------|--------|
| GPT-4 (estimated) | $100M+ | ~25,000 A100 | 100 days | ~1.8T |
| Llama 3 405B | $60M+ | 16,384 H100 | 54 days | 405B |
| DeepSeek V3 | **$5.6M** | **2,048 H800** | **2 months** | **671B** |

**Cost ratio:** DeepSeek V3 training cost = 1/18th of Llama 3 405B.

### 4.9 Model Variants

| Model | Params | Active | MoE Experts | Context | Released |
|-------|--------|--------|------------|---------|----------|
| DeepSeek V3-0324 | 685B | 37B | 256+1 | 162K | Mar 2025 |
| DeepSeek V3.1 | 671B | 37B | 256+1 | 128K | Aug 2025 |
| DeepSeek-R1 | 671B | 37B | 256+1 | 128K | Jan 2025 |

### 4.10 DeepSeek V3 vs V3.1

V3.1 (Aug 2025) improvements:
- 40% fewer output tokens (more concise)
- Better tool use
- Better Chinese language support
- Smaller KV cache footprint
- Better long-context performance

### 4.11 Deployment

**Cloud:**
- DeepSeek API: $0.27/M input, $1.10/M output (cache hit: $0.07/M)
- Hugging Face Inference API
- Fireworks AI, Together AI

**Local:**
- GGUF quantized versions available
- Requires 48GB+ RAM for Q4 quantized 671B
- llama.cpp / Ollama supported

---

## 5. Head-to-Head Comparison

### 5.1 Architecture Comparison

| Feature | Gemma 4 E4B | Qwen 3-14B | DeepSeek V3 |
|---------|-------------|------------|-------------|
| **Total Params** | 8B | 14B | 671B |
| **Active Params** | 4.5B (PLE) | 14B (Dense) | 37B (MoE) |
| **Type** | Dense (PLE) | Dense | MoE |
| **Layers** | 34 | 40 | 61 |
| **Hidden Size** | 3,584 | 5,120 | 7,168 |
| **Attention Heads** | 16 | 40 | 128 |
| **KV Heads** | 16 | 8 | 128 |
| **Context Length** | 128K | 131K | 128K |
| **Vocab Size** | 262,144 | 151,936 | 129,280 |
| **Attention Type** | Hybrid SWA | GQA | MLA |
| **MoE** | No | No | Yes (256+1 experts) |
| **Multimodal** | Yes (Vision+Audio) | Yes (Vision) | No (Text only) |
| **Thinking Mode** | No | Yes | No |
| **License** | Gemma Terms | Apache 2.0 | MIT |

### 5.2 Efficiency Comparison

| Metric | Gemma 4 E4B | Qwen 3-14B | DeepSeek V3 |
|--------|-------------|------------|-------------|
| **RAM (Q4)** | ~4GB | ~9GB | ~48GB+ |
| **Inference Cost** | Very Low | Low | Medium |
| **KV Cache/Token** | Small (512 window) | Medium | 72KB (MLA) |
| **Training Cost** | N/A (Google) | N/A (Alibaba) | $5.6M |
| **Params/Token** | 4.5B | 14B | 37B |
| **Efficiency Ratio** | Best (PLE) | Good (GQA) | Good (MoE) |

### 5.3 Quality Comparison

| Benchmark | Gemma 4 E4B | Qwen 3-14B | DeepSeek V3 |
|-----------|-------------|------------|-------------|
| **MMLU** | ~75 (est.) | ~82 | 88.5 |
| **Math** | ~60 (est.) | ~75 | 90.2 |
| **Code** | ~55 (est.) | ~70 | 82.6 |
| **Multilingual** | Good | Good | Good |
| **Visual Q&A** | Excellent | Good | N/A |

### 5.4 Use Case Recommendation

| Use Case | Best Model | Why |
|----------|-----------|-----|
| **Mobile/Edge (4GB RAM)** | Gemma 4 E4B | Only 4GB needed, multimodal |
| **Phone/Laptop (8GB RAM)** | Qwen 3-8B | Good balance, Apache 2.0 |
| **Desktop (16GB RAM)** | Qwen 3-32B | Beats 235B MoE on math |
| **Server (48GB+ RAM)** | DeepSeek V3 | Best quality overall |
| **Vision + Audio** | Gemma 4 E4B | Native multimodal |
| **Math/Science** | Qwen 3-32B | Beats larger models |
| **Code Generation** | DeepSeek V3 | LiveCodeBench #1 |
| **Commercial (no restrictions)** | Qwen 3 / DeepSeek V3 | Apache 2.0 / MIT |

---

## 6. Key Takeaways

### 6.1 The New Paradigm: Efficiency > Size

Gemma 4 E4B proves that **effective params matter more than total params**. With PLE (Per-Layer Embeddings), an 8B "total" model can match a 4.5B "effective" model — beating models 3-5x its size.

### 6.2 MoE is the Future (But Dense Isn't Dead)

DeepSeek V3's 671B total / 37B active per token shows MoE's power: massive knowledge base with efficient inference. But Qwen 3-32B (dense) beats Qwen 3-235B (MoE) on math — proving dense architectures still have their place.

### 6.3 Attention Innovation Drives Efficiency

- **Gemma 4:** Hybrid sliding-window (local + global) — efficient for long context
- **Qwen 3:** Group Query Attention (GQA) — 5x KV cache reduction
- **DeepSeek V3:** Multi-head Latent Attention (MLA) — 75% KV cache reduction

### 6.4 Training Cost Revolution

DeepSeek V3 trained 671B params for $5.6M on 2,048 GPUs in 2 months — proving open-source can compete with $100M+ closed models. FP8 training was the breakthrough.

### 6.5 Open Source Wins

All three models are fully open:
- **Gemma 4:** Gemma Terms (commercial OK)
- **Qwen 3:** Apache 2.0 (fully open)
- **DeepSeek V3:** MIT License (fully open)

The gap between open and closed models is closing fast.

---

## 7. References

1. Google. "Gemma 4." ai.google.dev/gemma/docs/gemma-4. Apr 2, 2026.
2. Google. "Gemma 3." ai.google.dev/gemma/docs/gemma-3. Dec 2025.
3. Google. "Gemma 3 QAT GGUF." huggingface.co/google/gemma-3-QAT-1B-it-GGUF.
4. Qwen Team. "Qwen3." qwenlm.github.io/blog/qwen3. Apr 2025.
5. Qwen Team. "Qwen3-32B." huggingface.co/Qwen/Qwen3-32B. Apr 2025.
6. Qwen Team. "Qwen3 Technical Report." arxiv.org/html/2505.09388v1. May 2025.
7. DeepSeek AI. "DeepSeek-V3 Technical Report." arxiv.org/html/2501.12948v1. Jan 2025.
8. DeepSeek AI. "DeepSeek-V3." deepseek.com. Dec 2024.
9. DeepSeek AI. "DeepSeek-V3.1." huggingface.co/deepseek-ai/DeepSeek-V3.1. Aug 2025.
10. DeepWiki. "DeepSeek-V3." deepwiki.com/deepseek-ai/deepseek-v3. Aug 2025.

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
- **Topic 9: Model Architecture Specs — `AI-Model-Report-Topic9-Model-Architecture-Specs.md`** (This Report)
