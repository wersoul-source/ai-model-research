# Topic 7: Detailed Model Assembly Principles + Assembly Flowchart (Referencing Topic 3)

> Research Report by: Kampun (BOI Family)
> Session: S10 (2026-08-10)
> Status: Completed
> References: Topic 3 (Frameworks, Research Papers, Steps to Build Models)

---

## Table of Contents
1. [Overview: What is "Model Assembly"?](#1-overview)
2. [10 Core Principles of Model Assembly](#2-10-core-principles)
3. [The Complete Assembly Flowchart](#3-the-complete-assembly-flowchart)
4. [Detailed Step-by-Step Guide (with Topic 3 References)](#4-detailed-step-by-step-guide)
5. [Transformer-Specific Assembly](#5-transformer-specific-assembly)
6. [Common Pitfalls and How to Avoid Them](#6-common-pitfalls)
7. [Summary](#7-summary)

---

## 1. Overview

"Model Assembly" is the **end-to-end process** of taking a raw idea and turning it into a deployed, production-ready AI model. It's not just "write code and train" — it's a systematic pipeline with 10+ stages, each with specific inputs, outputs, and quality gates.

**This report references Topic 3** which covered:
- Frameworks (PyTorch, TensorFlow, JAX, Keras3)
- Research papers (Transformer, GPT, BERT, LLaMA, etc.)
- 5 difficulty levels of model building

**The flowchart below shows the COMPLETE journey** from idea to production.

---

## 2. 10 Core Principles of Model Assembly

### Principle 1: Problem-First, Not Model-First
> "Choose the right problem before choosing the right model."

- Define the business/research objective FIRST
- Determine the type of ML task (classification, regression, generation, etc.)
- Set success metrics BEFORE building anything
- Reference: Topic 3 Level 1 — "Using Pre-trained Models"

### Principle 2: Data is the Foundation
> "Garbage in, garbage out — but gold in, gold out."

- Collect high-quality, relevant data
- Clean, validate, and preprocess BEFORE any modeling
- Split into train/validation/test BEFORE feature engineering
- Reference: Topic 3 Level 2 — "Fine-tuning Pre-trained Models"

### Principle 3: Start Simple, Then Scale
> "Beginners build simple models. Experts also build simple models first."

- Always start with a baseline (logistic regression, random forest)
- Then try more complex models only if needed
- Document why each upgrade was necessary
- Reference: Topic 3 Level 3 — "Building Models from Existing Architectures"

### Principle 4: Modular Design
> "Build blocks, not monoliths."

- Design components that can be swapped independently
- Tokenizer, Embedding, Attention, FFN, Output — each is a module
- Test each module separately before combining
- Reference: Topic 3 Level 4 — "Custom Architecture Design"

### Principle 5: Reproducibility is Non-Negotiable
> "If you can't reproduce it, it didn't happen."

- Version control everything: code, data, configs, models
- Use fixed random seeds
- Log all experiments with hyperparameters and results
- Reference: Topic 3 Level 5 — "Research-Level Model Building"

### Principle 6: Evaluation Before Optimization
> "Measure twice, optimize once."

- Evaluate on a held-out test set ONCE at the end
- Use validation set for hyperparameter tuning
- Never optimize on test set (data leakage!)
- Reference: Topic 3 — "Evaluation Metrics"

### Principle 7: Hardware-Aware Design
> "Design for the hardware you have, not the hardware you wish you had."

- Know your GPU VRAM budget before choosing model size
- Use mixed precision (FP16/BF16) for 2x memory savings
- Consider quantization early, not as an afterthought
- Reference: Topic 3 — "Hardware Requirements"

### Principle 8: Iteration is the Process
> "The first model is never the final model."

- Expect to iterate 5-10x before reaching acceptable performance
- Each iteration should be documented
- Keep what works, discard what doesn't
- Reference: Topic 3 — "Model Building Steps"

### Principle 9: Deployment is Not an Afterthought
> "A model that can't be deployed is just a research paper."

- Design for inference from the start
- Consider latency, throughput, and memory constraints
- Test the full pipeline: data -> model -> prediction -> response
- Reference: Topic 3 — "Frameworks for Deployment"

### Principle 10: Monitor and Maintain
> "Models don't degrade — the world changes around them."

- Track model performance in production
- Set up alerts for data drift and performance degradation
- Schedule periodic retraining
- Reference: Topic 3 — "Production Considerations"

---

## 3. The Complete Assembly Flowchart

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    AI MODEL ASSEMBLY FLOWCHART                          ║
║                    (Referencing Topic 3: Frameworks, Papers, Steps)     ║
╚══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: FOUNDATION (Topic 3 Level 1-2)                               │
│  Time: 10-20% of total project                                         │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  1. PROBLEM       │
    │  DEFINITION       │
    │                   │
    │  - Business goal  │
    │  - ML task type   │
    │  - Success metrics│
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  2. DATA           │
    │  COLLECTION        │
    │                    │
    │  - Sources         │
    │  - Volume needed   │
    │  - Quality check   │
    └────────┬──────────┘
             │
             ▼
    ┌──────────────────┐
    │  3. DATA            │
    │  PREPROCESSING      │
    │                     │
    │  - Cleaning         │
    │  - Normalization    │
    │  - Augmentation     │
    │  - Split (70/15/15) │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────┐
    │  4. TOKENIZATION   │
    │  / FEATURE ENG.    │
    │                    │
    │  - BPE / WordPiece │
    │  - Embeddings      │
    │  - Positional enc. │
    └────────┬──────────┘
             │

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: ARCHITECTURE (Topic 3 Level 3-4)                             │
│  Time: 10-15% of total project                                         │
└─────────────────────────────────────────────────────────────────────────┘

             │
             ▼
    ┌──────────────────┐
    │  5. MODEL           │
    │  SELECTION          │
    │                     │
    │  - Framework:       │
    │    PyTorch/TensorFlow│
    │  - Architecture:    │
    │    CNN/RNN/Transf.  │
    │  - Size: 1B/7B/70B  │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────┐
    │  6. ARCHITECTURE    │
    │  DESIGN             │
    │                     │
    │  - Layers & dims    │
    │  - Attention heads  │
    │  - FFN expansion    │
    │  - Residual connect.│
    │  - LayerNorm/RMSNorm│
    └────────┬──────────┘
             │

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: TRAINING (Topic 3 Level 4-5)                                 │
│  Time: 50-70% of total project                                         │
└─────────────────────────────────────────────────────────────────────────┘

             │
             ▼
    ┌──────────────────┐
    │  7. HYPERPARAMETER │
    │  CONFIGURATION      │
    │                     │
    │  - Learning rate    │
    │  - Batch size       │
    │  - Epochs           │
    │  - Optimizer        │
    │  - Scheduler        │
    │  - Dropout          │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────┐
    │  8. TRAINING        │
    │  LOOP               │
    │                     │
    │  ┌───────────────┐  │
    │  │ Forward Pass   │  │
    │  │ (Compute Loss) │  │
    │  └───────┬───────┘  │
    │          │          │
    │  ┌───────▼───────┐  │
    │  │ Backward Pass  │  │
    │  │ (Compute Grad) │  │
    │  └───────┬───────┘  │
    │          │          │
    │  ┌───────▼───────┐  │
    │  │ Optimizer Step │  │
    │  │ (Update Weights)│ │
    │  └───────┬───────┘  │
    │          │          │
    │     Repeat N times  │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────┐
    │  9. VALIDATION      │
    │  & EARLY STOPPING   │
    │                     │
    │  - Monitor val loss │
    │  - Stop if overfit  │
    │  - Save best model  │
    └────────┬───────────┘
             │

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: OPTIMIZATION (Topic 3 Level 5)                               │
│  Time: 10-15% of total project                                         │
└─────────────────────────────────────────────────────────────────────────┘

             │
             ▼
    ┌──────────────────┐
    │  10. EVALUATION     │
    │                     │
    │  - Test set metrics │
    │  - Benchmark scores │
    │  - Manual inspection│
    │  - Ablation studies │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────┐
    │  11. OPTIMIZATION   │
    │                     │
    │  - Quantization     │
    │    (INT8/INT4)      │
    │  - Pruning          │
    │  - Distillation     │
    │  - Flash Attention  │
    └────────┬───────────┘
             │

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: DEPLOYMENT                                                   │
│  Time: 5-10% of total project                                          │
└─────────────────────────────────────────────────────────────────────────┘

             │
             ▼
    ┌──────────────────┐
    │  12. EXPORT         │
    │                     │
    │  - Save weights     │
    │  - Save tokenizer   │
    │  - Save config      │
    │  - Format: SafeTens.│
    │    / GGUF / ONNX    │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────┐
    │  13. DEPLOY         │
    │                     │
    │  - API server       │
    │  - Container (Docker│
    │  - Edge device      │
    │  - Cloud (AWS/GCP)  │
    └────────┬───────────┘
             │
             ▼
    ┌──────────────────┐
    │  14. MONITOR        │
    │  & MAINTAIN         │
    │                     │
    │  - Performance      │
    │  - Data drift       │
    │  - Retrain cycle    │
    │  - Version updates  │
    └────────────────────┘
```

---

## 4. Detailed Step-by-Step Guide (with Topic 3 References)

### Step 1: Problem Definition

**What to do:**
- Write a clear problem statement (1-2 sentences)
- Define input (what data you have) and output (what you want)
- Determine task type: Classification / Regression / Generation / Retrieval
- Set success metrics: Accuracy / F1 / BLEU / Perplexity / Latency

**Topic 3 Reference:**
- Level 1: "Using Pre-trained Models via API" — Problem is already solved
- Level 2: "Fine-tuning" — Problem is defined, adapting model to your data
- Level 3-5: "Building from scratch" — You define everything

**Example:**
```
Problem: "Classify customer support tickets into categories"
Input: Text ticket
Output: Category (billing, technical, sales, other)
Metrics: F1-score > 0.90, Latency < 100ms
```

---

### Step 2: Data Collection

**What to do:**
- Identify data sources (databases, APIs, web scraping, synthetic)
- Estimate volume needed (1K for fine-tuning, 1B+ for pre-training)
- Check quality: completeness, accuracy, bias
- Ensure legal compliance (GDPR, copyright)

**Topic 3 Reference:**
- Framework: Hugging Face Datasets, TensorFlow Datasets
- Paper: "Scaling Laws for Neural Language Models" (Kaplan 2020) — data size matters

**Data Volume Guidelines:**
| Task | Minimum Data | Recommended |
|------|-------------|-------------|
| Fine-tuning classification | 1,000 examples | 10,000+ |
| Fine-tuning generation | 10,000 examples | 100,000+ |
| Pre-training LLM | 1B tokens | 1T+ tokens |
| Vision fine-tuning | 1,000 images | 10,000+ |

---

### Step 3: Data Preprocessing

**What to do:**
- Clean: Remove duplicates, fix errors, handle missing values
- Normalize: Lowercase, remove special chars, standardize format
- Augment: Synonym replacement, back-translation, noise injection
- Split: Train (70%) / Validation (15%) / Test (15%)

**Topic 3 Reference:**
- Framework: Pandas, NumPy, spaCy, NLTK
- Rule: Split BEFORE any feature engineering to prevent data leakage

**Preprocessing Pipeline:**
```
Raw Data
  │
  ├── Remove duplicates
  ├── Handle missing values
  ├── Normalize text/images
  ├── Augment (if needed)
  ├── Split (70/15/15)
  │
  └── Clean Dataset
```

---

### Step 4: Tokenization / Feature Engineering

**What to do:**
- Choose tokenizer: BPE / WordPiece / SentencePiece
- Build vocabulary (or use pre-built)
- Convert text to token IDs
- Add special tokens: [CLS], [SEP], [PAD], [MASK]

**Topic 3 Reference:**
- Framework: Hugging Face Tokenizers, SentencePiece
- Paper: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin 2018)

**Tokenizer Comparison:**
| Tokenizer | Used By | Vocab Size | Language Support |
|-----------|---------|------------|------------------|
| BPE | GPT-2/3/4, LLaMA | 32K-128K | Multi-language |
| WordPiece | BERT, DistilBERT | 30K | Multi-language |
| SentencePiece | T5, mBART | 32K-256K | Language-agnostic |

---

### Step 5: Model Selection

**What to do:**
- Choose framework: PyTorch (research) / TensorFlow (production)
- Choose architecture based on task:
  - Text classification: BERT, RoBERTa
  - Text generation: GPT, LLaMA, Mistral
  - Image classification: ResNet, ViT
  - Multimodal: GPT-4V, Gemini
- Choose model size: 1B / 7B / 13B / 70B

**Topic 3 Reference:**
- Framework: PyTorch (most popular), TensorFlow, JAX
- Papers: Topic 3 covered 10 foundational papers

**Architecture Selection Guide:**
```
Task Type?
│
├── Text Classification → BERT / RoBERTa / DeBERTa
├── Text Generation → GPT / LLaMA / Mistral
├── Translation → T5 / mBART / NLLB
├── Summarization → BART / Pegasus / T5
├── Image Classification → ResNet / ViT / EfficientNet
├── Object Detection → YOLO / DETR / Faster R-CNN
├── Multimodal → GPT-4V / Gemini / Claude 3.5
└── Code Generation → CodeLlama / StarCoder / Codestral
```

---

### Step 6: Architecture Design

**What to do:**
- Define number of layers (depth)
- Define hidden dimension (width)
- Define number of attention heads
- Define FFN expansion factor (typically 4x)
- Add residual connections and layer normalization

**Topic 3 Reference:**
- Paper: "Attention Is All You Need" (Vaswani 2017) — Transformer architecture
- Paper: "LLaMA: Open and Efficient Foundation Language Models" (Touvron 2023)

**Architecture Parameters (LLaMA-style):**
| Model Size | Layers | Hidden Dim | Heads | FFN Dim | Params |
|-----------|--------|-----------|-------|---------|--------|
| 1B | 16 | 2048 | 32 | 5632 | 1.3B |
| 7B | 32 | 4096 | 32 | 11008 | 6.7B |
| 13B | 40 | 5120 | 40 | 13824 | 13.0B |
| 70B | 80 | 8192 | 64 | 28672 | 68.3B |

---

### Step 7: Hyperparameter Configuration

**What to do:**
- Set learning rate (1e-4 to 5e-4 for Transformers)
- Set batch size (depends on VRAM)
- Set number of epochs (3-10 for fine-tuning, 1-3 for pre-training)
- Choose optimizer (AdamW for Transformers)
- Choose scheduler (cosine with warmup)

**Topic 3 Reference:**
- Paper: "Adam: A Method for Stochastic Optimization" (Kingma 2014)

**Hyperparameter Starting Points:**
| Parameter | Fine-tuning | Pre-training |
|-----------|------------|--------------|
| Learning Rate | 1e-5 to 5e-5 | 1e-4 to 3e-4 |
| Batch Size | 16-32 | 256-2048 |
| Epochs | 3-10 | 1-3 |
| Warmup Steps | 10% of total | 1% of total |
| Weight Decay | 0.01 | 0.1 |
| Dropout | 0.1 | 0.0 |

---

### Step 8: Training Loop

**What to do:**
- Forward pass: Compute predictions
- Compute loss: CrossEntropy / MSE / etc.
- Backward pass: Compute gradients
- Optimizer step: Update weights
- Repeat for N epochs

**Topic 3 Reference:**
- Framework: PyTorch `model.train()` + `loss.backward()` + `optimizer.step()`
- Framework: TensorFlow `model.fit()`

**Training Loop Pseudocode:**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch.input)
        loss = loss_fn(outputs, batch.target)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    val_loss = validate(model, val_loader)
    if val_loss < best_val_loss:
        save_model(model)
```

---

### Step 9: Validation & Early Stopping

**What to do:**
- Monitor validation loss after each epoch
- Stop if validation loss doesn't improve for N epochs (patience)
- Save the best model checkpoint
- Use learning rate scheduling to reduce overfitting

**Topic 3 Reference:**
- Technique: Early Stopping, Model Checkpointing

**Early Stopping Logic:**
```
patience = 5
best_val_loss = infinity
counter = 0

for epoch in range(max_epochs):
    val_loss = validate(model)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

---

### Step 10: Evaluation

**What to do:**
- Load best model checkpoint
- Run on held-out test set (ONCE!)
- Compute metrics: Accuracy, F1, BLEU, ROUGE, Perplexity
- Analyze errors: Where does the model fail?
- Ablation studies: What components matter most?

**Topic 3 Reference:**
- Framework: Hugging Face `evaluate`, `scikit-learn`
- Paper: "GLUE: A Multi-Task Benchmark" (Wang 2018)

**Evaluation Metrics by Task:**
| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Classification | F1-Score | Accuracy, Precision, Recall, AUC |
| Generation | Perplexity | BLEU, ROUGE, Human Eval |
| Translation | BLEU | METEOR, chrF++ |
| Summarization | ROUGE-L | ROUGE-1, ROUGE-2 |
| Retrieval | MRR | Recall@K, NDCG |

---

### Step 11: Optimization

**What to do:**
- Quantization: Reduce precision (FP16 -> INT8 -> INT4)
- Pruning: Remove unnecessary weights
- Distillation: Train smaller model from larger one
- Flash Attention: Optimize attention computation
- KV Cache: Speed up autoregressive generation

**Topic 3 Reference:**
- Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao 2022)
- Paper: "GPTQ: Accurate Post-Training Quantization" (Frantar 2023)

**Optimization Techniques:**
| Technique | Speedup | Quality Loss | When to Use |
|-----------|---------|--------------|-------------|
| FP16 | 2x | None | Always |
| INT8 | 4x | ~1% | Production |
| INT4 (GPTQ) | 8x | ~3-5% | Edge deployment |
| Pruning | 2-3x | ~1-2% | Memory constrained |
| Distillation | 10x | ~5-10% | Mobile/edge |

---

### Step 12: Export

**What to do:**
- Save model weights
- Save tokenizer
- Save config.json
- Choose format: SafeTensors / GGUF / ONNX

**Topic 3 Reference:**
- Framework: Hugging Face `model.save_pretrained()`
- Format: Topic 5 covered file formats in detail

**Export Commands:**
```bash
# Hugging Face format
model.save_pretrained("./my-model")
tokenizer.save_pretrained("./my-model")

# GGUF (for llama.cpp)
python convert_hf_to_gguf.py ./my-model --outfile model.gguf

# ONNX
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

### Step 13: Deployment

**What to do:**
- Set up API server (FastAPI / Flask / TGI / vLLM)
- Containerize (Docker)
- Deploy to cloud (AWS / GCP / Azure) or edge
- Set up load balancing and auto-scaling

**Topic 3 Reference:**
- Framework: FastAPI, Docker, Kubernetes
- Tool: vLLM, TGI (Text Generation Inference)

**Deployment Options:**
| Option | Latency | Cost | Scalability |
|--------|---------|------|-------------|
| Cloud API (OpenAI) | Low | High | Unlimited |
| Self-hosted (vLLM) | Low | Medium | Manual |
| Edge (llama.cpp) | Very Low | Low | Limited |
| Serverless (Lambda) | Medium | Pay-per-use | Auto |

---

### Step 14: Monitor & Maintain

**What to do:**
- Track prediction quality over time
- Monitor for data drift (input distribution changes)
- Set up alerts for performance degradation
- Schedule periodic retraining with fresh data
- Version control model updates

**Topic 3 Reference:**
- Tool: Prometheus, Grafana, Weights & Biases
- Technique: A/B testing, Shadow deployment

**Monitoring Dashboard:**
```
┌─────────────────────────────────────────┐
│  MODEL HEALTH DASHBOARD                 │
├─────────────────────────────────────────┤
│  Accuracy: 92.3% (target: >90%)  ✅    │
│  Latency P99: 85ms (target: <100ms) ✅ │
│  Throughput: 1,200 req/s          ✅    │
│  Error Rate: 0.1% (target: <1%)   ✅   │
│  Data Drift: 0.02 (threshold: 0.1) ✅  │
│  Last Retrain: 2026-08-01         ✅    │
└─────────────────────────────────────────┘
```

---

## 5. Transformer-Specific Assembly

### Transformer Block Assembly Order

```
Input Tokens
    │
    ▼
┌─────────────────────────────────────┐
│  TOKEN EMBEDDING                    │
│  tokens → token_ids → embeddings    │
│  Shape: (batch, seq_len) → (batch,  │
│          seq_len, d_model)          │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  POSITIONAL ENCODING                │
│  Add position information           │
│  - Sinusoidal (original)            │
│  - RoPE (LLaMA, modern)            │
│  - ALiBi (BLOOM)                   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  TRANSFORMER BLOCK (x N layers)     │
│  ┌───────────────────────────────┐  │
│  │  LayerNorm → Multi-Head Attn  │  │
│  │  → Residual Connection        │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  LayerNorm → FFN (4x expand) │  │
│  │  → Residual Connection        │  │
│  └───────────────────────────────┘  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  FINAL LAYERNORM                    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  LINEAR OUTPUT LAYER                │
│  (batch, seq_len, d_model) →        │
│  (batch, seq_len, vocab_size)       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  SOFTMAX → Token Probabilities      │
│  → Argmax → Next Token              │
└─────────────────────────────────────┘
```

### Multi-Head Attention Assembly

```
Input (batch, seq_len, d_model)
    │
    ├──→ W_q → Q (batch, seq_len, d_model)
    ├──→ W_k → K (batch, seq_len, d_model)
    └──→ W_v → V (batch, seq_len, d_model)
              │
              ▼
    ┌─────────────────────────────────┐
    │  Split into H heads             │
    │  Q: (batch, heads, seq_len, d_k)│
    │  K: (batch, heads, seq_len, d_k)│
    │  V: (batch, heads, seq_len, d_k)│
    └─────────────────┬───────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │  Attention(Q, K, V)             │
    │  = softmax(QK^T / sqrt(d_k)) V  │
    │  per head                       │
    └─────────────────┬───────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │  Concatenate heads              │
    │  (batch, seq_len, d_model)      │
    └─────────────────┬───────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │  W_o → Output                   │
    │  (batch, seq_len, d_model)      │
    └─────────────────────────────────┘
```

---

## 6. Common Pitfalls and How to Avoid Them

| Pitfall | Symptom | Prevention |
|---------|---------|------------|
| **Data Leakage** | Test accuracy much higher than real-world | Split data BEFORE any preprocessing |
| **Overfitting** | Train acc >> Val acc | More data, regularization, early stopping |
| **Underfitting** | Both train and val acc low | Bigger model, more epochs, lower regularization |
| **Wrong Metric** | High accuracy but bad UX | Choose metrics that match business goals |
| **Ignoring Edge Cases** | Model fails on rare inputs | Add diverse test cases, stress testing |
| **No Baseline** | Can't tell if model is good | Always start with simple baseline |
| **Premature Optimization** | Wweeks on quantization before training | Get model working FIRST, optimize later |
| **No Version Control** | Can't reproduce results | Git + DVC for code + data + models |

---

## 7. Summary

### Key Takeaways:

1. **Model Assembly has 14 steps** across 5 phases:
   - Phase 1: Foundation (Problem, Data, Preprocessing, Tokenization)
   - Phase 2: Architecture (Selection, Design)
   - Phase 3: Training (Config, Loop, Validation)
   - Phase 4: Optimization (Evaluation, Optimization)
   - Phase 5: Deployment (Export, Deploy, Monitor)

2. **10 Core Principles** guide every decision:
   - Problem-first, Data-first, Start-simple, Modular, Reproducible
   - Evaluate-before-optimize, Hardware-aware, Iterative, Deploy-aware, Monitor

3. **Topic 3 References** throughout:
   - Levels 1-5 map to different stages of the pipeline
   - Frameworks (PyTorch, TensorFlow) are tools for each step
   - Research papers provide the theoretical foundation

4. **The flowchart is your roadmap:**
   - Follow it step-by-step
   - Don't skip phases
   - Each step has specific inputs and outputs

5. **Common pitfalls are avoidable:**
   - Data leakage, overfitting, wrong metrics, no baseline
   - Prevention is easier than fixing

### Quick Reference:

| Question | Answer |
|----------|--------|
| How many steps? | 14 steps across 5 phases |
| What's Phase 1? | Foundation (Problem, Data, Preprocess, Tokenize) |
| What's Phase 2? | Architecture (Selection, Design) |
| What's Phase 3? | Training (Config, Loop, Validation) |
| What's Phase 4? | Optimization (Eval, Optimize) |
| What's Phase 5? | Deployment (Export, Deploy, Monitor) |
| What framework to use? | PyTorch (research), TensorFlow (production) |
| What paper to read first? | "Attention Is All You Need" (2017) |

---

## References

1. Vaswani et al., "Attention Is All You Need" (2017)
2. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
3. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
4. Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
5. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
6. Frantar et al., "GPTQ: Accurate Post-Training Quantization" (2023)
7. Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
8. Wang et al., "GLUE: A Multi-Task Benchmark" (2018)
9. Hugging Face Documentation — https://huggingface.co/docs
10. PyTorch Documentation — https://pytorch.org/docs

---

**Report completed by: Kampun (BOI Family)**
**Date: 2026-08-10**
**Session: S10**
**Status: DONE**
