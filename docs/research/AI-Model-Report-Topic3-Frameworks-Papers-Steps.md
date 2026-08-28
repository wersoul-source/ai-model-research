# Topic 3: Frameworks สำหรับสร้างโมเดล + เอกสารวิจัย + ขั้นตอนการสร้างโมเดล

> รายงานโดย: คำปัน (Kampun) | BOI Family
> วันที่: 2026-08-19
> แหล่งอ้างอิง: 40+ แหล่ง (เอกสารวิจัย, เอกสารเทคนิค, บทความวิเคราะห์)

---

## สารบัญ

1. [AI Frameworks เปรียบเทียบ](#1-ai-frameworks-เปรียบเทียบ)
2. [Distributed Training Frameworks](#2-distributed-training-frameworks)
3. [เอกสารวิจัยพื้นฐาน (Foundational Papers)](#3-เอกสารวิจัยพื้นฐาน-foundational-papers)
4. [ขั้นตอนการสร้างโมเดล: ง่ายสุด → ยากสุด](#4-ขั้นตอนการสร้างโมเดล-ง่ายสุด--ยากสุด)
5. [สรุป: Framework Selection Guide](#5-สรุป-framework-selection-guide)

---

## 1. AI Frameworks เปรียบเทียบ

### 1.1 ภาพรวมของ Framework Landscape (2025-2026)

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI FRAMEWORK ECOSYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   PyTorch   │  │  TensorFlow │  │     JAX     │            │
│  │  (Meta)     │  │  (Google)   │  │  (Google)   │            │
│  │  60-75%     │  │  20-38%     │  │  5-15%      │            │
│  │  Research   │  │  Production │  │  Research   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                 ┌────────▼────────┐                             │
│                 │     Keras 3     │                             │
│                 │ (Backend-Agnostic)                             │
│                 │  TF + PyTorch + JAX                           │
│                 └─────────────────┘                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              High-Level Libraries                       │   │
│  │  Hugging Face Transformers • PyTorch Lightning         │   │
│  │  Keras • FastAI • Flax (for JAX)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 PyTorch (Meta) — ครองตลาด Research

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| **Developed by** | Meta (Facebook) |
| **First Release** | 2016 |
| **Market Share** | 60-75% (research), 55% (production) |
| **Strengths** | Dynamic computation graph, Pythonic, debugging ง่าย |
| **Weaknesses** | Production deployment ต้องใช้ TorchScript/ONNX |
| **Best For** | Research, prototyping, NLP, LLMs |

**ทำไม PyTorch ครองตลาด Research:**
1. **Dynamic Graph** — แก้ code ได้ตลอด training (เหมือน Python ธรรมดา)
2. **Debugging** — ใช้ pdb/IDE debugger ได้เลย
3. **Pythonic** — เขียนเหมือน Python ทั่วไป
4. **Community** — 75%+ ของ paper ใหม่ใช้ PyTorch
5. **Hugging Face** — 500,000+ models เป็น PyTorch-native

**ตัวอย่าง Code:**
```python
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()      # Dynamic graph
        optimizer.step()
```

### 1.3 TensorFlow (Google) — ครองตลาด Production

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| **Developed by** | Google |
| **First Release** | 2015 |
| **Market Share** | 20-38% (production) |
| **Strengths** | Production tools, TPU support, deployment ecosystem |
| **Weaknesses** | Learning curve สูง, debugging ยากกว่า |
| **Best For** | Enterprise, mobile (TFLite), web (TF.js) |

**Production Ecosystem:**
- **TensorFlow Serving** — model serving
- **TensorFlow Lite** — mobile/edge deployment
- **TensorFlow.js** — browser deployment
- **TensorFlow Extended (TFX)** — ML pipeline
- **TensorBoard** — visualization

**ตัวอย่าง Code:**
```python
import tensorflow as tf

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Train
model.fit(train_data, epochs=10)
```

### 1.4 JAX (Google) — ครองตลาด High-Performance Research

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| **Developed by** | Google |
| **First Release** | 2018 |
| **Market Share** | 5-15% (growing) |
| **Strengths** | Functional programming, JIT compilation, XLA optimization |
| **Weaknesses** | Learning curve สูงมาก, ecosystem เล็ก |
| **Best For** | TPU training, scientific computing, high-performance |

**Key Features:**
- **`jit`** — JIT compilation ผ่าน XLA
- **`vmap`** — vectorization อัตโนมัติ
- **`grad`** — automatic differentiation
- **Functional** — pure functions, no side effects

**ตัวอย่าง Code:**
```python
import jax
import jax.numpy as jnp

# Pure function
def predict(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# JIT compiled
@jax.jit
def update(params, x, y, lr=0.01):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

### 1.5 Keras 3 — Backend-Agnostic (Game Changer)

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| **First Release** | Keras 3.0 (late 2023) |
| **Strengths** | เขียนครั้งเดียว รันบน TF/PyTorch/JAX |
| **Best For** | Beginners, rapid prototyping, multi-framework |

**Keras 3 Revolution:**
```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # เปลี่ยน backend ได้!

import keras
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10)
])
# รันบน PyTorch, TensorFlow, หรือ JAX
```

### 1.6 Framework Comparison Matrix

| Criteria | PyTorch | TensorFlow | JAX | Keras 3 |
|----------|---------|------------|-----|---------|
| **Learning Curve** | ง่าย | ปานกลาง | ยาก | ง่ายมาก |
| **Debugging** | ง่ายมาก | ยาก | ปานกลาง | ง่าย |
| **Research Adoption** | 75%+ | 15% | 10% | via backends |
| **Production Tools** | ดี | ดีมาก | ปานกลาง | ดี |
| **TPU Support** | ดี | ดีมาก | ดีมาก | ขึ้นกับ backend |
| **Mobile Deploy** | ปานกลาง | ดีมาก | ไม่支持 | via TF Lite |
| **Community Size** | ใหญ่มาก | ใหญ่ | เล็ก | ใหญ่ |
| **Pre-trained Models** | 500K+ (HF) | TensorFlow Hub | น้อย | ขึ้นกับ backend |

### 1.7 Decision Matrix: เลือก Framework ไหน?

| Use Case | เลือก | เหตุผล |
|----------|-------|--------|
| **Research / Prototyping** | PyTorch | Dynamic graph, debugging ง่าย |
| **Production Deployment** | TensorFlow | Mature tools, TFLite, TF Serving |
| **TPU Training** | JAX | XLA optimization, functional |
| **Beginners** | Keras 3 | High-level API, backend-agnostic |
| **LLM Fine-tuning** | PyTorch + Hugging Face | Ecosystem ครบ |
| **Edge/Mobile** | TensorFlow Lite | Production-ready |
| **Scientific Computing** | JAX | Performance, differentiability |

---

## 2. Distributed Training Frameworks

### 2.1 ทำไมต้อง Distributed Training?

```
┌─────────────────────────────────────────────────────────────────┐
│                 MEMORY BOTTLENECK                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model Size: 7B parameters                                     │
│  FP16: 14 GB (fits in 1 GPU)                                  │
│                                                                 │
│  Model Size: 70B parameters                                    │
│  FP16: 140 GB (ต้องการ 2× A100 80GB)                          │
│                                                                 │
│  Model Size: 405B parameters                                   │
│  FP16: 810 GB (ต้องการ 16K× H100 80GB)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Parallelism Strategies

| Strategy | วิธีทำงาน | When to Use |
|----------|----------|-------------|
| **Data Parallelism (DP)** | แบ่ง data บน GPU หลายตัว | Model ใส่ GPU เดียวได้ |
| **Tensor Parallelism (TP)** | แบ่ง layers บน GPU หลายตัว | Model ใหญ่เกิน 1 GPU |
| **Pipeline Parallelism (PP)** | แบ่ง stages ตามลำดับ | Model ใหญ่มาก |
| **Sequence Parallelism (SP)** | แบ่ง sequence length | Long context training |
| **Expert Parallelism (EP)** | แบ่ง MoE experts | Mixture of Experts |

### 2.3 DeepSpeed (Microsoft)

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| ** Developed by** | Microsoft |
| **Key Feature** | ZeRO (Zero Redundancy Optimizer) |
| **Best For** | Memory-constrained training, NVMe offload |

**ZeRO Stages:**
```
ZeRO-1: Shard optimizer states only
ZeRO-2: + Shard gradients
ZeRO-3: + Shard parameters (= FSDP)
ZeRO-Infinity: + Offload to CPU/NVMe
```

**Config Example:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "nvme"}
  },
  "bf16": {"enabled": true},
  "train_micro_batch_size_per_gpu": 1
}
```

### 2.4 Megatron-LM (NVIDIA)

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| **Developed by** | NVIDIA |
| **Key Feature** | 3D Parallelism (TP + PP + DP) |
| **Best For** | Frontier-scale training (100B+) |

**Config Example:**
```python
megatron_args = {
    "tensor_model_parallel_size": 8,    # TP=8
    "pipeline_model_parallel_size": 4,  # PP=4
    "context_parallel_size": 2,         # CP=2
    "sequence_parallel": True,
    "use_distributed_optimizer": True,
}
```

**Used by:** LLaMA 3, Mistral, Mixtral, DeepSeek, Nemotron

### 2.5 PyTorch FSDP

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| **Developed by** | PyTorch Team |
| **Key Feature** | PyTorch-native ZeRO-3 |
| **Best For** | 7B-30B fine-tuning |

**Config Example:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

### 2.6 Framework Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│           DISTRIBUTED TRAINING DECISION TREE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model Size < 7B?                                              │
│  └─ YES → Single GPU or DDP (Data Parallel)                    │
│                                                                 │
│  Model Size 7B-30B?                                            │
│  └─ YES → FSDP2 (PyTorch-native, simple API)                   │
│                                                                 │
│  Model Size 30B-70B + tight memory?                            │
│  └─ YES → DeepSpeed ZeRO-3 + CPU/NVMe offload                  │
│                                                                 │
│  Model Size 70B+ or pretraining?                               │
│  └─ YES → Megatron-LM (3D parallelism)                         │
│                                                                 │
│  Need to swap backends?                                        │
│  └─ YES → Hugging Face Accelerate                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. เอกสารวิจัยพื้นฐาน (Foundational Papers)

### 3.1 Timeline: วิวัฒนาการของ AI Architecture

```
1998  ───────────────────────────────────────────────────────► 2024+
│
├─ 1998: LeNet (CNN)
├─ 2012: AlexNet (Deep CNN + GPU)
├─ 2014: GAN (Generative Adversarial Networks)
├─ 2014: Seq2Seq (Encoder-Decoder)
├─ 2015: ResNet (Deep Residual Learning)
├─ 2017: Transformer (Attention Is All You Need) ◄── GAME CHANGER
├─ 2018: BERT (Encoder-only)
├─ 2018: GPT (Decoder-only)
├─ 2020: GPT-3 (Scaling Laws)
├─ 2022: InstructGPT/ChatGPT (RLHF)
├─ 2023: LLaMA (Open-weights)
├─ 2023: GPT-4 (Multimodal)
├─ 2024: Mixtral (MoE)
├─ 2024: LLaMA 3 (Frontier open-source)
└─ 2024: DeepSeek-R1 (Reasoning)
```

### 3.2 เอกสารสำคัญ 10 ฉบับที่ต้องอ่าน

#### 1. Attention Is All You Need (2017)
** Vaswani et al. | Google Brain**

**สิ่งที่ introduc:**
- Transformer Architecture
- Self-Attention Mechanism
- Multi-Head Attention
- Positional Encoding

**ทำไมสำคัญ:**
> "เปลี่ยน from RNN/LSTM → Attention; เป็น foundation ของ GPT, BERT, และ LLM ทุกตัว"

**link:** https://arxiv.org/abs/1706.03762

---

#### 2. BERT: Pre-training of Deep Bidirectional Transformers (2018)
**Devlin et al. | Google**

**สิ่งที่ introduc:**
- Encoder-only Transformer
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Fine-tuning paradigm

**ทำไมสำคัญ:**
> "พิสูจน์ว่า pre-training + fine-tuning ใช้ได้กับ NLP ทุก task"

**link:** https://arxiv.org/abs/1810.04805

---

#### 3. Language Models are Unsupervised Multitask Learners (GPT-2, 2019)
**Radford et al. | OpenAI**

**สิ่งที่ introduc:**
- Decoder-only Transformer
- Zero-shot learning
- Text generation

**ทำไมสำคัญ:**
> "พิสูจน์ว่า large language models สามารถทำ task ได้โดยไม่ต้อง fine-tune"

**link:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

---

#### 4. Language Models are Few-Shot Learners (GPT-3, 2020)
**Brown et al. | OpenAI**

**สิ่งที่ introduc:**
- Scaling Laws (175B parameters)
- In-context learning
- Few-shot prompting

**ทำไมสำคัญ:**
> "แสดงให้เห็นว่า scale = capability; เปลี่ยนวิธีคิดเรื่อง model size"

**link:** https://arxiv.org/abs/2005.14165

---

#### 5. Deep Residual Learning for Image Recognition (ResNet, 2015)
**He et al. | Microsoft**

**สิ่งที่ introduc:**
- Residual Connections (Skip connections)
- Deep networks (152+ layers)
- Highway Networks

**ทำไมสำคัญ:**
> "แก้ vanishing gradient; ทำให้ train deep networks ได้"

**link:** https://arxiv.org/abs/1512.03385

---

#### 6. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet, 2012)
**Krizhevsky et al. | University of Toronto**

**สิ่งที่ introduc:**
- Deep CNN on GPU
- ReLU activation
- Dropout regularization
- Data augmentation

**ทำไมสำคัญ:**
> "จุดเริ่มต้นของ deep learning revolution; GPU training"

**link:** https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

---

#### 7. Generative Adversarial Networks (GAN, 2014)
**Goodfellow et al. | Université de Montréal**

**สิ่งที่ introduc:**
- Generator vs Discriminator
- Adversarial training
- Generative models

**ทำไมสำคัญ:**
> "เปลี่ยนวิธีคิดเรื่อง generative models; ต้นแบบของ StyleGAN, DALL-E"

**link:** https://arxiv.org/abs/1406.2661

---

#### 8. Efficient Estimation of Word Representations (Word2Vec, 2013)
**Mikolov et al. | Google**

**สิ่งที่ introduc:**
- Word embeddings
- Skip-gram, CBOW
- Semantic vector space

**ทำไมสำคัญ:**
> "ทำให้ NLP เข้าใจ semantics; ก่อนจะมาเป็น Transformer"

**link:** https://arxiv.org/abs/1301.3781

---

#### 9. Training language models to follow instructions (InstructGPT, 2022)
**Ouyang et al. | OpenAI**

**สิ่งที่ introduc:**
- RLHF (Reinforcement Learning from Human Feedback)
- Instruction tuning
- Alignment

**ทำไมสำคัญ:**
> "เปลี่ยน from autocomplete → helpful assistant; ChatGPT foundation"

**link:** https://arxiv.org/abs/2203.02155

---

#### 10. LLaMA: Open and Efficient Foundation Language Models (2023)
**Touvron et al. | Meta**

**สิ่งที่ introduc:**
- Open-weights model
- Efficient training (7B-65B)
- Competitive with GPT-3

**ทำไมสำคัญ:**
> "ปล่อย open-source LLM; ทำให้ everyone สร้าง LLM ได้"

**link:** https://arxiv.org/abs/2302.13971

---

### 3.3 Paper Categories

| Category | Papers | Key Concept |
|----------|--------|-------------|
| **Architecture** | Transformer, ResNet, AlexNet | Model structure |
| **Language Models** | BERT, GPT-2, GPT-3, LLaMA | Pre-training paradigm |
| **Alignment** | InstructGPT, Constitutional AI | Human values |
| **Efficiency** | FlashAttention, LoRA, Quantization | Resource optimization |
| **Generation** | GAN, Diffusion, VAE | Content creation |
| **Multimodal** | CLIP, GPT-4, Gemini | Multiple modalities |

---

## 4. ขั้นตอนการสร้างโมเดล: ง่ายสุด → ยากสุด

### 4.1 Difficulty Levels

```
┌─────────────────────────────────────────────────────────────────┐
│                 MODEL BUILDING DIFFICULTY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: ใช้ Pre-trained Model (ง่ายสุด)                      │
│  ├─ เวลา: 1-7 วัน                                             │
│  ├─ Skill: Python basic                                        │
│  ├─ Hardware: Laptop                                          │
│  └─ Cost: ฟรี                                                  │
│                                                                 │
│  Level 2: Fine-tune Model                                      │
│  ├─ เวลา: 1-4 สัปดาห์                                         │
│  ├─ Skill: Python + ML basics                                  │
│  ├─ Hardware: GPU (1x RTX 3060+)                               │
│  └─ Cost: $100-1,000                                           │
│                                                                 │
│  Level 3: Train from Scratch (Small)                           │
│  ├─ เวลา: 1-3 เดือน                                           │
│  ├─ Skill: Deep Learning + Math                                │
│  ├─ Hardware: GPU cluster (4-8 GPUs)                           │
│  └─ Cost: $10,000-100,000                                      │
│                                                                 │
│  Level 4: Train from Scratch (Large)                           │
│  ├─ เวลา: 3-12 เดือน                                          │
│  ├─ Skill: Distributed Systems + ML                           │
│  ├─ Hardware: GPU cluster (64-1000+ GPUs)                      │
│  └─ Cost: $1M-100M                                             │
│                                                                 │
│  Level 5: Frontier Model (ยากสุด)                               │
│  ├─ เวลา: 1-3 ปี                                              │
│  ├─ Skill: Research + Engineering team                         │
│  ├─ Hardware: 1000-16,000+ GPUs                                │
│  └─ Cost: $100M-1B+                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Level 1: ใช้ Pre-trained Model (ง่ายสุด)

**เป้าหมาย:** ใช้ model ที่คนอื่น train แล้ว

**ขั้นตอน:**
```
1. เลือก task (classification, generation, etc.)
2. ค้นหา pre-trained model (Hugging Face Hub)
3. ติดตั้ง library (transformers, torch)
4. เขียน code 5-20 บรรทัด
5. Run!
```

**ตัวอย่าง:**
```python
from transformers import pipeline

# ใช้ model สำเร็จรูป
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

**เครื่องมือ:**
- Hugging Face Transformers
- TensorFlow Hub
- PyTorch Hub

---

### 4.3 Level 2: Fine-tune Model

**เป้าหมาย:** ปรับ pre-trained model ให้เหมาะกับ task เฉพาะ

**ขั้นตอน:**
```
1. รวบรวมข้อมูล (100-10,000 samples)
2. เตรียม data (tokenization, formatting)
3. เลือก pre-trained model
4. ตั้ง hyperparameters
5. Train (fine-tune)
6. Evaluate
7. Deploy
```

**เทคนิคที่ใช้:**
- **Full Fine-tuning** — ปรับทุก parameter
- **LoRA** — ปรับเฉพาะ small matrices
- **QLoRA** — Fine-tune ด้วย quantized model
- **Prompt Tuning** — ปรับ prompt เท่านั้น

**ตัวอย่าง LoRA:**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(base_model, config)
# Train เฉพาะ parameters ที่เลือก
```

**ต้นทุน:**
- 1 GPU (RTX 3060 12GB): 7B model — 1-2 ชั่วโมง
- Cloud (A100 80GB): 70B model — 1-7 วัน

---

### 4.4 Level 3: Train from Scratch (Small Model)

**เป้าหมาย:** สร้าง model ใหม่ตั้งแต่ต้น (1M-1B parameters)

**ขั้นตอน:**
```
1. ออกแบบ architecture
2. รวบรวมข้อมูล (100GB-1TB)
3. ตั้ง hyperparameters
4. Train บน GPU cluster
5. Evaluate + Iterate
6. Deploy
```

**สิ่งที่ต้องรู้:**
- Neural network architecture
- Backpropagation
- Gradient descent
- Regularization
- Mixed precision training

**Hardware:**
- 4-8 GPUs (RTX 4090 or A100)
- 128-256 GB RAM
- 1-4 TB Storage

**ต้นทุน:**
- Cloud: $10,000-100,000
- On-premise: $50,000-200,000

---

### 4.5 Level 4: Train from Scratch (Large Model)

**เป้าหมาย:** สร้าง LLM ขนาด 7B-70B parameters

**ขั้นตอน:**
```
1. ออกแบบ architecture (Transformer)
2. รวบรวมข้อมูล (100GB-10TB)
3. ตั้ง distributed training
4. Train บน GPU cluster (64-1000 GPUs)
5. Monitor + Adjust
6. Evaluate
7. Deploy
```

**สิ่งที่ต้องรู้:**
- Distributed training (FSDP, DeepSpeed, Megatron)
- Mixed precision (FP16, BF16, FP8)
- Gradient accumulation
- Checkpointing
- Fault tolerance

**Hardware:**
- 64-256 GPUs (A100 or H100)
- 1-4 TB RAM
- 10-100 TB Storage
- High-speed networking (InfiniBand)

**ต้นทุน:**
- Cloud: $1M-10M
- On-premise: $5M-50M

---

### 4.6 Level 5: Frontier Model (ยากสุด)

**เป้าหมาย:** สร้าง model ขนาด 100B-1T+ parameters

**ขั้นตอน:**
```
1. วิจัย architecture ใหม่
2. ออกแบบ training pipeline
3. รวบรวมข้อมูล (100TB-1PB)
4. ตั้ง cluster (1000-16,000+ GPUs)
5. Train หลายเดือน
6. Iterate + Debug
7. Align (RLHF, Constitutional AI)
8. Deploy
```

**สิ่งที่ต้องรู้:**
- Cutting-edge research
- Custom CUDA kernels
- Fault-tolerant distributed systems
- Alignment techniques
- Safety measures

**Hardware:**
- 1,000-16,000+ GPUs (H100/H200)
- 100+ TB RAM
- 1+ PB Storage
- Petabit networking

**ต้นทุน:**
- $100M-1B+
- ทีม 50-200+ people
- 1-3 ปี

**ตัวอย่าง: LLaMA 3 405B Training**
- Hardware: 16,384 × H100 80GB
- Framework: Megatron-LM + custom stack
- Time: ~54 days
- Cost: ~$100M+

---

### 4.7 Comparison Table

| Level | Model Size | Data | Hardware | Time | Cost | Skill |
|-------|-----------|------|----------|------|------|-------|
| **1** | Pre-trained | 0 | Laptop | 1-7 days | Free | Python |
| **2** | Fine-tune | 100-10K | 1 GPU | 1-4 weeks | $100-1K | Python + ML |
| **3** | 1M-1B | 100GB-1TB | 4-8 GPUs | 1-3 months | $10K-100K | DL + Math |
| **4** | 7B-70B | 100GB-10TB | 64-256 GPUs | 3-12 months | $1M-10M | Distributed |
| **5** | 100B-1T+ | 100TB-1PB | 1K-16K+ GPUs | 1-3 years | $100M+ | Research |

---

## 5. สรุป: Framework Selection Guide

### 5.1 Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              FRAMEWORK SELECTION GUIDE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. เริ่มจาก Use Case:                                         │
│     ├─ Research → PyTorch                                       │
│     ├─ Production → TensorFlow                                 │
│     ├─ TPU/Performance → JAX                                   │
│     └─ Beginners → Keras 3                                     │
│                                                                 │
│  2. ถ้าต้อง Distributed Training:                              │
│     ├─ < 7B → DDP / FSDP2                                      │
│     ├─ 7B-30B → FSDP2                                          │
│     ├─ 30B-70B → DeepSpeed ZeRO-3                              │
│     ├─ 70B+ → Megatron-LM                                      │
│     └─ Flexible → Hugging Face Accelerate                      │
│                                                                 │
│  3. ถ้าต้อง Deployment:                                        │
│     ├─ Mobile → TensorFlow Lite                                │
│     ├─ Web → TensorFlow.js                                     │
│     ├─ Server → TorchServe / TF Serving                        │
│     └─ Edge → ONNX Runtime                                     │
│                                                                 │
│  4. ถ้าต้อง Speed:                                             │
│     ├─ Training → CUDA + Mixed Precision                       │
│     ├─ Inference → TensorRT / ONNX                             │
│     └─ Production → Triton Inference Server                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Recommended Stack by Level

| Level | Framework | Training | Deployment |
|-------|-----------|----------|------------|
| **1** | Keras/Transformers | Single GPU | Hugging Face |
| **2** | PyTorch + PEFT | Single GPU | TorchServe |
| **3** | PyTorch + Lightning | Multi-GPU | ONNX |
| **4** | PyTorch + FSDP/DeepSpeed | Multi-node | Triton |
| **5** | Megatron-LM + Custom | Cluster | Custom |

### 5.3 Learning Path

```
Beginner → Intermediate → Advanced → Expert
    │           │            │          │
    ▼           ▼            ▼          ▼
  Keras 3    PyTorch      FSDP/      Megatron
             + HF         DeepSpeed   Custom
```

---

## อ้างอิง

1. Vaswani et al. (2017). "Attention Is All You Need"
2. Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Brown et al. (2020). "Language Models are Few-Shot Learners"
4. He et al. (2015). "Deep Residual Learning for Image Recognition"
5. Krizhevsky et al. (2012). "ImageNet Classification with Deep CNNs"
6. Goodfellow et al. (2014). "Generative Adversarial Networks"
7. Mikolov et al. (2013). "Efficient Estimation of Word Representations"
8. Ouyang et al. (2022). "Training language models to follow instructions"
9. Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
10. Shoeybi et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models"
11. Rajbhandari et al. (2019). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
12. Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters"
13. PyTorch Documentation. "FSDP Tutorial"
14. DeepSpeed Documentation. "ZeRO Optimization"
15. NVIDIA Megatron-LM. "GitHub Repository"

---

**End of Report — Topic 3**
**คำปัน (Kampun) | BOI Family | 2026-08-19**
