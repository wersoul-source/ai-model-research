# Topic 2: Model AI มีส่วนประกอบอะไรบ้าง (แบบละเอียด) และขั้นต่ำของระบบที่โมเดลจะรองรับ + เครื่องรองรับ

> รายงานโดย: คำปัน (Kampun) | BOI Family
> วันที่: 2026-08-19
> แหล่งอ้างอิง: 30+ แหล่ง (学术論文, เอกสารเทคนิค, บทความวิเคราะห์)

---

## สารบัญ

1. [ภาพรวมของ AI Model Architecture](#1-ภาพรวมของ-ai-model-architecture)
2. [ส่วนประกอบภายใน Model (Detailed)](#2-ส่วนประกอบภายใน-model-แบบละเอียด)
3. [Minimum System Requirements ตามขนาด Model](#3-minimum-system-requirements-ตามขนาด-model)
4. [Quantization — ลดขนาด Model อย่างมีประสิทธิภาพ](#4-quantization--ลดขนาด-model-อย่างมีประสิทธิภาพ)
5. [Compatible Devices & Hardware per Tier](#5-compatible-devices--hardware-per-tier)
6. [Build Your Own AI Model — Checklist](#6-build-your-own-ai-model--checklist)

---

## 1. ภาพรวมของ AI Model Architecture

### 1.1 โครงสร้างหลักของ Transformer Model

```
┌─────────────────────────────────────────────────────────┐
│                    AI MODEL ARCHITECTURE                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐                                       │
│  │   TOKENIZER  │  แปลง text → tokens ( IDs)           │
│  └──────┬───────┘                                       │
│         │                                               │
│  ┌──────▼───────┐                                       │
│  │  EMBEDDINGS  │  แปลง tokens → vectors (semantic)    │
│  └──────┬───────┘                                       │
│         │                                               │
│  ┌──────▼───────┐                                       │
│  │  POSITIONAL  │  เพิ่มตำแหน่งใน sequence             │
│  │   ENCODING   │                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│  ┌──────▼───────┐  ◄── Repeat N times (layers)         │
│  │   SELF-      │                                       │
│  │   ATTENTION  │  ให้ model ให้ความสำคัญกับ            │
│  │  (Multi-Head)│  token ที่เกี่ยวข้องกัน                │
│  └──────┬───────┘                                       │
│         │                                               │
│  ┌──────▼───────┐                                       │
│  │  FEED-FORWARD│  แปลงความเข้าใจ → output              │
│  │  NETWORK     │                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│  ┌──────▼───────┐                                       │
│  │   RESIDUAL   │  เพิ่ม skip connection + norm          │
│  │  + LAYER NORM│                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│  ┌──────▼───────┐                                       │
│  │   OUTPUT     │  ทำนาย token ถัดไป                     │
│  │  PROJECTION  │                                       │
│  └──────────────┘                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 ขนาด Model ที่นิยมในปัจจุบัน

| ขนาด | ตัวอย่าง | Parameters | ใช้งานทั่วไป |
|------|----------|------------|-------------|
| Small | GPT-2, DistilBERT | < 1B | Text classification, simple Q&A |
| Medium | LLaMA 7B, Mistral 7B | 1B - 10B | Chat, summarization, code |
| Large | LLaMA 13B, Falcon 40B | 10B - 50B | Complex reasoning, code gen |
| Very Large | LLaMA 70B, Falcon 180B | 50B - 200B | Expert-level reasoning |
| Frontier | GPT-4, Claude, Gemini | 200B+ | Frontier capabilities |

---

## 2. ส่วนประกอบภายใน Model (แบบละเอียด)

### 2.1 Tokenizer

**หน้าที่:** แปลงข้อความ (text) ให้เป็นตัวเลข (token IDs) ที่ model เข้าใจ

| ประเภท | วิธีทำงาน | ตัวอย่าง |
|--------|----------|---------|
| **BPE** (Byte-Pair Encoding) | รวม byte ที่อยู่ติดกันบ่อยที่สุด | GPT-2/3/4, LLaMA |
| **WordPiece** | เลือก subword ที่ maximize likelihood | BERT, DistilBERT |
| **SentencePiece** | ไม่ต้อง pre-tokenize | T5, ALBERT |
| **Unigram** | ลบ subword ที่ไม่ค่อยเกิด | XLNet |

**ข้อเท็จจริงสำคัญ:**
- Vocabulary ทั่วไป: 32,000 - 100,000 tokens
- ภาษาไทย: ต้องการ vocabulary พิเศษ (ไม่ใช่ BPE ธรรมดา)
- 1 token ≈ 4 ตัวอักษรภาษาอังกฤษ ≈ 1-2 ตัวอักษรไทย

### 2.2 Embeddings

**หน้าที่:** แปลง token IDs ให้เป็น vector (list ของตัวเลข) ที่มีความหมาย

```
Token "cat" → [0.2, -0.5, 0.8, 0.1, ...] (维度: d_model)
Token "dog" → [0.3, -0.4, 0.7, 0.2, ...] (ใกล้กับ "cat")
Token "car" → [-0.1, 0.6, -0.3, 0.9, ...] (ไกลจาก "cat")
```

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| **Dimension** | 768 (small) → 4096 (large) → 12,288 (GPT-4) |
| **Learned** | ฝึกจากข้อมูล (ส่วนใหญ่) |
| **Positional** | เพิ่มข้อมูลตำแหน่งใน sequence |

### 2.3 Positional Encoding

**หน้าที่:** บอก model ว่าแต่ละ token อยู่ตำแหน่งไหนในประโยค

| ประเภท | วิธี | ใช้ใน |
|--------|------|-------|
| **Sinusoidal** | ใช้ sin/cos functions | Original Transformer |
| **Learned** | ฝึก embedding สำหรับตำแหน่ง | BERT, GPT-2 |
| **Rotary (RoPE)** | หมุน vectors ตามตำแหน่ง | LLaMA, Mistral |
| **ALiBi** | เพิ่ม bias ตามระยะห่าง | BLOOM |

### 2.4 Self-Attention Mechanism

**หน้าที่:** ให้ model "ให้ความสำคัญ" กับ token ที่เกี่ยวข้องกัน

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

| ประเภท | ความแตกต่าง | ใช้ใน |
|--------|------------|-------|
| **Multi-Head Attention** | หลายหัว parallel | Original Transformer |
| **Multi-Query Attention (MQA)** | Query หลายหัว, KV หัวเดียว | Falcon |
| **Grouped-Query Attention (GQA)** | Query หลายหัว, KV กลุ่ม | LLaMA 2/3 |
| **Flash Attention** | IO-aware, เร็วขึ้น 2-4x | LLaMA, Mistral |

**ตัวเลขสำคัญ:**
- Heads: 8 (small) → 32 (large) → 96 (GPT-4)
- d_k (head dimension): 64 → 128
- Context length: 2K → 4K → 32K → 128K → 1M

### 2.5 Feed-Forward Network (FFN)

**หน้าที่:** แปลงความเข้าใจจาก Attention ให้เป็น output

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

| ประเภท | สูตร | ใช้ใน |
|--------|------|-------|
| **Standard FFN** | W₁ (d_model → 4×d_model) → W₂ | BERT |
| **GLU Variants** | Gate × Up → Down | LLaMA, Mistral |
| **SwiGLU** | Swish(xW₁) ⊙ (xW₃) → W₂ | LLaMA 2/3 |
| **MoE (Mixture of Experts)** | เลือก FFN หลายตัวตาม routing | Mixtral, GShard |

**MoE — Mixture of Experts:**
- มี FFN หลาย "experts" (เช่น 8 experts)
- Router เลือก 2 experts ต่อ token
- ลด computation 50-75% แต่ parameter เท่าเดิม
- ตัวอย่าง: Mixtral 8x7B = 47B params, ใช้ 12.9B ต่อ token

### 2.6 Residual Connections + Layer Normalization

**หน้าที่:** ป้องกัน vanishing gradient + ทำให้ training ราบเรียบ

| ประเภท | วิธี | ใช้ใน |
|--------|------|-------|
| **Post-LN** | LayerNorm หลัง residual | Original Transformer |
| **Pre-LN** | LayerNorm ก่อน attention | GPT-2/3 |
| **RMSNorm** | ไม่มี centering | LLaMA, Mistral |

### 2.7 Output Projection

**หน้าที่:** แปลง hidden state ให้เป็น probability distribution สำหรับ token ถัดไป

```
logits = hidden_state × W_vocab  (d_model → vocab_size)
probs = softmax(logits / temperature)
```

---

## 3. Minimum System Requirements ตามขนาด Model

### 3.1 VRAM Requirements (GPU Memory)

| ขนาด Model | FP32 | FP16/BF16 | INT8 | INT4 (GPTQ/GGUF) |
|------------|------|-----------|------|-------------------|
| **1B** | 4 GB | 2 GB | 1 GB | 0.6 GB |
| **3B** | 12 GB | 6 GB | 3 GB | 1.8 GB |
| **7B** | 28 GB | 14 GB | 7 GB | 4 GB |
| **13B** | 52 GB | 26 GB | 13 GB | 7.5 GB |
| **30B** | 120 GB | 60 GB | 30 GB | 17 GB |
| **70B** | 280 GB | 140 GB | 70 GB | 40 GB |
| **180B** | 720 GB | 360 GB | 180 GB | 100 GB |
| **405B** | 1,620 GB | 810 GB | 405 GB | 230 GB |

**สูตรคำนวณ:**
```
VRAM (GB) ≈ Parameters (B) × Bytes per Parameter
FP32: 4 bytes/param
FP16: 2 bytes/param
INT8: 1 byte/param
INT4: 0.5 bytes/param
```

### 3.2 RAM Requirements (System Memory)

| ขนาด Model | Minimum RAM | Recommended RAM | With 128K Context |
|------------|-------------|-----------------|-------------------|
| **1B** | 4 GB | 8 GB | 16 GB |
| **3B** | 8 GB | 16 GB | 32 GB |
| **7B** | 16 GB | 32 GB | 64 GB |
| **13B** | 32 GB | 64 GB | 128 GB |
| **30B** | 64 GB | 128 GB | 256 GB |
| **70B** | 128 GB | 256 GB | 512 GB |
| **180B** | 256 GB | 512 GB | 1 TB |
| **405B** | 512 GB | 1 TB | 2 TB |

### 3.3 CPU Requirements

| ขนาด Model | Minimum CPU | Recommended CPU | Cores |
|------------|-------------|-----------------|-------|
| **1B** | Intel i3 / AMD Ryzen 3 | Intel i5 / Ryzen 5 | 4-6 |
| **3B** | Intel i5 / AMD Ryzen 5 | Intel i7 / Ryzen 7 | 6-8 |
| **7B** | Intel i7 / AMD Ryzen 7 | Intel i9 / Ryzen 9 | 8-16 |
| **13B** | Intel i9 / AMD Ryzen 9 | Threadripper / EPYC | 16-32 |
| **30B+** | Threadripper / EPYC | Dual Socket Server | 32-64 |

### 3.4 Storage Requirements

| ขนาด Model | FP16 | INT8 | INT4 (GGUF Q4) |
|------------|------|------|-----------------|
| **1B** | 2 GB | 1 GB | 0.6 GB |
| **3B** | 6 GB | 3 GB | 1.8 GB |
| **7B** | 14 GB | 7 GB | 4 GB |
| **13B** | 26 GB | 13 GB | 7.5 GB |
| **30B** | 60 GB | 30 GB | 17 GB |
| **70B** | 140 GB | 70 GB | 40 GB |

---

## 4. Quantization — ลดขนาด Model อย่างมีประสิทธิภาพ

### 4.1 Quantization Methods เปรียบเทียบ

| วิธี | ประเภท | คุณภาพ | ความเร็ว | ใช้ใน |
|------|--------|--------|---------|-------|
| **GPTQ** | Post-training | ดีมาก | เร็ว | GPU (AutoGPTQ) |
| **AWQ** | Post-training | ดีที่สุด | เร็ว | GPU |
| **GGUF/GGML** | Post-training | ดี | เร็ว | CPU + GPU (llama.cpp) |
| **BitsAndBytes** | Dynamic | ดี | ปานกลาง | Hugging Face |
| **QLoRA** | Training | ดีมาก | ช้า | Fine-tuning |
| **SpQR** | Post-training | ดีมาก | เร็ว | GPU |

### 4.2 Quantization Levels เปรียบเทียบ

| Level | Bits | Size (7B) | Quality Loss | Use Case |
|-------|------|-----------|--------------|----------|
| **Q2_K** | 2 | 2.7 GB | สูง | ทดลองเท่านั้น |
| **Q3_K_S** | 3 | 3.0 GB | ปานกลาง-สูง | Memory-constrained |
| **Q4_K_M** | 4 | 4.0 GB | ต่ำ (สมดุล) | **แนะนำสำหรับ most use cases** |
| **Q5_K_M** | 5 | 4.8 GB | ต่ำมาก | Quality-critical |
| **Q6_K** | 6 | 5.5 GB | น้อยที่สุด | Near-lossless |
| **Q8_0** | 8 | 7.0 GB | แทบไม่มี | Production |
| **F16** | 16 | 14.0 GB | ไม่มี | Full quality |

### 4.3 Quantization สำหรับ Device ต่างๆ

| Device | แนะนำ quant | ขนาดสูงสุด | หมายเหตุ |
|--------|-------------|-----------|---------|
| **Phone (8GB RAM)** | Q4_K_M | 3B | ช้าแต่ใช้ได้ |
| **Phone (12GB RAM)** | Q4_K_M | 7B | ต้องมี swap |
| **Laptop (16GB RAM)** | Q4_K_M | 7B | CPU-only |
| **Laptop (32GB RAM)** | Q5_K_M | 13B | CPU-only |
| **Desktop (RTX 3060 12GB)** | Q4_K_M | 7B | GPU-accelerated |
| **Desktop (RTX 4090 24GB)** | Q5_K_M | 13B | GPU-accelerated |
| **Server (2×A100 80GB)** | Q8_0 | 70B | Production |

---

## 5. Compatible Devices & Hardware per Tier

### 5.1 Tier 1: Small Models (< 3B params)

| Device | RAM | VRAM | Speed | Quality |
|--------|-----|------|-------|---------|
| **iPhone 15 Pro** | 8 GB | 6 GB (Metal) | 5-15 tok/s | Good for simple tasks |
| **Samsung Galaxy S24** | 8 GB | 4 GB (Adreno) | 3-10 tok/s | Basic chat |
| **MacBook Air M1** | 8 GB | Unified | 10-30 tok/s | Very good |
| **Raspberry Pi 5** | 8 GB | CPU only | 1-3 tok/s | Very slow |
| **Chromebook (i5)** | 8 GB | CPU only | 2-8 tok/s | Basic tasks |

### 5.2 Tier 2: Medium Models (3B - 13B params)

| Device | RAM | VRAM | Speed | Quality |
|--------|-----|------|-------|---------|
| **MacBook Pro M3 Pro** | 18 GB | Unified | 20-40 tok/s | Excellent |
| **Mac Mini M2 Pro** | 16 GB | Unified | 15-35 tok/s | Very good |
| **Desktop RTX 3060** | 16 GB | 12 GB | 30-60 tok/s | Excellent |
| **Desktop RTX 4070** | 32 GB | 12 GB | 40-80 tok/s | Excellent |
| **Desktop RTX 4090** | 32 GB | 24 GB | 60-120 tok/s | Production-grade |
| **Cloud T4** | 16 GB | 16 GB | 30-60 tok/s | Good (cheapest GPU) |

### 5.3 Tier 3: Large Models (13B - 70B params)

| Device | RAM | VRAM | Speed | Quality |
|--------|-----|------|-------|---------|
| **Mac Studio M2 Ultra** | 192 GB | Unified | 30-60 tok/s | Production |
| **Mac Pro M2 Ultra** | 192 GB | Unified | 30-60 tok/s | Production |
| **2×RTX 3090** | 64 GB | 48 GB | 15-30 tok/s | Research |
| **2×RTX 4090** | 64 GB | 48 GB | 30-60 tok/s | Production |
| **1×A100 80GB** | 128 GB | 80 GB | 40-80 tok/s | Production |
| **Cloud A100** | 128 GB | 80 GB | 40-80 tok/s | Production |

### 5.4 Tier 4: Very Large Models (70B+ params)

| Device | RAM | VRAM | Speed | Quality |
|--------|-----|------|-------|---------|
| **4×RTX 3090** | 128 GB | 96 GB | 10-20 tok/s | Research |
| **2×A100 80GB** | 256 GB | 160 GB | 20-40 tok/s | Production |
| **4×A100 80GB** | 512 GB | 320 GB | 30-60 tok/s | Production |
| **8×H100 80GB** | 1 TB | 640 GB | 50-100 tok/s | Frontier |
| **Cloud H100** | 1 TB | 80 GB | 50-100 tok/s | Frontier |

### 5.5 Cloud vs On-Prem vs Edge

| ประเภท | ต้นทุน | Latency | Privacy | Best For |
|--------|--------|---------|---------|----------|
| **Edge (Phone/Laptop)** | ต่ำ (ครั้งเดียว) | ต่ำมาก | สูงมาก | Offline, personal use |
| **Local Server** | กลาง (hardware) | ต่ำ | สูง | Company internal |
| **Cloud GPU** | สูง (รายชม.) | ปานกลาง | ปานกลาง | Development, small scale |
| **Cloud Cluster** | สูงมาก | ปานกลาง | ต่ำ | Production, large scale |

---

## 6. Build Your Own AI Model — Checklist

### Phase 1: Planning

```
□ กำหนด use case (chat, code, translation, etc.)
□ กำหนดขนาด model ที่ต้องการ
□ กำหนด budget (hardware/cloud)
□ กำหนด latency requirement
□ กำหนด privacy requirement
```

### Phase 2: Data

```
□ เก็บข้อมูล training (text, code, etc.)
□ Clean data (remove noise, duplicates)
□ Tokenize data
□ Split train/validation/test
□ Format สำหรับ training framework
```

### Phase 3: Infrastructure

```
□ Hardware selection (GPU/TPU/RAM/Storage)
□ Software stack (OS, CUDA, Python, frameworks)
□ Distributed training setup (ถ้า model > 10B)
□ Monitoring & logging
□ Backup strategy
```

### Phase 4: Training

```
□ เลือก base model (ถ้า fine-tuning)
□ กำหนด hyperparameters
□ เริ่ม training
□ Monitor loss, gradients, memory
□ Checkpointing
□ Early stopping
```

### Phase 5: Optimization

```
□ Quantization (GPTQ/AWQ/GGUF)
□ Pruning (ลด parameter ที่ไม่จำเป็น)
□ Distillation (สอน model เล็กจาก model ใหญ่)
□ ONNX export (สำหรับ deployment)
□ TensorRT optimization (สำหรับ NVIDIA GPU)
```

### Phase 6: Deployment

```
□ เลือก serving framework (vLLM, TGI, llama.cpp)
□ Load balancing (ถ้า multi-GPU)
□ API endpoint
□ Rate limiting
□ Monitoring & alerting
□ A/B testing
```

---

## สรุป: หลักการสำคัญ

### 1. ขนาด Model = คุณภาพ + ต้นทุน

> "ยิ่ง model ใหญ่ ยิ่งดี แต่ยิ่งแพง" — ต้อง balance ตาม use case

### 2. Quantization คือกุญแจสำคัญ

> "ลด size 75% แต่คุณภาพลดแค่ 5%" — ใช้ Q4_K_M เป็น default

### 3. Hardware ต้อง match Model

> "อย่าเอา 70B ไป run บน phone" — ตรวจสอบ VRAM/RAM ก่อนเสมอ

### 4. Edge vs Cloud = Privacy vs Power

> "ถ้า data ละเอียดอ่อน → Edge; ถ้าต้องการ power → Cloud"

---

## อ้างอิง

1. Vaswani et al. (2017). "Attention Is All You Need" — Original Transformer paper
2. Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
3. Jiang et al. (2023). "Mistral 7B" — efficient smaller model
4. Frantar et al. (2023). "GPTQ: Accurate Post-Training Quantization"
5. Lin et al. (2024). "AWQ: Activation-aware Weight Quantization"
6. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
7. NVIDIA (2024). "TensorRT-LLM Documentation"
8. Hugging Face (2024). "Transformers Documentation"
9. llama.cpp (2024). "GGUF Format Specification"
10. vLLM (2024). "High-throughput LLM serving"

---

**End of Report — Topic 2**
**คำปัน (Kampun) | BOI Family | 2026-08-19**
