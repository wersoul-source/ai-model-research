# บทที่ 2 — โครงสร้างเทคนิครายโมเดล

> หนึ่งโมเดล = หนึ่งปรัชญาออกแบบ อ่าน config.json ของแต่ละตัวแล้วเจอ "จุดตั้งใจ" ชัดเจน
> ผังทั้งหมดอ้างจาก tensor จริงที่ parse แล้ว (ไม่ใช่เอกสาร marketing)

## 2.1 Qwen3-4B — "Thinking Machine"

**จุดตั้งใจ:** ตัวเล็ก (4B) แต่ใส่กลไก "คิดก่อนตอบ" และเทคนิค stabilize training ที่ใหม่ล่าสุด

```
Input: ข้อความ (+ /think /no_think + tools)
  │
  ├─ BBPE Tokenizer (vocab 151,936 · merges 151,387)
  ├─ Chat Template: enable_thinking + tool_call → prompt
  ▼
Language Model — 36 layers, full attention ทุก layer
├─ embed_tokens [151936, 2560] ← tie กับ lm_head (ประหยัด ~256M params)
├─ 36 × Transformer Block:
│   ├─ GQA Attention: 32 Q-heads / 8 KV-heads, head_dim 128
│   │   └─ ★ QK-Norm (q_norm + k_norm) — ของใหม่ Qwen3 ไม่มีใน Qwen2
│   ├─ RMSNorm ×2 (input + post-attention)
│   ├─ SwiGLU MLP: gate/up [9728,2560], down [2560,9728]
│   └─ RoPE θ=1,000,000 + YaRN scaling (32K→128K)
└─ norm [2560] → logits [151936] → softmax
```

**3 เทคนิคเด่น (ที่สองตัวอื่นไม่มี):**

| เทคนิค | ทำอะไร | ทำไมสำคัญ |
|---|---|---|
| **QK-Norm** | normalize query/key ก่อน attention | training ใหญ่ stable ขึ้น — โมเดลเล็ก train ได้คุณภาพสูง |
| **Thinking mode 3 ชั้น** | flag (`enable_thinking`) + token pair (`<think>`/`</think>`) + switch (`/think`) | คุมพฤติกรรม reasoning ได้ runtime ไม่ต้อง retrain |
| **Tie embeddings** | embed_tokens = lm_head tensor เดียวกัน | ประหยัด ~6% params — Llama จ่าย 13% แบบ no-tie |

**ผังอ้างอิง:** `docs/research/qwen3-4B/model-architecture.mmd` · `file-relationships.mmd`

## 2.2 Gemma 4 E4B-it — "Multimodal ประหยัด"

**จุดตั้งใจ:** 8B params แต่ effective ~4.5B — จ่าย memory เท่า 4B ได้ความสามารถ text+ภาพ+เสียง+วิดีโอ

```
Input: ข้อความ + ภาพ + เสียง + วิดีโอ
  │
  ├─ ข้อความ → BPE tokenizer (vocab 262,144 — ใหญ่สุดในสามตัว) → embed [262144, 2560]
  ├─ ภาพ   → image_processor (patch 16) → Vision Tower 16L → embed_vision [2560, 768]
  ├─ เสียง → audio_extractor (16kHz mel 128) → Audio Tower 12L → embed_audio [2560, 1536]
  └─ วิดีโอ → video_processor (32 frames) → Vision Tower → embed_vision
       ▼
Language Model — 42 layers, hybrid attention + PLE
├─ Sliding attention (window 512, RoPE 10k) × 35 layers
├─ Full attention (global, p-RoPE 1M) × 7 layers
├─ PLE: per-layer embeddings (256-dim/layer) — ป้อนทุก layer โดยไม่โต embedding
└─ KV sharing 18/42 layers (42.9%) — ครึ่งหนึ่งแชร์ cache กัน
     ▼
Output logits [262144] → softmax → ข้อความ
```

**3 เทคนิคเด่น:**

| เทคนิค | ทำอะไร | ทำไมสำคัญ |
|---|---|---|
| **Hybrid attention 35:7** | sliding window 512 ส่วนใหญ่ + full attention 7 layer | long-range ยังไปถึงผ่าน 7 global layers — KV cache เล็กลงมหาศาล (ดูบท 5) |
| **PLE (Per-Layer Embeddings)** | ป้อน embedding เข้าทุก layer แยก 256-dim | ความลึกความรู้คงเดิม แต่ effective params ~4.5B จาก total 8B |
| **KV sharing 18/42** | ครึ่งหนึ่งของ layers แชร์ KV cache | cache โตช้าลง — จุดต่างสำคัญสุดเรื่อง context cost |

**ผังอ้างอิง:** `docs/research/gemma-4-E4B/model-architecture.mmd` · `file-relationships.mmd`

## 2.3 Llama 3.1 8B Instruct — "Textbook Transformer"

**จุดตั้งใจ:** ไม่เพิ่มของแปลก — 9 tensors/layer มาตรฐานสุด แล้วใช้ budget ไปกับ **data 15T tokens** และ context extension

```
Input: ข้อความ (+ tools ผ่าน python_tag/header tokens)
  │
  ├─ BPE Tokenizer (vocab 128,000 base + 256 added = 128,256)
  ├─ Chat Template: header ids (<|start_header_id|>) ฝังใน tokenizer
  ▼
Language Model — 32 layers, full attention ทุก layer
├─ embed_tokens [128256, 4096] ←★ แยกจาก lm_head (tie=false)
├─ 32 × Transformer Block:
│   ├─ GQA Attention: 32 Q / 8 KV, head_dim 128 — ไม่มี QK-Norm
│   ├─ RMSNorm ×2
│   ├─ SwiGLU MLP: gate/up [14336,4096], down [4096,14336]
│   └─ RoPE θ=500,000 + llama3 scaling (factor 8: 8K→128K)
├─ norm [4096]
└─ lm_head [128256, 4096] ←★ แยก — 584M params พิเศษ
```

**3 ลักษณะเด่น:**

| ลักษณะ | รายละเอียด | ผล |
|---|---|---|
| **เรียบง่ายสุด** | 9 tensors/layer — ไม่มี QK-Norm/PLE/sliding | baseline เรียนรู้ architecture ที่ดีที่สุด ecosystem รองรับทุก tool |
| **No-tie จ่ายแพง** | embed + lm_head แยก = 1,050M params = **13% ของโมเดล** | แลก flexibility ตอน fine-tune — Meta เลือกจ่าย |
| **EOS ซ้อน 3 ระดับ** | end_of_text (base) + eom_id (mid-turn) + eot_id (chat turn) | คุมการจบ turn ละเอียดสุดในสามตัว |

**ผังอ้างอิง:** `docs/research/llama3.1-8B/model-architecture.mmd` · `llama-vs-qwen.mmd`

## 2.4 DeepSeek V3 — "MoE ยักษ์ประหยัด"

**จุดตั้งใจ:** 671B แต่จ่าย 37B/โทเคน — แยก capacity จาก compute ด้วย MoE + แยก storage จาก compute ด้วย MLA + แยก communication จาก compute ด้วย FP8/DualPipe

```
Input: ข้อความ (129280 vocab → 7168)
  │
  ├─ BPE Tokenizer (vocab 129,280) → embed [129280, 7168]
  ▼
Language Model — 61 layers
├─ Layers 0-2: Dense FFN 18432 (full) — ให้ base stability
├─ Layers 3-60: MLA (128 heads, dc=512, dc'=1536, rope 64) + DeepSeekMoE ×58
│   ├─ MLA: c^{KV} 512 + k^R 64 cache (-93% vs MHA) + RMSNorm หลัง latent
│   └─ MoE: 1 shared + 256 routed (2048 each), top-8, group 8→4, scale 2.5
└─ MTP Layer 61: Emb+OutHead+TRM (14B) — predict token t+2
      ▼
Output [129280] → softmax — FP8 weights, BF16 ต้อง convert
```

**4 เทคนิคเด่น (ไม่มีใน 3 ตัวเล็ก):**

| เทคนิค | ทำอะไร | ทำไมสำคัญ |
|---|---|---|
| **MLA 512+64** | cache latent 512+64 แทน k/v 32768 dim | KV -93% → 128K batch ได้ |
| **256 fine-grained + 1 shared** | 256 experts เล็กๆ เลือก 8 + shared ตายตัว 1 | capacity 671B แต่ active 37B |
| **Aux-loss-free (bias γ0.001)** | bias update แทน aux loss | balance ไม่ทำลาย language loss |
| **FP8+DualePipe+M=4** | FP8 W8A8 + bidirectional pipeline + ≤4 nodes/token | train 14.8T ด้วย 2.788M H800 ไม่ spike |

## 2.5 สรุปเทียบโครงสร้าง — ตารางเดียวจบ (4 โมเดล)

| มิติ | Qwen3-4B | Gemma 4 E4B-it | Llama 3.1 8B | **DeepSeek V3** |
|---|---|---|---|
| Layers | 36 (full) | 42 (35 sliding+7 full) | 32 (full) | **61 (3 dense +58 MoE)** |
| Hidden size | 2,560 | 2,560 | 4,096 | **7,168** |
| Attention | GQA 32Q/8KV + QK-Norm | hybrid + KV share | GQA 32Q/8KV | **MLA 128H dc512/dc'1536 rope64** |
| Vocab | 151,936 | **262,144** | 128,256 | 129,280 |
| Embedding tie | ✓ tie | PLE | ✗ no-tie | ✗ no-tie |
| Multimodal | ✗ | ✓ | ✗ | ✗ (text only) |
| MoE | ✗ dense | ✗ dense | ✗ dense | **✓ 256+1 top8** |
| KV cache @128K | ~18GB | ~3.8GB | ~16GB | **~576 dim/token (MLA) — ถูกสุดถ้านับต่อ param** |
| Thinking/MTP | ✓ thinking 3 ชั้น | ✗ | ✗ | **MTP D=1 (predict 2 tokens)** |

---

*ไปต่อ: บทที่ 3 — น้ำหนัก & 6 รูปแบบไฟล์*
