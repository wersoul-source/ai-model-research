# รายงาน Llama 3.1 8B Instruct — Original Safetensors (ตรวจจริง)

> ไฟล์: `/root/models/Llama-3.1-8B/` (10 ไฟล์, 16.06 GB) | ตรวจ: **39/39 PASS** (diagnose-llama3 v1.1.0 safetensors)
> แหล่ง: NousResearch mirror (meta-llama gated) | เทียบคู่กับ GGUF Q4_K_M ที่ตรวจแล้ว (S43, 21/21 PASS)

## 1. โครงสร้างไฟล์ (10 ไฟล์)

| ไฟล์ | ขนาด | เนื้อหา |
|------|------|---------|
| model-00001-of-00004.safetensors | 5.0G | 82 tensors / 2.49B params |
| model-00002-of-00004.safetensors | 5.0G | 104 tensors / 2.50B params |
| model-00003-of-00004.safetensors | 4.9G | 100 tensors / 2.46B params |
| model-00004-of-00004.safetensors | 1.2G | 5 tensors / 584M (embed+lm_head+norm) |
| model.safetensors.index.json | 24K | weight_map 291 tensors, total_size 16,060,522,496 |
| config.json | 855 B | พิมพ์เขียวทั้งโมเดล |
| generation_config.json | 184 B | eos [128001,128008,128009] |
| tokenizer.json | 8.7M | BPE vocab 128,000 + merges 280,147 + added 256 |
| tokenizer_config.json | 50K | PreTrainedTokenizerFast |
| special_tokens_map.json | 296 B | bos/eos |

## 2. Tensor Map (291 = 288 + 3)

| กลุ่ม | จำนวน | Shape | dtype |
|-------|--------|-------|-------|
| ต่อ layer × 32 | 288 | q/o [4096,4096], k/v [1024,4096], gate/up [14336,4096], down [4096,14336], norms ×2 [4096] | BF16 100% |
| model.embed_tokens.weight | 1 | [128256, 4096] | BF16 |
| lm_head.weight (**แยก** tie=false) | 1 | [128256, 4096] | BF16 |
| model.norm.weight | 1 | [4096] | BF16 |

**Total params: 8,030,261,248 = 8.03B** ✓

## 3. Config 15/15 PASS

model_type llama · layers 32 · hidden 4096 · GQA 32/8 (head_dim 128) · SwiGLU 14336 · vocab 128256 · rope θ=500K · context 131072 (128K native) · tie=false · rms_eps 1e-05 · rope_scaling llama3 factor 8.0 (orig_max 8192 → 128K)

## 4. Tokenizer Math

```
128,000 base vocab + 256 added_tokens = 128,256 = config.vocab_size ✓
merges 280,147
Key tokens: <|begin_of_text|>=128000 · <|end_of_text|>=128001
            <|eot_id|>=128009 · <|eom_id|>=128008 · <|python_tag|>=128010
            <|start_header_id|>=128006 · <|finetune_right_pad_id|>=128004
reserved_special_token 0-24 (128002-128027 บางส่วน)
```

## 5. Safetensors vs GGUF (291 vs 292 tensors)

| มิติ | Safetensors (BF16) | GGUF Q4_K_M |
|------|--------------------|--------------|
| tensors | **291** | **292** (+rope_freqs F32 [64]) |
| ขนาด | 16.06 GB | 4.92 GB (**69.4% saved**) |
| dtype | BF16 100% | Q4_K 193 + Q6_K 33 + F32 66 |
| lm_head | `lm_head.weight` BF16 | `output.weight` Q6_K |
| embed | `model.embed_tokens.weight` BF16 | `token_embd.weight` Q4_K |
| rope_freqs | ❌ compute runtime | ✓ precompute buffer |
| sharding | 4 shards + index.json | 1 file |
| ecosystem | transformers/vLLM | llama.cpp/Ollama/LM Studio |

## 6. หลักการ Llama Original (12 ข้อ)

| # | หลักการ | รายละเอียด |
|---|---------|------------|
| 1 | **index.json คือสัญญา** | weight_map 291 entries ↔ shard headers ตรงเป๊ะ (missing/extra/wrong-shard = 0) |
| 2 | **BF16 100% ไม่ผสม** | ต่างจากบางโมเดลที่ norm เป็น F32 — Meta เก็บ uniform BF16 ทั้งหมด |
| 3 | **No-tie = จ่าย 2 เท่า** | embed + lm_head แยก = 1,050M params (13% ของโมเดล!) — Qwen tie ประหยัดครึ่ง |
| 4 | **9 tensors/layer มาตรฐาน** | ไม่มี QK-Norm → โครงสร้าง layer เรียบง่ายสุดในบรรดา modern LLM |
| 5 | **RoPE scaling อยู่ใน config ไม่ใช่ weights** | llama3 type (factor 8, low/high_freq_factor) — ขยาย 8K→128K แก้แค่ config |
| 6 | **rope_freqs ไม่ถูก persist** | transformers compute inv_freq runtime — GGUF เลือก precompute เป็น tensor |
| 7 | **Shard 4 เล็กพิเศษ (584M)** | เก็บ embedding bookends (embed+lm_head+norm) — แชร์ layout ตามการ train/save |
| 8 | **EOS ซ้อน 3 ระดับ** | end_of_text (base) + eom_id (mid-turn) + eot_id (chat turn) — generation_config รวมหมด |
| 9 | **256 added tokens ครบชุด chat/tool** | header ids, python_tag, pad, reserved — ฝังใน tokenizer.json ไม่ใช่ code |
| 10 | **vocab math ต้องเท่า config** | base+added=128256 — ถ้าไม่เท่า = tokenizer/config mismatch แน่นอน |
| 11 | **Mirror ต้องเช็ค junk** | NousResearch มี original/*.pth 16.1G ซ้ำซ้อน — --include ให้ชัดเจน |
| 12 | **HF cache copy trap** | cp จาก cache = พื้นที่ 2 เท่าชั่วขณะ — disk จอง → ใช้ mv blob (rename) แทน |

## 7. บทเรียน BOI X-1

- **Llama 3.1 = "textbook transformer"** — 9 tensors/layer, no extras — เหมาะเป็น baseline เรียนรู้ architecture
- **Qwen3 เพิ่ม 3 อย่าง**: QK-Norm (+2 tensors/layer), tie embeddings (-1 tensor), thinking mode (template-level)
- **128K context ไม่ฟรี** — มาจาก rope_scaling trick ไม่ใช่ arch ใหม่ — KV cache ที่ 128K ยังกิน RAM มหาศาล (32 layers × 8 KV heads × 128 dim × 2 × 131072 tokens ≈ 16 GB FP16!)

---

*รายงานโดย คำปัน (Kampun) — 2026-08-21 — BOI X-1 Knowledge Synthesis — Llama 3.1 8B Original 39/39 PASS*
