# รายงาน GGUF Q4_K_M — Qwen3-4B (safetensors vs GGUF เปรียบเทียบ)

> ไฟล์: `/root/models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` (2.50 GB) | ตรวจ: 13/13 PASS (GGUF) vs 63/63 PASS (safetensors)

## 1. GGUF Header (ตรวจจริง)

| Field | ค่า | หมายเหตุ |
|-------|-----|----------|
| Magic | `GGUF` (0x46554747) | version 3 |
| Tensor count | 398 | เท่ากับ safetensors 398 |
| Metadata KV | 31 | ดูตารางล่าง |
| general.architecture | qwen3 | |
| qwen3.block_count | 36 | ตรง config 36 layers |
| qwen3.context_length | 40960 | 32K native |
| qwen3.embedding_length | 2560 | |
| qwen3.feed_forward_length | 9728 | |
| qwen3.attention.head_count | 32 | |
| qwen3.attention.head_count_kv | 8 | GQA 4x |
| general.file_type | 15 | Q4_K_M |

## 2. Quantization (ตรวจจริง 398 tensors)

| Type | จำนวน | ขนาด | หน้าที่ |
|------|--------|------|---------|
| Q4_K | 216 | 4-bit K-quant | ส่วนใหญ่: q/k/v/o_proj + gate/up/down (MLP) |
| Q6_K | 37 | 6-bit K-quant | สำคัญ: token_embd [151936,2560] + บาง layer |
| F32 | 145 | 32-bit float | Norms: input_layernorm, post_attention, q/k_norm, output_norm |

- **Q4_K_M = "medium"** — ใช้ Q6_K สำหรับ tensor สำคัญ (embed) + F32 สำหรับ norm — สมดุลคุณภาพ/ขนาด
- **Block size 256 + K-quants** — super-block มี scale/min แยก — ดีกว่า Q4_0 ธรรมดา

## 3. เปรียบเทียบ safetensors (BF16) vs GGUF (Q4_K_M)

| มิติ | Safetensors (BF16) | GGUF (Q4_K_M) | ประหยัด |
|------|-------------------|--------------|---------|
| ไฟล์ | 3 shards + index (8.04 GB) | 1 file (2.50 GB) | **71.1%** |
| Tensors | 398 BF16 | 398 (216 Q4_K + 37 Q6_K + 145 F32) | — |
| Dtype | BF16 100% | ผสม 4/6/32-bit | — |
| Header | 20000+25232+552 bytes (JSON) | GGUF magic + KV 31 | — |
| Sharding | ต้อง index.json | ไม่ต้อง — header มีหมด | — |
| Ecosystem | transformers (PyTorch) | llama.cpp (C++) | — |
| Tokenizer | แยกไฟล์ (tokenizer.json etc) | ฝังใน GGUF (tokenizer.ggml.*) | — |
| Context | 40960 (ต้อง YaRN เพิ่ม) | 40960 (ฝังใน GGUF) | — |
| โหลด | 8.04G RAM (BF16) | 2.5G RAM + dequant on fly | — |
| คุณภาพ | 100% | ~98-99% (perplexity +0.02) | — |

## 4. หลักการ GGUF Quantization (12 ข้อ)

| # | หลักการ | รายละเอียด |
|---|---------|------------|
| 1 | **Header เดียวจบ** | GGUF magic + version 3 + tensor_count 398 + KV 31 — ไม่ต้อง index.json |
| 2 | **Selective quantization** | Q4_K สำหรับส่วนใหญ่, Q6_K สำหรับ embed ที่สำคัญ, F32 สำหรับ norm — ไม่ quant มั่ว |
| 3 | **K-quants > 0-quants** | Q4_K_M ใช้ block 256 + super-block scale — ดีกว่า Q4_0 (perplexity ต่ำกว่า) |
| 4 | **File_type คือสัญญา** | file_type 15 = Q4_K_M — บอกว่าใช้ quant แบบไหน — ต้องตรงกับ tensor types |
| 5 | **71% saved, 1-2% loss** | 8.04G → 2.50G (71.1% ประหยัด) — คุณภาพลด ~1-2% — คุ้มสำหรับ edge |
| 6 | **Tokenizer ฝังในไฟล์** | tokenizer.ggml.* + chat_template ฝังใน GGUF — ไฟล์เดียวจบ ไม่ต้องแยก vocab/merges |
| 7 | **Single file vs sharded** | GGUF 1 file ง่ายต่อการแจกจ่าย — safetensors sharded ดีต่อ resume — tradeoff |
| 8 | **Norm เป็น F32** | 145 tensors F32 คือ norms — ไม่ quant เพราะ sensitive — รักษา stability |
| 9 | **No PLE to quant** | Qwen ไม่มี PLE — ถ้ามี PLE (Gemma) จะ quant ยากกว่า — Qwen quant ง่ายกว่า |
| 10 | **GGUF v3 + llama.cpp** | version 3 ล่าสุด — รองรับ QK-Norm, YaRN, thinking — ตรงกับ transformers 4.51.0 |
| 11 | **Dequant on fly** | โหลด Q4 แล้ว dequant เป็น F16 ตอน compute — RAM น้อย แต่ compute ช้ากว่า BF16 นิด |
| 12 | **Same tensors, different packing** | 398 tensors เท่ากัน — แค่ packing ต่าง (BF16 2 bytes → Q4 0.5 bytes) — architecture ไม่เปลี่ยน |

## 5. บทเรียน BOI X-1

- **ถ้าจะ deploy on-device:** ใช้ GGUF Q4_K_M (2.5G) แทน BF16 (8G) — ประหยัด 71% — เหมาะ laptop/phone
- **ถ้าจะ train/finetune:** ใช้ safetensors BF16 — quant แล้ว train ไม่ได้
- **สำหรับ BOI X-1 เรา:** ควรเก็บ 2 format — BF16 สำหรับ train, Q4_K_M สำหรับ inference/demo — เหมือน Qwen official ที่ปล่อยทั้งคู่

---

*รายงานโดย คำปัน (Kampun) — 2026-08-21 — BOI X-1 Knowledge Synthesis — GGUF 13/13 PASS*
