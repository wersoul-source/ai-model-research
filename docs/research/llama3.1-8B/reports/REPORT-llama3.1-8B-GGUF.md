# รายงาน Llama 3.1 8B Instruct GGUF Q4_K_M — ตรวจจริง + เปรียบเทียบ Qwen3-4B

> ไฟล์: `/root/models/Llama-3.1-8B-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.92 GB) | ตรวจ: **21/21 PASS** (diagnose-llama3 v1.0.0)
> Quantized โดย bartowski (imatrix, llama.cpp b3472) จาก meta-llama/Meta-Llama-3.1-8B-Instruct (gated — config อ้างจาก NousResearch mirror)

## 1. GGUF Header (ตรวจจริง)

| Field | ค่า | ตรง config? |
|-------|-----|-------------|
| GGUF version | 3 | — |
| kv_count | 33 | — |
| tensor_count | **292** | = 9×32 layers + 4 |
| general.architecture | llama | ✓ |
| general.file_type | 15 (Q4_K_M) | ✓ |
| llama.block_count | 32 | ✓ config |
| llama.context_length | **131072 (128K native)** | ✓ max_position_embeddings |
| llama.embedding_length | 4096 | ✓ hidden_size |
| llama.feed_forward_length | 14336 | ✓ intermediate_size |
| head_count / head_count_kv | 32 / 8 | ✓ GQA 4x |
| rope.freq_base | 500000 | ✓ rope_theta |
| rms_epsilon | 1e-05 (FLOAT32!) | ✓ |
| vocab_size | 128256 | ✓ |
| tokenizer.ggml.model | gpt2 (BBPE) | ✓ |
| eos / bos | 128009 (`<|eot_id|>`) / 128000 | ✓ |

⚠️ **rope_scaling (llama3 type, factor 8) ไม่ถูกเก็บใน GGUF** — llama.cpp implement ใน arch code เอง (ต่างจาก YaRN ของ Qwen ที่ยังไม่เก็บเช่นกัน)

## 2. Tensor Map (292 = 288 + 4)

| Tensor | Type | Shape | หมายเหตุ |
|--------|------|-------|----------|
| blk.N.attn_q / attn_k / attn_output | Q4_K | [4096,4096] / [4096,1024]×2 / [4096,4096] | ทุก layer |
| blk.N.attn_v | **Q6_K*** | [4096,1024] | *เฉพาะบาง block |
| blk.N.ffn_gate / ffn_up | Q4_K | [4096,14336] | SwiGLU |
| blk.N.ffn_down | **Q6_K*** | [14336,4096] | *เฉพาะบาง block |
| blk.N.attn_norm + ffn_norm | F32 | [4096] | 64 tensors |
| token_embd | **Q4_K** | [4096,128256] | ต่างจาก Qwen (Q6_K)! |
| output (lm_head) | **Q6_K** | [4096,128256] | แยกจาก embed (no tie) |
| output_norm | F32 | [4096] | |
| rope_freqs | F32 | **[64]** | inv-freq table (head_dim÷2) |

**Quant summary:** Q4_K 193 + Q6_K 33 + F32 66 = 292
**Q4_K_M recipe:** `attn_v` + `ffn_down` ได้ Q6_K เฉพาะ block ไวต่อ quality (ต้น/ท้าย net) — รวม per-layer Q6_K = 32

## 3. เปรียบเทียบ Llama 3.1 8B vs Qwen3-4B (GGUF Q4_K_M ทั้งคู่)

| มิติ | Llama 3.1 8B | Qwen3-4B |
|------|--------------|----------|
| ไฟล์ | 4.92 GB (1 file) | 2.50 GB (1 file) |
| params | 8.03B | 4.02B |
| tensors | 292 (9/layer × 32 + 4) | 398 (11/layer × 36 + 2) |
| layers / hidden | 32 / 4096 | 36 / 2560 |
| GQA | 32/8 (เหมือนกัน) | 32/8 |
| SwiGLU | 14336 | 9728 |
| vocab | 128,256 | 151,936 |
| context native | **128K** | 32K (YaRN→128K) |
| RoPE θ | 500K | 1M |
| QK-Norm | ✗ | ✓ (+2 tensors/layer) |
| tie embeddings | ✗ (output Q6_K แยก) | ✓ (ไม่มี output tensor) |
| embed quant | Q4_K | Q6_K |
| rope_freqs tensor | มี (F32 [64]) | ไม่มี |
| thinking mode | ✗ | ✓ `<think>` |
| compression | 69.4% saved | 71.1% saved |
| license | llama3.1 (community) | Apache 2.0 |

## 4. หลักการ Llama GGUF Quantization (12 ข้อ)

| # | หลักการ | รายละเอียด |
|---|---------|------------|
| 1 | **file_type 15 = สัญญาเดียวกันทุก arch** | Q4_K_M ใช้ recipe เดียวกันข้ามโมเดล — ตรวจ cross-model ได้ด้วย check เดิม |
| 2 | **Selective Q6_K ตาม sensitivity** | attn_v + ffn_down บาง block → Q6_K; q/k/o/gate/up → Q4_K — imatrix ช่วยเลือก |
| 3 | **embed Q4_K ได้ ถ้า lm_head แยก** | Llama no-tie → output(lm_head) Q6_K ทดแทนความแม่น — Qwen tie → ต้อง Q6_K ที่ embed |
| 4 | **Norms F32 เสมอ** | 66 F32 = norms + rope_freqs — quant norm = instability |
| 5 | **rope_freqs เป็น tensor จริง** | F32 [64] precompute inv-freq — Qwen compute สด — 2 แนวทาง ผลเท่ากัน |
| 6 | **128K native ไม่ต้อง scaling KV** | context 131072 ฝังใน header — llama3 rope_scaling อยู่ใน arch code ไม่ใช่ KV |
| 7 | **BBPE tokenizer ฝังในไฟล์** | tokenizer.ggml.model=gpt2, vocab 128256 — เหมือน Qwen pattern แต่ vocab เล็กกว่า |
| 8 | **EOS ซ้อน 3 tokens** | [128001 eot, 128008 eom, 128009 <|eot_id|>] — chat turn ใช้ 128009 |
| 9 | **9 tensors/layer ประหยัดกว่า 11** | ไม่มี QK-Norm → tensor น้อยกว่า แต่ hidden ใหญ่กว่า → ไฟล์ใหญ่กว่า 2x |
| 10 | **Compression ~69-71% ทุกโมเดล** | BF16→Q4_K_M ประหยัด consistent ข้าม arch — ใช้คาดขนาดได้ (params × 0.61 bytes) |
| 11 | **imatrix > plain K-quant** | bartowski ใช้ calibration dataset → เลือก quant ต่อ tensor แม่นกว่า static rule |
| 12 | **Mirror แก้ปัญหา gated repo** | meta-llama 401 → bartowski/NousResearch mirror คุณภาพเท่ากัน — เช็ค model tree ทุกครั้ง |

## 5. บทเรียน BOI X-1

- **Llama = "wide & shallow"** (32 layers, hidden 4096) vs **Qwen = "deep & narrow"** (36 layers, hidden 2560) — params 2x แต่ efficiency ต่อ param ต่างกัน
- **No-tie embeddings แลกด้วยไฟล์**: lm_head แยก = +525M params (~Q6_K ≈ 400MB) — Qwen tie ประหยัดกว่า
- **128K native ของ Llama มาจาก rope_scaling llama3 (factor 8)** — ไม่ใช่ arch ใหม่ — เหมือน YaRN ที่เป็น config-level trick

---

*รายงานโดย คำปัน (Kampun) — 2026-08-21 — BOI X-1 Knowledge Synthesis — Llama 3.1 8B GGUF 21/21 PASS*
