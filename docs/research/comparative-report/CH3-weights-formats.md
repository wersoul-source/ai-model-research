# บทที่ 3 — น้ำหนักและ 6 รูปแบบไฟล์

> โมเดลเดียวกัน ขนาดต่างกันได้ 3 เท่า — สิ่งที่เปลี่ยนคือ "วิธีจัดเก็บ" ไม่ใช่ "ความฉลาด"
> ทุกตัวเลขในบทนี้ = ขนาดจริงบนดิสก์ที่ parse แล้ว

## 3.1 สองฟอร์แมตหลัก — คนละโลก คนละ ecosystem

| มิติ | Safetensors (original) | GGUF (quantized) |
|---|---|---|
| dtype | BF16 100% (ไม่ผสม) | Q4_K / Q6_K / F32 ผสมตาม tensor |
| การกระจาย | sharded + index.json (contract) | single file |
| Ecosystem | transformers, vLLM, research | llama.cpp, Ollama, LM Studio |
| ความปลอดภัย | header-only format ไม่รันโค้ด | metadata map ชัดเจน |
| ใช้เมื่อ | fine-tune, serve production, ศึกษา architecture | รันบนเครื่องทรงพลังต่ำ / local inference |

## 3.2 ตารางขนาดจริง — 6 รูปแบบที่ตรวจแล้ว

| # | โมเดล × ฟอร์แมต | ขนาดจริง | Tensors | หมายเหตุ |
|---|---|---|---|---|
| 1 | Qwen3-4B safetensors BF16 | **8.04 GB** | 398 (3 shards: 174+219+5) | tie embeddings |
| 2 | Qwen3-4B GGUF Q4_K_M | **2.50 GB** | 398 (216 Q4_K + 37 Q6_K + 145 F32) | ประหยัด **71%** |
| 3 | Gemma 4 E4B-it safetensors BF16 | **15.99 GB** | 2,130 (single file!) | total 8B + PLE + vision/audio towers |
| 4 | Llama 3.1 8B safetensors BF16 | **16.06 GB** | 291 (4 shards: 82+104+100+5) | no-tie → embed+lm_head แยก |
| 5 | Llama 3.1 8B GGUF Q4_K_M | **4.92 GB** | 292 (+rope_freqs F32 [64]) | ประหยัด **69.4%** |
| 6 | Gemma GGUF (community) | ~4–5 GB (ประมาณการ) | — | ยังไม่ได้ตรวจ — ดูข้อจำกัด CH1 |

## 3.3 สามข้อสังเกตจากการ quantize จริง

**1. Quantization ไม่แตะทุก tensor เท่ากัน**
GGUF Q4_K_M เก็บ attention/MLP weights ที่ Q4_K (4-bit) แต่:
- `output.weight` (lm_head) → Q6_K — เพราะ logits ต้องละเอียด
- norm tensors + rope_freqs → F32 — เพราะเล็กและ sensitive
→ นี่คือเหตุผลที่ "Q4" จริงๆ ไม่ใช่ 4-bit ทุก byte

**2. Tensor count อาจไม่เท่ากันข้ามฟอร์แมต — และถูกต้อง**
Llama: safetensors 291 tensors vs GGUF 292 tensors
→ GGUF **precompute rope_freqs เป็น tensor F32 [64]** ส่วน transformers compute runtime
→ same model, different persistence philosophy — ไม่ใช่ corruption

**3. Sharding มีเหตุผลเชิงปฏิบัติ**
Qwen/Llama แบ่ง shards ~4–5GB — resume download ง่าย โหลด parallel ได้
Gemma เลือก single file 16GB — ง่ายต่อ HF cache แต่ resume เจ็บ
→ shard สุดท้ายของ Llama เล็กพิเศษ (1.2GB) เพราะเก็บ embedding bookends (embed+lm_head+norm)

## 3.4 เลือกฟอร์แมตยังไง — decision table

```
ถ้าต้องการ...                    → ใช้
─────────────────────────────────────────────
รันบน RAM 8GB / laptop           → GGUF Q4_K_M
Fine-tune / LoRA                 → Safetensors original
Serve หลาย user (vLLM)           → Safetensors original
ศึกษา architecture               → Safetensors (parse header ได้)
แจกจ่ายให้ community             → ทั้งคู่ (HF repo มาตรฐาน)
```

---

*ไปต่อ: บทที่ 4 — เปรียบเทียบ 6 มิติ*
