# บทที่ 5 — Context Window Deep Dive: อะไรกำหนดว่าโมเดล "รับไหว"

> คำถามหลักของบท: **"จุดที่ทำให้เกิดปริมาณ Context Window — อะไรคือตัวบอกสิ่งนั้น?"**
> คำตอบสั้น: ไม่มี field เดียว — มี **4 ชั้น** อ่านไล่จาก config.json ลงไปถึง RAM ของเครื่อง

## 5.1 แผนที่ 4 ชั้น — อ่านจากบนลงล่าง

```
ชั้น 1  ประกาศ      max_position_embeddings   ← ตัวเลขที่ model card โฆษณา
ชั้น 2  กลไก RoPE    rope_theta + rope_scaling ← ตัวบอกว่า "โครงสร้าง" รองรับไกลแค่ไหน
ชั้น 3  การออกแบบ    attention type            ← ตัวบอกว่า "จำได้จริง" หรือ "จำแบบเลือก"
ชั้น 4  ข้อจำกัดจริง  KV cache memory          ← ตัวบอกว่า "เครื่องนี้รับไหว" — จุดที่พังจริง
```

**หลักการ:** ชั้น 1–2 คือ *ความสามารถ* (capability) · ชั้น 3 คือ *คุณภาพการจำ* (quality) · ชั้น 4 คือ *ความเป็นจริง* (reality) — โมเดลรับไหวจริง = min(ทั้ง 4 ชั้น)

## 5.2 ชั้น 1 — ประกาศใน config.json

```json
// Qwen3-4B
"max_position_embeddings": 32768        // native 32K — 128K มาจาก scaling
// Llama 3.1 8B
"max_position_embeddings": 131072       // 128K — แต่ train จริงแค่ 8K, scale ขึ้นมา
// Gemma 4 E4B-it
"text_config": { "max_position_embeddings": 131072 }
```

⚠️ ตัวเลขนี้เป็นแค่ **ป้าย** — บอกว่า inference framework จะยอมรับ input ยาวเท่าไร ไม่ได้การันตีคุณภาพการจำ

## 5.3 ชั้น 2 — RoPE: หัวใจของ "ไกลแค่ไหน"

RoPE (Rotary Position Embedding) encode ตำแหน่ง token ด้วยการหมุน vector — `rope_theta` คือความถี่ฐานของการหมุน

| | Qwen3-4B | Llama 3.1 8B | Gemma 4 E4B |
|---|---|---|---|
| rope_theta | **1,000,000** | 500,000 | 10,000 (sliding) / 1,000,000 (global p-RoPE) |
| Native train length | 32,768 | 8,192 | 131,072 |
| rope_scaling | YaRN factor 4 | llama3 factor 8.0 | ไม่ต้อง (native) |

**หลักการอ่าน:** theta ยิ่งสูง → ความถี่หมุนช้าลง → แยกตำแหน่งไกลๆ ออกจากกันได้ → extend context ได้โดยไม่เสีย precision

**rope_scaling คือเทคนิค "ยืด" โดยไม่ retrain:**
- **YaRN (Qwen):** 32K × 4 = 128K — interpolate ความถี่แบบ adaptive ต่อ dimension
- **llama3 (Llama):** 8K × 8 = 128K — แบ่ง frequency bands: low-freq ยืดเต็มที่ high-freq คงเดิม (รักษา local order)

→ **ตัวบอกชั้นนี้:** `rope_theta` (เพดานเชิงโครงสร้าง) + `rope_scaling` (วิธียืดจาก native)

## 5.4 ชั้น 3 — Attention Design: "จำได้จริง" หรือ "จำแบบเลือก"

```
Full attention (Qwen, Llama):
  token ที่ N เห็น token 0…N-1 ทั้งหมด → จำทุกอย่าง แต่ KV cache โตเชิงเส้นตรง

Hybrid sliding window (Gemma):
  35/42 layers เห็นแค่ 512 tokens ล่าสุด (local detail)
  7/42 layers เห็นทั้งหมด (long-range gist)
  → long document ยังเข้าใจภาพรวม แต่ "quote คำเป๊ะๆ จากต้นเอกสาร" อ่อนกว่า
```

→ **ตัวบอกชั้นนี้:** ประเภท attention ต่อ layer (config: `sliding_window`, `layer_types`) — สองโมเดลประกาศ 128K เท่ากันแต่ *คุณภาพการจำต่างกัน*

## 5.5 ชั้น 4 — KV Cache: ข้อจำกัดจริงบนเครื่อง (จุดที่พังจริง)

**สูตร (จำไว้ได้ใช้ทุกโมเดล GQA):**

```
KV cache = seq_len × layers × 2(K+V) × kv_heads × head_dim × bytes_per_element

GQA มาตรฐาน (kv_heads 8, head_dim 128, BF16=2 bytes):
= seq_len × layers × 4 KB/token
```

**ตัวเลขจริง @128K context:**

| | Layers | Cache/token | Cache @128K | เทียบ weights |
|---|---|---|---|---|
| Llama 3.1 8B | 32 | 128 KB | **~16.0 GB** | = ขนาด weights เป๊ะ! |
| Qwen3-4B | 36 | 144 KB | **~18.0 GB** | > weights 2 เท่า |
| Gemma 4 E4B | 42 (35 sliding) | ~28 KB เฉลี่ย | **~3.8 GB** | < weights 4 เท่า |

**ข้อค้นพบเชิง root cause:**

1. **Context 128K ของ Llama/Qwen "ฟรี" แค่บนกระดาษ** — ตอน runtime ต้องจ่าย RAM เท่าหรือมากกว่าตัว weights ทั้งโมเดล
2. **Gemma ออกแบบมาเพื่อ context ยาวจริง** — sliding window ตัด cache 35 layers เหลือคงที่ 73MB + KV sharing อีกครึ่ง → จ่าย ~3.8GB ได้ 128K
3. **Qwen แพงสุดต่อ token** — full attention 36 layers แม้ params น้อยสุด เพราะ cache แปรผันกับ layer count ไม่ใช่ param count

→ **ตัวบอกชั้นนี้:** `num_hidden_layers × num_key_value_heads × head_dim` — สามค่าใน config คูณกัน = ราคาต่อ token ของ context

## 5.6 สรุปบท — "อะไรคือตัวบอก" ครบทุกชั้น

| คำถาม | ตัวบอก | อยู่ที่ไหน |
|---|---|---|
| รับได้กี่ token? | `max_position_embeddings` | config.json |
| โครงสร้างรองรับไกลแค่ไหน? | `rope_theta` + `rope_scaling.type/factor` | config.json |
| จำได้จริงหรือจำเลือก? | attention types (`sliding_window`, layer_types) | config.json |
| เครื่องนี้รับไหว? | layers × kv_heads × head_dim × 4KB × seq_len | คำนวณเอง — ไม่มีใครบอกคุณตรงนี้ |

> **Principle ของบท:** Context window ที่โฆษณา = capability · Context ที่ใช้ได้จริง = min(capability, quality, RAM) — และตัวที่คำนวณได้ล่วงหน้าจาก config อย่างเดียวคือ RAM

---

*ไปต่อ: บทที่ 6 — สรุปและข้อเสนอ*
