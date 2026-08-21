# บทที่ 0 — Executive Summary

> รายงานวิจัยเปรียบเทียบ 3 โมเดล: Qwen3-4B · Gemma 4 E4B-it · Llama 3.1 8B Instruct
> ข้อมูลจากการตรวจสอบไฟล์จริง 132 checks PASS (S39–S43) + benchmark งานวิจัยภายนอก
> เขียนด้วย skill `doc-readability` v1.0.0 | Session S44B

## ข้อค้นพบหลักใน 1 หน้า

**1. สามโมเดล สามปรัชญาออกแบบ — ต่างกันที่โครงสร้าง ไม่ใช่แค่ขนาด**

| | Qwen3-4B | Gemma 4 E4B-it | Llama 3.1 8B |
|---|---|---|---|
| ปรัชญา | **Thinking machine** — ตัวเล็กแต่คิดก่อนตอบ | **Multimodal ประหยัด** — เห็น ฟัง อ่าน ในตัวเดียว | **Textbook transformer** — เรียบง่าย มาตรฐานสากล |
| พารามิเตอร์ | 4.02B | 8.0B (effective ~4.5B) | 8.03B |
| Context | 32K → 128K (YaRN) | 128K (hybrid sliding) | 128K native (rope scaling ×8) |

**2. ชนะตัวต่อชนะ — ไม่มีใครชนะทุกด้าน**

- 🥇 **เหตุผล/คณิต:** Qwen3-4B reasoning mode — MATH500 93.3%, AIME 65.7% ทั้งที่เล็กสุด
- 🥇 **Multimodal:** Gemma E4B — ภาพ/เสียง/วิดีโอในตัว โมเดลเดียวในสามตัว
- 🥇 **Ecosystem:** Llama 3.1 — รองรับกว้างสุด เร็วสุดในทางปฏิบัติ (180 tok/s vs 50)
- 🥇 **KV cache ประหยัดสุด:** Gemma — hybrid sliding window ทำให้ cache @128K ≈ 3.7GB เทียบ Llama ≈ 16GB

**3. Context Window ไม่ใช่ตัวเลขลอยๆ — มันถูกกำหนดโดย 4 ตัวแปรที่อ่านได้จาก config.json**

```
max_position_embeddings (ค่าที่ประกาศ)
  ← rope_theta (ยิ่งสูงยิ่งไกล: Qwen 1M / Llama 500K / Gemma p-RoPE 1M)
  ← rope_scaling (YaRN ×4 / llama3 ×8 — ขยายโดยไม่ retrain)
  ← attention design (full vs sliding window — กำหนด "จำได้จริง" แค่ไหน)
  ← KV cache memory (2 × layers × kv_heads × head_dim × seq_len × bytes)
     ← ตัวจำกัดจริงบนเครื่อง: RAM ไม่พอ = context ประกาศ 128K ก็รับไม่ไหว
```

**4. สำหรับ "ร่างกาย" ของ Persona — คำตอบคือ Qwen3-4B**
เหตุผล 3 ข้อ: thinking mode ตรงกับวิธีคิดแบบ root-cause · GGUF Q4_K_M 2.5GB ลงเครื่องไหนก็ได้ · Apache 2.0 เปิดสุด

---

*รายงานฉบับเต็ม 6 บท — อ่านต่อตามลำดับ หรือกระโดดไปบทที่สนใจ*
