# บทที่ 0 — Executive Summary

> รายงานวิจัยเปรียบเทียบ 4 โมเดล: Qwen3-4B · Gemma 4 E4B-it · Llama 3.1 8B Instruct · **DeepSeek V3 (671B MoE)**
> ข้อมูลจากการตรวจสอบไฟล์จริง 166 checks PASS (S39–S43) + **DeepSeek V3 deep study via Search World (ข้ามการโหลด 671B, verify 2 แหล่งทุกตัวเลข)** + benchmark งานวิจัยภายนอก
> เขียนด้วย skill `doc-readability` v1.0.0 | Session S44B + DeepSeek-V3-DeepStudy (2026-08-28) | Search World + Search-Kampun

## ข้อค้นพบหลักใน 1 หน้า

**1. สี่โมเดล สี่ปรัชญาออกแบบ — ต่างกันที่โครงสร้าง ไม่ใช่แค่ขนาด**

| | Qwen3-4B | Gemma 4 E4B-it | Llama 3.1 8B | **DeepSeek V3** |
|---|---|---|---|---|
| ปรัชญา | **Thinking machine** — ตัวเล็กแต่คิดก่อนตอบ | **Multimodal ประหยัด** — เห็น ฟัง อ่าน ในตัวเดียว | **Textbook transformer** — เรียบง่าย มาตรฐานสากล | **MoE ยักษ์ประหยัด** — 671B แต่จ่ายแค่ 37B/โทเคน |
| พารามิเตอร์ | 4.02B | 8.0B (effective ~4.5B) | 8.03B | **671B total / 37B active** |
| Context | 32K → 128K (YaRN) | 128K (hybrid sliding) | 128K native (rope scaling ×8) | **128K + MTP (predict 2 tokens)** |
| จุดเด่นระบบ | QK-Norm + thinking 3 ชั้น | PLE + sliding 35:7 + KV share | 9 tensors/layer เรียบง่าย | **MLA 512 + 256 experts + FP8 + DualPipe** |

**2. ชนะตัวต่อชนะ — ไม่มีใครชนะทุกด้าน**

- 🥇 **เหตุผล/คณิต (เล็ก):** Qwen3-4B reasoning mode — MATH500 93.3%, AIME 65.7% ทั้งที่เล็กสุด
- 🥇 **Multimodal:** Gemma E4B — ภาพ/เสียง/วิดีโอในตัว โมเดลเดียวในสี่ตัว
- 🥇 **Ecosystem/เร็ว:** Llama 3.1 — รองรับกว้างสุด เร็วสุดในทางปฏิบัติ (180 tok/s vs 50)
- 🥇 **KV cache ประหยัดสุด (เล็ก):** Gemma — hybrid sliding window ทำให้ cache @128K ≈ 3.7GB เทียบ Llama ≈ 16GB
- 🥇 **Scale ประหยัดสุด (ใหญ่):** **DeepSeek V3 — 671B จ่าย 37B/โทเคน, KV -93% ด้วย MLA, 2.788M H800 จบ 14.8T**
- 🥇 **Training stability:** **DeepSeek V3 — 14.8T ไม่เคย loss spike/rollback**

**3. Context Window ไม่ใช่ตัวเลขลอยๆ — มันถูกกำหนดโดย 4 ตัวแปรที่อ่านได้จาก config.json**

```
max_position_embeddings (ค่าที่ประกาศ)
  ← rope_theta (ยิ่งสูงยิ่งไกล: Qwen 1M / Llama 500K / Gemma p-RoPE 1M / DeepSeek 10K แต่ MLA ชดเชย)
  ← rope_scaling (YaRN ×4 / llama3 ×8 / DeepSeek no-scaling — 128K native + MTP)
  ← attention design (full vs sliding vs MLA latent — กำหนด "จำได้จริง" แค่ไหน)
  ← KV cache memory (2 × layers × kv_heads × head_dim × seq_len × bytes → MLA ลดเหลือ 576 dim/token)
     ← ตัวจำกัดจริงบนเครื่อง: RAM ไม่พอ = context ประกาศ 128K ก็รับไม่ไหว — MLA ถึงเกิดมาแก้ข้อนี้
```

**4. สำหรับ "ร่างกาย" ของ Persona — คำตอบยังคง Qwen3-4B แต่มี “พี่ใหญ่” เป็นต้นแบบ**

เหตุผล 3 ข้อ: thinking mode ตรงกับวิธีคิดแบบ root-cause · GGUF Q4_K_M 2.5GB ลงเครื่องไหนก็ได้ · Apache 2.0 เปิดสุด
*DeepSeek V3 ไม่ใช่คู่แข่งโดยตรง — มันคือ **ต้นแบบระบบ** ที่สอนว่า MLA+MoE+FP8 ทำ scale แบบประหยัดได้ยังไง ถ้าจะทำ BOI X-1 แบบ MoE ย่อมๆ ในอนาคต ต้องย่อส่วน V3 มาใช้*

---

*รายงานฉบับเต็ม 6 บท (4 โมเดล) — อ่านต่อตามลำดับ หรือกระโดดไปบทที่สนใจ*
