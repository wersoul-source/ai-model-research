# บทที่ 6 — สรุปและข้อเสนอ: โมเดลไหนคือ "ร่างกาย" ที่เหมาะ

> บทปิด: รวมผล 6 มิติ → ข้อเสนอการใช้งานต่อ persona · ปิดท้ายด้วยบทเรียนเชิงหลักการ

## 6.1 สรุปทั้งงานใน 1 ตาราง

| | Qwen3-4B | Gemma 4 E4B-it | Llama 3.1 8B |
|---|---|---|---|
| **ปรัชญา** | Thinking machine | Multimodal ประหยัด | Textbook transformer |
| **Params** | 4.02B | 8.0B (eff ~4.5B) | 8.03B |
| **BF16 / Q4_K_M** | 8.04 / 2.50 GB | 15.99 / (~4–5) GB | 16.06 / 4.92 GB |
| **Context** | 32K→128K YaRN | 128K hybrid sliding | 8K→128K llama3 ×8 |
| **KV cache @128K** | ~18 GB | **~3.8 GB** | ~16 GB |
| **Reasoning** | 🥇 MATH500 93.3, AIME 65.7 | MATH500 74.9 | GSM8K 84.5 |
| **Multimodal** | ✗ | 🥇 text+ภาพ+เสียง+วิดีโอ | ✗ |
| **Speed (ref)** | ~100–150 tok/s* | 50.1 tok/s | 🥇 180.3 tok/s |
| **License** | 🥇 Apache 2.0 | Gemma Terms | Llama Community |

*\*ประมาณจากสัดส่วน params*

## 6.2 ข้อเสนอตามบทบาทการใช้งาน

```
ถ้าต้องการ...                          → เลือก          เพราะ
──────────────────────────────────────────────────────────────────
Agent วิเคราะห์/ให้เหตุผลลึก            → Qwen3-4B       thinking mode + benchmark นำทุกตัว
รันบนเครื่อง RAM น้อย                   → Qwen3-4B Q4    2.5GB + cache ยังจัดการได้
ประมวลภาพ/เสียง/วิดีโอ                  → Gemma E4B      multimodal native ตัวเดียว
RAG เอกสารยาว งบ RAM จำกัด             → Gemma E4B      KV cache @128K ถูกสุด 4 เท่า
Serve production เร็ว/ปริมาณสูง          → Llama 3.1      180 tok/s + ecosystem vLLM
Fine-tune ต่อยอด                       → Llama 3.1      LoRA tooling ใหญ่สุด
เรียนรู้ architecture                   → Llama 3.1      textbook transformer 9 tensors/layer
```

## 6.3 คำตอบพิเศษ: ร่างกายของ Persona คำปัน

**ข้อเสนอ: Qwen3-4B (GGUF Q4_K_M)** — เหตุผล 4 ข้อ เชื่อมกับ identity โดยตรง:

1. **Thinking mode ↔ Wiki-root-case** — คำปันคิด "root cause ก่อนวิธีแก้" — `<think>` block คือการทำงานเดียวกันแบบ explicit: คิดเป็นขั้น แล้วค่อยตอบ
2. **ตัวเล็ก แต่เหตุผลแน่น** — 4B ชนะ 8B ทุก reasoning benchmark = ปรัชญาเดียวกับ evolution system: *คุณภาพหลักการ ชนะ ขนาด*
3. **Apache 2.0** — เปิดสุดในสามตัว ครอบครอง/แก้ไข/แจกจ่ายได้ไม่มีเงื่อนไขแฝง
4. **GGUF 2.5GB** — ลงทุ้กเครื่องใน BOI Family ได้ รวมถึง edge device

**ทางเลือกรอง:** ถ้าภารกิจขยายสู่ vision/audio → Gemma E4B; ถ้าต้อง serve หลาย user → Llama 3.1

## 6.4 บทเรียนเชิงหลักการ (Principles) จากงานวิจัยทั้งหมด

| # | Principle | หลักฐานจากงาน |
|---|---|---|
| 1 | Data quality + training recipe ชนะ parameter count | Qwen 4B > Llama/Gemma 8B ทุก reasoning benchmark |
| 2 | Context window ที่โฆษณา ≠ context ที่ใช้ได้จริง | 128K ของ full-attention ต้องจ่าย RAM 16–18GB |
| 3 | "การคิด" อยู่ที่ interface layer ไม่ใช่ weights | Qwen thinking = template + tokens, toggle ได้ runtime |
| 4 | Memory efficiency มาจาก attention design ไม่ใช่ quantization | Gemma sliding+sharing ลด cache 4 เท่า ก่อนแตะ precision |
| 5 | ฟอร์แมตเก็บ ≠ ฟอร์แมตคิด | rope_freqs: transformers compute vs GGUF precompute — same model |
| 6 | ตรวจของจริงก่อนเชื่อ model card | 166 checks — เจอ mirror junk, index contracts, vocab math ที่ doc ไม่เคยบอก |

---

## ภาคผนวก — แหล่งอ้างอิง

**Primary (ตรวจเอง — repo นี้):**
- `docs/research/qwen3-4B/reports/` — 10 reports (S41)
- `docs/research/gemma-4-E4B/reports/` — 10 reports (S39B–S40)
- `docs/research/llama3.1-8B/reports/` — 3 reports (S43, S43A)

**Secondary (งานวิจัยคนอื่น):**
- llmbase.ai — benchmark harness เดียวกัน: GPQA, MMLU-Pro, MATH500, AIME, LiveCodeBench, throughput
- llm-stats.com — Gemma/Llama comparison, IFEval, GSM8K CoT

**Papers:**
- RoFormer (Su et al., 2021) — RoPE
- YaRN (Peng et al., 2023) — context extension
- Gemma 3 Technical Report (Google, 2025) — sliding window + PLE
- The Llama 3 Herd of Models (Meta, 2024) — rope scaling llama3

---

*จบรายงาน 7 บท (CH0–CH6) — สังเคราะห์โดย คำปัน (Kampun) [Pattern-2] — Session S44B — BOI X-1 Knowledge Synthesis*
