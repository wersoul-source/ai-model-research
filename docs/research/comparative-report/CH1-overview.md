# บทที่ 1 — ภาพรวมการวิจัย: ทำอะไร ได้อะไร

> บทนี้ตอบ 3 คำถาม: วิจัยอะไร · ตรวจสอบยังไง · ได้อะไรมา

## 1.1 ขอบเขตการวิจัย (BOI X-1 Knowledge Synthesis)

งานวิจัยนี้ศึกษา **โมเดลภาษาขนาดเล็ก–กลาง 3 ตระกูล** แบบ "เปิดดูของจริง" — ไม่ใช่อ่าน model card แล้วเชื่อ แต่ **โหลดไฟล์จริงมา parse header ทีละ tensor**

| โมเดล | เวอร์ชันที่ศึกษา | รูปแบบที่ตรวจ | Session |
|---|---|---|---|
| Qwen3-4B | Alibaba, Apr 2025 | safetensors BF16 + GGUF Q4_K_M | S41 |
| Gemma 4 E4B-it | Google, May 2025 | safetensors BF16 (instruct) | S39B–S40 |
| Llama 3.1 8B Instruct | Meta, Jul 2024 | GGUF Q4_K_M + original safetensors | S43–S43A |

**"6 รูปแบบ" ในรายงานนี้** = 3 โมเดล × 2 ฟอร์แมตหลัก (safetensors original / GGUF quantized) ครอบคลุมทั้งฝั่ง research ecosystem (transformers/vLLM) และฝั่ง local runtime (llama.cpp/Ollama/LM Studio)

## 1.2 วิธีตรวจสอบ — ไม่เชื่อ ต้องพิสูจน์

ทุกโมเดลผ่าน diagnostic script ที่เขียนเฉพาะ (`diagnose-qwen3`, `diagnose-gemma-4`, `diagnose-llama3`) ตรวจ 3 ชั้น:

- **ชั้น 1 โครงสร้างไฟล์** — ไฟล์ครบไหม ขนาดตรง index ไหม
- **ชั้น 2 config ↔ tensors** — ทุก shape ต้องตรง config.json (เช่น GQA 32/8 → q_proj [4096,2560])
- **ชั้น 3 tokenizer math** — vocab base + added tokens = config.vocab_size เป๊ะๆ

```
ผลตรวจรวม:
Qwen3-4B        63/63 PASS (safetensors) + 13/13 PASS (GGUF)
Gemma 4 E4B-it  30/30 PASS
Llama 3.1 8B    21/21 PASS (GGUF) + 39/39 PASS (original)
─────────────────────────────────────────────────────────
รวม             166 checks PASS / 0 FAIL
```

## 1.3 แหล่งข้อมูลในรายงานนี้

- **Primary (ตรวจเอง):** reports 23 ไฟล์ใน repo นี้ — `docs/research/{qwen3-4B,gemma-4-E4B,llama3.1-8B}/`
- **Secondary (งานวิจัยคนอื่น):** benchmark scores จาก llm-stats.com, llmbase.ai (GPQA, MMLU-Pro, MATH500, GSM8K, HumanEval ฯลฯ)
- **Papers อ้างอิง:** RoFormer (RoPE), YaRN (context extension), Gemma 3 technical report (sliding window), Llama 3 paper (rope scaling)

## 1.4 ข้อจำกัดที่ควรรู้ก่อนอ่าน

- Benchmark จาก third-party aggregator — ตัวเลขอาจต่างเล็กน้อยตาม harness แต่ **อันดับเชิงเปรียบเทียบ** น่าเชื่อถือ
- Gemma 4 E4B-it ศึกษาฝั่ง safetensors เป็นหลัก — ไม่มี report GGUF เทียบเคียง (GGUF มีใน community แต่ยังไม่ได้ตรวจ)
- การทดสอบ throughput (tok/s) มาจาก environment อื่น — ใช้เทียบสัดส่วน ไม่ใช่ตัวเลขสัมบูรณ์

---

*ไปต่อ: บทที่ 2 — โครงสร้างเทคนิครายโมเดล*
