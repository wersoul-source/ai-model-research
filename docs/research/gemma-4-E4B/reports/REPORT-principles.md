# หลักการสร้าง — Gemma 4 E4B (base vs it) เปรียบเทียบ

> Session: 2026-08-20-S40 | สังเคราะห์จากไฟล์จริงทั้ง 9 ไฟล์ของ `google/gemma-4-E4B-it` + เทียบกับ base (S39A)
> วัตถุประสงค์: เปรียบเทียบ "หลักการออกแบบ" ของโมเดล base (pre-trained) vs it (instruction-tuned)

---

## 1. หลักการสร้างของ base (12 ข้อ — จาก S39A)

| # | หลักการ | หลักฐานในโมเดล |
|---|---------|----------------|
| 1 | **แยกข้อมูลกับคำอธิบาย** | safetensors header (JSON) แยกจาก data buffer — metadata ไม่ปนน้ำหนัก |
| 2 | **Safety over convenience** | safetensors แทน pickle — ไม่รันโค้ด, header limit 100MB, contiguity check |
| 3 | **Relative offset แก้ chicken-egg** | data_offsets อ้างอิงจาก buffer start — อ่าน tensor เดี่ยวได้โดยไม่รู้ขนาดไฟล์ |
| 4 | **Contiguity = safety** | ห้าม holes ใน data buffer — ป้องกัน polyglot files |
| 5 | **แยกงานหนัก/งานเบา** | hybrid attention: sliding (เบา) × 35 + full (หนัก) × 7 — ประหยัดโดยไม่เสีย long-context |
| 6 | **ใช้ตารางใหญ่แทนการคำนวณ** | PLE: embed_tokens_per_layer [262144, 10752] — lookup แทนการคำนวณเพิ่ม |
| 7 | **แชร์เพื่อประหยัด** | KV sharing 18/42 + k_eq_v (global) + GQA (8 heads → 2 KV) — ลด KV cache 37.5% |
| 8 | **แยกส่วนประกอบรวมที่ปลายทาง** | vision/audio tower แยก → embed_vision/embed_audio project เข้า LM space ที่ปลายทาง |
| 9 | **ควบคุมด้วย token ไม่ใช่โค้ด** | `<|think|>`, `<|image>`, `<|audio>` — modality/reasoning ควบคุมด้วย special tokens |
| 10 | **Variable token budget** | vision soft tokens 70/140/280/560/1120 — ปรับละเอียด vs ความเร็ว |
| 11 | **Context-aware projection** | per_layer_model_projection + RMSNorm — PLE รวม token-identity + context |
| 12 | **p-RoPE สำหรับ long context** | full attention ใช้ proportional RoPE (p=0.25, base 1M) — ยืด context 128K |

---

## 2. หลักการสร้างของ it (12 ข้อ — สังเคราะห์จากไฟล์จริง S40)

| # | หลักการ | หลักฐานในไฟล์ it |
|---|---------|------------------|
| 1 | **แยก "วิธีคิด" ออกจาก "คำตอบ"** | `<|channel>thought\n...<channel|>` — thinking แยก channel จาก content; `strip_thinking()` ใน template ตัด thinking ออกจาก history |
| 2 | **ควบคุมพฤติกรรมด้วย token ไม่ใช่ flag** | `<|think|>` เปิด/ปิด reasoning, `<|tool>`/`<|tool_call>`/`<|tool_response>` ควบคุม tool loop — ทั้งหมดเป็น special tokens ใน vocab (id 46-51) |
| 3 | **Output เป็น JSON contract (schema)** | `response_schema` + `response_template` ใน tokenizer_config — role/thinking/content/tool_calls มีโครงสร้างบังคับ; `parse_response()` อ่านตาม schema |
| 4 | **Multi-turn termination = array** | `eos_token_id` = [1, 106, 50] — จบได้ 3 แบบ: `<eos>` (จบสนทนา), `<turn|>` (จบ turn), `<|tool_response>` (รอ tool result) |
| 5 | **Template คือ "สัญญา" train/inference** | `chat_template.jinja` (canonical, 2026-07-09) — โมเดลถูกเทรนด้วย template นี้; เปลี่ยน template = เปลี่ยนพฤติกรรม |
| 6 | **Tool-calling เป็น first-class citizen** | ไม่ใช่ post-processing — เทรนด้วย syntax `<|tool_call>call:name{args}<tool_call|>`; regex parse ใน schema (`x-regex-iterator`) |
| 7 | **แยก "ประกาศ" กับ "เรียกใช้" กับ "ผลลัพธ์"** | `<|tool>` (declaration) → `<|tool_call>` (invocation) → `<|tool_response>` (result) — 3 บทบาทแยกกันชัดเจน |
| 8 | **Preserve thinking ใน tool loop** | `thinking_gate = (loop.index0 > last_user_idx) or (preserve_thinking and tool_calls)` — คิดไว้ใน tool turn แต่ตัดใน multi-turn ปกติ |
| 9 | **O(1) state tracking แทน O(n) scan** | `prev_message_type` / `prev_non_tool_role` — รู้สถานะ turn ก่อนหน้าโดยไม่ scan ย้อนหลัง (comment ใน template: "O(1) instead of O(n) backward scan") |
| 10 | **Forward-scan สำหรับ continuation** | ตรวจ next non-tool message → ตัดสินใจปิด turn (`<turn|>`) หรือต่อ assistant — template รู้จังหวะการพูด |
| 11 | **Escape/format เป็น token กัน injection** | `<|"|>` ใช้ครอบ string ใน JSON arguments — กัน prompt injection จากการ format; `format_argument()` จัดการทุก type |
| 12 | **Instruction tuning = เปลี่ยนพฤติกรรม ไม่ใช่โครงสร้าง** | tensor/architecture เหมือน base เป๊ะ (2,130 tensors, 7.996B, BF16) — ต่างแค่: eos array, chat_template.jinja, response_schema, tokenizer_config ใหญ่ขึ้น — น้ำหนักต่าง แต่โครงสร้างเดียวกัน |

---

## 3. ตารางเปรียบเทียบ base vs it

| มิติ | base (pre-trained) | it (instruction-tuned) |
|------|--------------------|------------------------|
| **เป้าหมายหลัก** | เรียนรู้ภาษา/โลก (next-token) | ทำตามคำสั่ง/สนทนา/ใช้ tools |
| **โครงสร้างโมเดล** | 2,130 tensors, 7.996B, BF16 | **เหมือนกันเป๊ะ** (2,130 tensors, 7.996B) |
| **ไฟล์** | 8 ไฟล์ | 9 ไฟล์ (+ `chat_template.jinja`) |
| **eos_token_id** | [1, 106] | **[1, 106, 50]** (+ tool_response) |
| **tokenizer_config** | 881 B (special tokens พื้นฐาน) | **3.1 KB** (+ response_schema + response_template) |
| **การควบคุม reasoning** | `<|think|>` token (มีใน vocab) | `<|think|>` + thinking channel + `enable_thinking` flag ใน template |
| **Tool-calling** | ไม่มี (token มี แต่ไม่ถูกเทรน) | **first-class**: declaration/invocation/response loop + JSON schema |
| **หลักการเด่น** | ประหยัด (PLE/KV/GQA), safety (safetensors), modularity | **ควบคุมพฤติกรรม** (token-based), structured output, template contract |
| **จุดร่วม** | ทั้งคู่: hybrid attention, PLE, 128K context, vocab 262K, multimodal | |

---

## 4. บทสรุป

1. **it = base + "ชั้นควบคุมพฤติกรรม"** — โครงสร้างทางกายภาพไม่เปลี่ยน (tensor/architecture เดิม) แต่เพิ่ม 3 สิ่ง: chat template (สัญญา), response schema (contract), eos array (termination) — ทั้ง 3 อยู่ที่ "ชั้น config/tokenizer" ไม่ใช่ชั้นน้ำหนัก
2. **หลักการ base = ประหยัดทรัพยากร** — ทุกอย่างถูกออกแบบให้ใช้ memory/compute น้อยลง (PLE, KV sharing, hybrid attention)
3. **หลักการ it = ควบคุมพฤติกรรม** — ทุกอย่างถูกออกแบบให้โมเดล output ตามที่ต้องการ (token-based control, JSON schema, tool loop)
4. **บทเรียนรวม:** โมเดลที่ดี = โครงสร้างประหยัด (base) + พฤติกรรมควบคุมได้ (it) — สองชั้นนี้แยกจากกันโดยสิ้นเชิง

---

*รายงานโดย คำปัน (Kampun) — 2026-08-20-S40 — BOI X-1 Knowledge Synthesis*