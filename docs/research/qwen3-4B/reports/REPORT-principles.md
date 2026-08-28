# หลักการสร้าง — Qwen3 4B (และเปรียบเทียบ Gemma 4 E4B)

> Session: 2026-08-21-S41 | สังเคราะห์จากไฟล์จริง 13 ไฟล์ของ `Qwen/Qwen3-4B` + Qwen3 Technical Report + เทียบ Gemma 4 E4B
> Skill: diagnose-qwen3 (63/63 PASS)

---

## 1. หลักการสร้างของ Qwen3-4B (12 ข้อ)

| # | หลักการ | หลักฐานในโมเดล |
|---|---------|----------------|
| 1 | **Full attention ทุก layer — เรียบง่ายกว่าฉลาดกว่า** | 36 layers ใช้ full attention หมด (use_sliding_window false, sliding_window null) — ไม่ใช้ hybrid sliding/full แบบ Gemma — เน้นคุณภาพ attention เต็มที่ ยอมจ่าย compute มากกว่า |
| 2 | **GQA 4x + head_dim 128 — ประหยัด KV แบบมาตรฐาน** | 32 Q heads → 8 KV heads (q_proj 4096, k/v_proj 1024) — ลด KV cache 4x เหมือน Gemma (8→2) แต่ head_dim เล็กกว่า (128 vs 256/512) |
| 3 | **QK-Norm ทุก layer — stable training** | q_norm + k_norm [128] ทุก layer (36×2) — ใหม่ใน Qwen3 (ไม่มีใน Qwen2) — แก้ปัญหา training instability ที่ scale ใหญ่ |
| 4 | **SwiGLU 9728 — สูตรมาตรฐาน** | gate/up 9728, down 2560×9728 — ใกล้เคียง Gemma 10240 — ไม่ใช้ PLE, ไม่มี double-wide MLP |
| 5 | **RoPE 1M + YaRN + DCA + ABF — ขยาย context ด้วย config** | rope_theta 1M + rope_scaling null (default 32K) → เพิ่ม YaRN factor 4 → 128K — Gemma ใส่ 128K ใน architecture เลย, Qwen ใส่ใน config |
| 6 | **Sharded safetensors + index.json — แจกจ่าย** | 3 shards (174+219+5=398 tensors) + index.json เป็น contract — แบ่งไฟล์ ~4GB/shard เพื่อ resume ง่าย — Gemma ไฟล์เดียว 16GB |
| 7 | **BBPE 151K + merges 151K — ภาษาเยอะ** | vocab 151643 + 26 added + merges 151387 — vocab เล็กกว่า Gemma (262K) แต่ครอบคลุม 119 ภาษา + synthetic data |
| 8 | **Thinking/non-thinking ในโมเดลเดียว — สวิตช์ด้วย flag + token** | `enable_thinking` flag + `<think>`/`</think>` (151667/151668) + `/think`/`/no_think` soft switch + thinking budget — Gemma ใช้ `<|think|>` + channel |
| 9 | **Chat template คือสัญญา train/inference** | template ใน tokenizer_config — จัดการ thinking, tool_call, multi-step tool, im_start/im_end — เปลี่ยน template = เปลี่ยนพฤติกรรม เหมือน Gemma |
| 10 | **tie_word_embeddings — ประหยัด** | embed_tokens [151936,2560] ผูกกับ output — ประหยัด 0.39B params — Gemma ก็ tie |
| 11 | **Strong-to-Weak Distillation — เล็กเรียนรู้จากใหญ่** | Qwen3-4B ถูก distill จาก 32B/235B (off-policy + on-policy) — ไม่ต้องทำ 4-stage RL เอง — ประหยัด GPU 10x — Gemma ไม่มี |
| 12 | **Text-only specialist — ไม่ยัด multimodal** | ไม่มี vision/audio tower — text specialist ล้วน — multimodal แยกเป็น Qwen-VL — Gemma ยัดทุก modality ในตัวเดียว |

---

## 2. ตารางเปรียบเทียบ Qwen3-4B vs Gemma 4 E4B

| มิติ | Qwen3-4B | Gemma 4 E4B | บทเรียน |
|------|----------|-------------|---------|
| **Layers** | 36 full attention | 42 hybrid (35 sliding + 7 full) | Qwen เรียบง่าย, Gemma ประหยัด |
| **Hidden** | 2560 | 2560 | เท่ากัน |
| **Heads** | 32 Q / 8 KV (GQA 4x) | 8 Q / 2 KV (GQA 4x) | Qwen heads เยอะ, KV เยอะกว่า |
| **Head dim** | 128 | 256/512 | Gemma ใหญ่กว่า |
| **Intermediate** | 9728 SwiGLU | 10240 | ใกล้กัน |
| **Vocab** | 151,936 (BBPE) | 262,144 | Gemma ใหญ่กว่า 72% |
| **Context** | 32K → 128K YaRN | 128K native | Qwen ขยายด้วย config, Gemma ฝังใน arch |
| **RoPE** | 1M + YaRN/DCA/ABF | sliding 10K / full 1M p-RoPE | วิธีต่าง ผลลัพธ์เหมือนกัน 128K |
| **Norm** | RMSNorm + QK-Norm | RMSNorm | Qwen เพิ่ม QK-Norm |
| **PLE** | ไม่มี | มี 2.9B (effective 4.4B) | Gemma ใช้ lookup แทน compute |
| **Sharding** | 3 shards + index | 1 file | Qwen แจกจ่าย, Gemma ก้อนเดียว |
| **Thinking** | flag + `<think>` pair + budget | `<|think|>` + channel | วิธีต่าง ไอเดียเหมือนกัน |
| **Multimodal** | Text-only | Text+Image+Audio+Video | Gemma ออลอินวัน, Qwen แยก |
| **Vocab langs** | 119 ภาษา (36T tokens) | 140+ ภาษา | ใกล้กัน |
| **Distill** | Strong-to-Weak (10x ประหยัด) | ไม่มี | Qwen ประหยัด train เล็ก |
| **Params** | 4.02B (3.63B non-embed) | 7.99B (4.42B effective) | Qwen เล็กกว่า |
| **Memory BF16** | 8.04 GB | 16.0 GB | Qwen ครึ่งเดียว |

---

## 3. บทสรุป — 2 ปรัชญา

1. **Qwen = เรียบง่าย + ขยายด้วย config + distill** — Full attention ธรรมดา, ขยาย context ด้วย YaRN ใน config, เล็กเรียนรู้จากใหญ่ — เน้นทำซ้ำง่าย (reproducible)
2. **Gemma = ประหยัดสุด + ยัดทุกอย่างใน arch** — Hybrid attention, PLE, multimodal towers, native 128K — เน้น on-device ประหยัด
3. **จุดร่วม:** GQA 4x, SwiGLU, RoPE 1M, tie_embeddings, thinking mode, chat template เป็นสัญญา — ทั้งคู่สรุปว่า "วิธีต่าง ผลลัพธ์ใกล้กัน"
4. **บทเรียน BOI X-1:** ถ้าจะสร้างโมเดลเราเอง — เลือกได้: (A) แบบ Qwen เรียบง่าย + YaRN + distill (ทำง่าย) หรือ (B) แบบ Gemma ประหยัด + PLE + hybrid (on-device ดี) — หรือผสม: full attention + QK-Norm + YaRN + thinking flag (Qwen) + PLE แบบ Gemma

---

*รายงานโดย คำปัน (Kampun) — 2026-08-21-S41 — BOI X-1 Knowledge Synthesis*
