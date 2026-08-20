# รายงานการศึกษา: generation_config.json — ค่า Default การ Generate

> Session: 2026-08-20-S39B (อัปเดตเป็นรุ่น -it) | Skill: diagnose-gemma-4 | ขนาดไฟล์: 208 bytes

## 1. ภาพรวม

ไฟล์เล็กที่สุดในชุด (208 bytes) แต่สำคัญ — กำหนด **ค่า default ที่ใช้ตอน `model.generate()`** ถ้าโค้ดไม่ระบุค่าเอง จะใช้ค่าจากไฟล์นี้

## 2. เนื้อหาทั้งไฟล์ (รุ่น -it)

```json
{
  "bos_token_id": 2,
  "do_sample": true,
  "eos_token_id": [1, 106, 50],
  "pad_token_id": 0,
  "temperature": 1.0,
  "top_k": 64,
  "top_p": 0.95,
  "transformers_version": "5.5.0.dev0"
}
```

## 3. วิเคราะห์แต่ละค่า

| ค่า | ค่า | ความหมาย | ผลกระทบ |
|-----|-----|----------|---------|
| `bos_token_id` | 2 | `<bos>` | เริ่มประโยค |
| `eos_token_id` | **[1, 106, 50]** | `<eos>` + `<turn|>` + `<tool_response|>` | จบได้ 3 แบบ (ดูด้านล่าง) |
| `pad_token_id` | 0 | `<pad>` | เติมความยาว |
| `do_sample` | true | ใช้ sampling (ไม่ใช่ greedy) | สร้างข้อความหลากหลาย |
| `temperature` | 1.0 | ความสุ่ม | 1.0 = สมดุล (ตาม best practice ของ README) |
| `top_k` | 64 | จำกัด 64 อันดับแรก | ลด token แปลก ๆ |
| `top_p` | 0.95 | nucleus sampling | ตัด tail ที่ความน่าจะเป็นรวมเกิน 95% |

## 4. ⚠️ ความแตกต่างจากรุ่น base (E4B)

| รายการ | base (E4B) | it (E4B-it) |
|--------|-----------|-------------|
| `eos_token_id` | `1` (ตัวเดียว) | `[1, 106, 50]` (array) |
| ขนาดไฟล์ | 181 B | 208 B |

**ทำไม eos ต้องเป็น array:**
- `1` = `<eos>` — จบการสนทนาปกติ
- `106` = `<turn|>` — จบรอบ (turn) ใน multi-turn conversation
- `50` = `<tool_response|>` — จบการตอบกลับเครื่องมือ (tool-calling)

→ โมเดล -it ต้องรู้ว่า "จบตอนไหน" ได้หลายแบบ — ต่างจาก base ที่จบแค่ `<eos>`

## 4. สอดคล้องกับ Best Practices (README)

README ระบุ sampling ที่แนะนำ: `temperature=1.0, top_p=0.95, top_k=64` — **ตรงกับไฟล์นี้เป๊ะ** → ผู้ใช้ไม่ต้องตั้งค่าเองก็ได้ผลลัพธ์ตามที่ Google แนะนำ

## 5. สรุป

- ไฟล์นี้คือ "ค่าเริ่มต้นที่ปลอดภัย" — ตั้งมาให้ผลลัพธ์ดีโดยไม่ต้องปรับ
- ถ้าต้องการ deterministic output → ตั้ง `do_sample=false`
- ถ้าต้องการความคิดสร้างสรรค์สูง → เพิ่ม temperature (>1.0)
- หมายเหตุ: `transformers_version` บอกว่าโมเดลนี้สร้างด้วย transformers 5.5.0.dev0 — ต้องใช้เวอร์ชัน ≥ นี้ถึงจะโหลดได้