# รายงานการศึกษา: chat_template.jinja — Chat Template (ไฟล์ใหม่ของรุ่น -it)

> Session: 2026-08-20-S39B | Skill: diagnose-gemma-4 | ขนาดไฟล์: 18.6 KB

## 1. ภาพรวม

`chat_template.jinja` คือ **ไฟล์ใหม่ที่มาพร้อมรุ่น instruction-tuned (`-it`)** — ไม่มีในรุ่น base (E4B) เป็น template ภาษา Jinja ที่บอกวิธีจัดรูปแบบข้อความแชท (system/user/assistant/tool) ก่อนส่งเข้าโมเดล

## 2. ข้อมูลจากไฟล์

```
Template: Google Gemma 4 Canonical Chat Template
Author: Google Gemma Engineering Team
Published: 2026-07-09
Context: Fixed tool-calling loops, turn closures, and thinking content-ordering
```

## 3. หน้าที่หลัก

| ฟีเจอร์ | รายละเอียด |
|---------|-----------|
| **Tool-calling loops** | จัดรูปแบบรอบการเรียกเครื่องมือ (tool_call → tool_response) ให้ถูกต้อง |
| **Turn closures** | ปิดรอบการสนทนา (`<|turn>` / `<turn|>`) อย่างถูกต้อง |
| **Thinking content-ordering** | จัดลำดับเนื้อหาความคิด (`<|think|>`) ให้ถูกที่ |
| **format_parameters** | จัดรูปแบบพารามิเตอร์ของ function calling (properties/required/enum) |
| **Multi-turn** | รองรับการสนทนาหลายรอบ |

## 4. ความสัมพันธ์กับไฟล์อื่น

```
tokenizer_config.json (บอกชื่อ token) ──┐
                                        ├→ Gemma4Processor.apply_chat_template()
chat_template.jinja (รูปแบบการจัด)  ────┘
        ↓
ข้อความที่จัดรูปแบบแล้ว → tokenizer → token IDs → โมเดล
```

## 5. ทำไมรุ่น -it ต้องมี

- รุ่น base (E4B) ฝึกมาเพื่อ "เติมคำ" — ไม่ต้องมี template พิเศษ
- รุ่น -it ฝึกมาเพื่อ "สนทนา" — ต้องรู้ว่าใครพูดอะไร (system/user/assistant) + ใช้เครื่องมือได้
- template นี้คือ "กาว" ที่เชื่อมรูปแบบการสนทนากับการฝึกของโมเดล

## 6. สรุป

- chat_template.jinja = เอกลักษณ์ของรุ่น instruction-tuned
- ใช้ผ่าน `processor.apply_chat_template(messages, tokenize=True, ...)`
- ต้องใช้คู่กับ `enable_thinking` parameter (เปิด/ปิดโหมดคิด)