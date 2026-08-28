# รายงานการศึกษา: tokenizer.json — คำศัพท์ BPE 262K

> Session: 2026-08-20-S39 | Skill: diagnose-gemma-4 | ขนาดไฟล์: 31 MB

## 1. ภาพรวม

`tokenizer.json` คือ **พจนานุกรมของโมเดล** — แปลงข้อความ ↔ token IDs ประกอบด้วย vocab 262,144 คำ + กฎการตัดคำ (BPE) + special tokens 24 ตัว

## 2. โครงสร้าง (9 ส่วน)

```
version: 1.0
truncation: None
padding: None
added_tokens: 24 ตัว (special)
normalizer: Replace " " → "▁"
pre_tokenizer: Split by space, MergedWithPrevious
post_processor: (template สำหรับ chat)
decoder: Replace ▁→" " + ByteFallback + Fuse
model: BPE (vocab 262,144)
```

## 3. Vocab Analysis

| รายการ | ค่า |
|--------|-----|
| ขนาด | 262,144 (0 - 262143) |
| ประเภท | BPE (Byte Pair Encoding) |
| ID 0-4 | `<pad>` `<eos>` `<bos>` `<unk>` `<mask>` |
| ID 5 | `[multimodal]` |
| ID 6-9 | `<unused0-3>` (สำรอง) |
| ID 262140 | `<unused6223>` (สำรองท้าย) |

**ตัวอย่างคำจริง:**
```
id=1000   'put'
id=5000   'player'
id=10000  'த்தில்' (ทมิฬ)
id=50000  'んにちは' (ญี่ปุ่น)
id=100000 '▁Reactive'
id=200000 '▁Piv'
```

> สังเกต: vocab มีหลายภาษา (ไทย/ญี่ปุ่น/ทมิฬ ฯลฯ) — สอดคล้องกับ "140+ languages" ใน README

## 4. กลไกการทำงาน

### Normalizer
```
Replace: " " (space) → "▁" (U+2581)
```
แปลง space เป็นสัญลักษณ์ ▁ เพื่อให้ BPE เห็นขอบเขตคำ

### Pre-tokenizer
```
Split by " " (space), MergedWithPrevious
```
ตัดข้อความเป็น chunk ตาม space

### Decoder (ตอนแปลงกลับ)
```
1. Replace ▁ → " " (คืน space)
2. ByteFallback (จัดการ byte ที่ไม่ใช่ UTF-8)
3. Fuse (รวมผลลัพธ์)
```

## 5. Special Tokens (24 ตัว)

| กลุ่ม | Tokens | หน้าที่ |
|-------|--------|---------|
| พื้นฐาน | `<pad>` `<eos>` `<bos>` `<unk>` `<mask>` | มาตรฐาน |
| Tool | `<|tool>` `<tool|>` `<|tool_call>` `<tool_call|>` `<|tool_response>` `<tool_response|>` | function calling |
| Thinking | `<|think|>` | เปิดโหมดคิด |
| Channel | `<|channel>` `<channel|>` | กรอบความคิด |
| Turn | `<|turn>` `<turn|>` | เปลี่ยนผู้พูด |
| Escape | `<|"|>` | escape |
| Modality | `<|image>` `<|audio>` `<|image|>` `<|audio|>` `<image|>` `<audio|>` `<|video|>` | ภาพ/เสียง/วิดีโอ |

## 6. สรุป

- tokenizer.json คือไฟล์ที่ใหญ่เป็นอันดับ 2 (31MB) รองจาก model.safetensors
- เป็น BPE มาตรฐาน + special tokens สำหรับ multimodal/thinking/tool
- ต้องใช้คู่กับ `tokenizer_config.json` (บอกชื่อ token) — tokenizer.json มี vocab จริง
- vocab 262,144 ตรงกับ config.json (`vocab_size`) — ตรวจแล้ว PASS