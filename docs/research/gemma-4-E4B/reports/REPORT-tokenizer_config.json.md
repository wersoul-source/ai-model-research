# รายงานการศึกษา: tokenizer_config.json — การตั้งค่า Tokenizer

> Session: 2026-08-20-S39 | Skill: diagnose-gemma-4 | ขนาดไฟล์: 881 bytes

## 1. ภาพรวม

`tokenizer_config.json` คือ **ไฟล์ตั้งค่า** ของ tokenizer — บอกชื่อ special tokens ทั้งหมด + ค่าเริ่มต้น (padding side, class) ทำงานคู่กับ `tokenizer.json` (ที่มี vocab จริง)

## 2. เนื้อหาทั้งไฟล์ (33 บรรทัด)

```json
{
  "audio_token": "<|audio|>",
  "backend": "tokenizers",
  "boa_token": "<|audio>",
  "boi_token": "<|image>",
  "bos_token": "<bos>",
  "eoa_token": "<audio|>",
  "eoc_token": "<channel|>",
  "eoi_token": "<image|>",
  "eos_token": "<eos>",
  "eot_token": "<turn|>",
  "escape_token": "<|\"|>",
  "etc_token": "<tool_call|>",
  "etd_token": "<tool|>",
  "etr_token": "<tool_response|>",
  "extra_special_tokens": ["<|video|>"],
  "image_token": "<|image|>",
  "mask_token": "<mask>",
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": "<pad>",
  "padding_side": "left",
  "processor_class": "Gemma4Processor",
  "soc_token": "<|channel>",
  "sot_token": "<|turn>",
  "stc_token": "<|tool_call>",
  "std_token": "<|tool>",
  "str_token": "<|tool_response>",
  "think_token": "<|think|>",
  "tokenizer_class": "GemmaTokenizer",
  "unk_token": "<unk>"
}
```

## 3. วิเคราะห์แต่ละกลุ่ม

### Special Tokens (ชื่อ → ความหมาย)

| กลุ่ม | Token | ความหมาย |
|-------|-------|----------|
| พื้นฐาน | bos/eos/pad/unk/mask | มาตรฐาน |
| ภาพ | boi (`<|image`) / eoi (`<image|>`) / image (`<|image|>`) | เริ่ม/จบ/กลาง ภาพ |
| เสียง | boa (`<|audio`) / eoa (`<audio|>`) / audio (`<|audio|>`) | เริ่ม/จบ/กลาง เสียง |
| วิดีโอ | extra: `<|video|>` | token วิดีโอ |
| คิด | think (`<|think|>`) | เปิดโหมด reasoning |
| ช่อง | soc (`<|channel`) / eoc (`<channel|>`) | กรอบความคิด |
| รอบ | sot (`<|turn`) / eot (`<turn|>`) | เปลี่ยนผู้พูด |
| เครื่องมือ | std (`<|tool`) / etd (`<tool|>`) / stc (`<|tool_call`) / etc (`<tool_call|>`) / str (`<|tool_response`) / etr (`<tool_response|>`) | function calling |
| อื่น | escape (`<|"|>`) | escape |

### ค่าอื่น ๆ

| ค่า | ค่า | ความหมาย |
|-----|-----|----------|
| backend | tokenizers | ใช้ lib tokenizers (Rust) |
| tokenizer_class | GemmaTokenizer | คลาสใน transformers |
| processor_class | Gemma4Processor | คลาส processor |
| padding_side | left | เติม padding ด้านซ้าย (สำคัญสำหรับ decoder-only) |
| model_max_length | 1e30 (ใหญ่เกินจริง) | ไม่จำกัดความยาว (ใช้ config.json แทน) |

## 4. ความสัมพันธ์กับไฟล์อื่น

```
tokenizer_config.json (ชื่อ token)  ──┐
                                      ├→ GemmaTokenizer → แปลงข้อความ → token IDs
tokenizer.json (vocab + กฎ BPE)  ────┘
```

- `tokenizer_config.json` = "ป้ายชื่อ" — บอกว่าอะไรเรียกว่าอะไร
- `tokenizer.json` = "ตัวจริง" — มี vocab + กฎการตัดคำ
- ต้องตรงกันเสมอ — ถ้าชื่อ token ใน config ไม่มีใน vocab → error

## 5. สรุป

- ไฟล์เล็กแต่สำคัญ — กำหนด "ภาษาควบคุม" ของโมเดล (คิด/ดู/ฟัง/ใช้เครื่องมือ)
- `padding_side: left` บอกว่าโมเดลนี้เป็น decoder-only (เติม padding ด้านซ้าย)
- ตรวจแล้ว: special tokens ครบทุกตัวตรงกับ tokenizer.json (PASS)