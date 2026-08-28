# รายงาน generation_config.json — Qwen3-4B

> ไฟล์: `/root/models/Qwen3-4B/generation_config.json` (239 B) | ตรวจ: 63/63 PASS

## เนื้อหาไฟล์

```json
{
  "bos_token_id": 151643,
  "eos_token_id": [151645, 151643],
  "pad_token_id": 151643,
  "do_sample": true,
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95
}
```

## วิเคราะห์

- **eos = [151645, 151643]** — `<|im_end|>` และ `<|endoftext|>` — multi-eos เหมือน Gemma [1,106,50] แต่คนละ token
- **temperature 0.6 (thinking), top_k 20, top_p 0.95** — ค่าต่ำกว่า Gemma (1.0/64/0.95) — thinking mode ต้อง sampling แบบระมัดระวัง — DO NOT greedy
- **สำหรับ non-thinking:** temp 0.7, top_p 0.8, top_k 20 (จาก README)
- **Best practice:** presence_penalty 0-2 ช่วยลด repetition

## เทียบ Gemma 4 E4B-it

| Qwen3-4B | Gemma 4 E4B-it |
|----------|----------------|
| eos [151645,151643] | eos [1,106,50] |
| temp 0.6 | temp 1.0 |
| top_k 20 | top_k 64 |
| top_p 0.95 | top_p 0.95 |

*ตรวจด้วย diagnose-qwen3.py: generation checks 4/4 PASS*
