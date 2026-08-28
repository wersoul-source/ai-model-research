# รายงาน tokenizer — Qwen3-4B

> ไฟล์: `tokenizer.json` (11.4 MB) + `vocab.json` (2.8 MB) + `merges.txt` (1.7 MB) | ตรวจ: 63/63 PASS

## โครงสร้าง tokenizer.json

- **type:** BPE, version 1.0
- **vocab:** 151,643 entries (เช่น `!`→0, `"`→1, `<|endoftext|>`→151643)
- **merges:** 151,387 rules (เช่น `Ġ + Ġ → ĠĠ`)
- **added_tokens:** 26 (special)

## added_tokens 26 ตัว

| ID | Content | special |
|----|---------|---------|
| 151643 | `<|endoftext|>` | true |
| 151644 | `<|im_start|>` | true |
| 151645 | `<|im_end|>` | true |
| 151646-151656 | `<|object_ref_...|>`, `<|box_...|>`, `<|vision_...|>`, `<|image_pad|>` | true |
| 151657 | `<tool_call>` | false |
| 151658 | `</tool_call>` | false |
| 151659-151664 | `<|fim_...|>`, `<|repo_name|>`, `<|file_sep|>` | false |
| 151665 | `<tool_response>` | false |
| 151666 | `</tool_response>` | false |
| 151667 | `<think>` | false |
| 151668 | `</think>` | false |

## vocab.json + merges.txt

- **vocab.json:** dict 151,643 entries — map token → id
- **merges.txt:** 151,388 lines (รวม header) — BPE merges

## เทียบ Gemma 4 E4B

| Qwen3-4B | Gemma 4 E4B |
|----------|-------------|
| BBPE, 151K vocab, 151K merges, 26 added | BPE, 262K vocab, 514K merges, 24 added |
| vocab เล็กกว่า 42% |  |
| มี <think> pair + FIM + vision | มี <|think|> + image/audio |

*ตรวจด้วย diagnose-qwen3.py: tokenizer checks 3/3 PASS*
