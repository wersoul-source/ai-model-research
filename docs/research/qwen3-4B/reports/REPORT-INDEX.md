# รายงานสรุปรวม — การศึกษาทั้ง 13 ไฟล์ของ Qwen3-4B + GGUF Q4_K_M

> Session: 2026-08-21-S41 (safetensors) + S41-GGUF (Q4_K_M 13/13 PASS) | Skill: diagnose-qwen3 v1.1.0 (63/63 + 13/13 PASS) | ตรวจสอบจริงที่ `/root/models/Qwen3-4B/` + `/root/models/Qwen3-4B-GGUF/`

## 1. ตารางสรุปทั้ง 13 ไฟล์

| # | ไฟล์ | ขนาด | หน้าที่ | กลุ่ม |
|---|------|------|---------|------|
| 1 | model-00001-of-00003.safetensors | 3.96 GB | shard 1 (174 tensors) | ตัวโมเดล (sharded) |
| 2 | model-00002-of-00003.safetensors | 3.99 GB | shard 2 (219 tensors) | ตัวโมเดล (sharded) |
| 3 | model-00003-of-00003.safetensors | 99 MB | shard 3 (5 tensors) | ตัวโมเดล (sharded) |
| 4 | model.safetensors.index.json | 32 K | weight_map 398 tensors (contract) | ตัวโมเดล (index) |
| 5 | config.json | 726 B | พิมพ์เขียว 36 layers, GQA 32/8, RoPE 1M | ตัวโมเดล (config) |
| 6 | generation_config.json | 239 B | ค่า default generate (eos [151645,151643], temp 0.6) | ตัวโมเดล (config) |
| 7 | tokenizer.json | 11.4 MB | BBPE vocab 151643 + merges 151387 | ตัวแปลง input |
| 8 | tokenizer_config.json | 9.7 K | chat_template + special tokens (thinking/tool) | ตัวแปลง input |
| 9 | vocab.json | 2.8 MB | vocab mapping | ตัวแปลง input |
| 10 | merges.txt | 1.7 MB | BPE merges | ตัวแปลง input |
| 11 | README.md | 16.9 K | model card + thinking mode + YaRN | เอกสาร |
| 12 | LICENSE | 11 K | Apache 2.0 | เอกสาร |
| 13 | .gitattributes | 1.6 K | Git LFS rules | ระบบ |
| 14 | Qwen3-4B-Q4_K_M.gguf | 2.50 GB | GGUF Q4_K_M (398 tensors, 216 Q4_K + 37 Q6_K + 145 F32) | ตัวโมเดล (GGUF) |

> **Sharded vs single-file:** Qwen3-4B แบ่ง 3 shards (~4GB/shard) + index.json — Gemma 4 E4B ไฟล์เดียว 16GB — sharded ช่วยให้ resume ง่าย
> **safetensors vs GGUF:** BF16 8.04 GB (3 shards) vs Q4_K_M 2.50 GB (1 file, 71% saved) — 398 tensors เท่ากัน แค่ packing ต่าง (ดู REPORT-GGUF.md + gguf-comparison.svg)

## 2. ความสัมพันธ์ระหว่างไฟล์ (Data Flow)

```
┌─ กลุ่มโหลดโมเดล ───────────────────────────────────────┐
│  config.json ──→ สร้างโครงสร้าง 36 layers (GQA/QK-Norm) │
│  model.safetensors.index.json ──→ map 398 tensors → 3 shards │
│  model-0000X.safetensors ──→ น้ำหนัก BF16 8.04 GB      │
│  generation_config.json ──→ ค่า default generate       │
│       ↓                                                │
│  AutoModelForCausalLM.from_pretrained()                │
└────────────────────────────────────────────────────────┘

┌─ กลุ่มประมวลผล input ──────────────────────────────────┐
│  tokenizer.json + vocab.json + merges.txt ──→ BBPE    │
│  tokenizer_config.json ──→ chat_template (thinking)   │
│       ↓                                                │
│  AutoTokenizer.from_pretrained()                       │
│  tokenizer.apply_chat_template(enable_thinking=...)    │
└────────────────────────────────────────────────────────┘
```

## 3. ตัวเลขสำคัญที่ตรวจสอบแล้ว (ทุกค่า PASS 63/63)

| ตัวเลข | ค่า | ตรวจจาก |
|--------|-----|---------|
| พารามิเตอร์รวม | 4,022,468,096 (4.02B) | shards header (398 tensors) |
| พารามิเตอร์ non-embedding | 3,633,511,936 (3.63B) | total - embed |
| Layers | 36 (full attention ทุก layer) | config + tensors (11/layer) |
| Hidden size | 2560 | config + q_proj shape |
| GQA | 32 Q / 8 KV (head_dim 128) | q_proj [4096,2560], k_proj [1024,2560] |
| QK-Norm | 36 layers × (q_norm + k_norm) | tensors |
| Vocab | 151,936 (config) / 151,643 + 26 added | tokenizer |
| Merges | 151,387 | tokenizer.json |
| Context | 32K native → 128K YaRN (factor 4) | config + README |
| RoPE theta | 1,000,000 | config |
| dtype | BF16 100% (398 tensors) | shards header |
| Shards | 3 (174+219+5) | index.json |
| EOS | [151645, 151643] | generation_config |
| Thinking tokens | 151667/151668 | tokenizer_config |
| Header sizes | 20000, 25232, 552 bytes | shards header |
| Memory BF16 | 8.04 GB | total ×2 |

## 4. สถาปัตยกรรมโดยรวม (จากทุกไฟล์รวมกัน)

```
Input: ข้อความ (+ /think /no_think + tools)
  │
  ├─ BBPE Tokenizer (151K vocab, 151K merges) → input_ids
  ├─ Chat Template (enable_thinking + tool_call) → prompt
  │
  ▼
Language Model (36 layers, full attention)
├─ embed_tokens [151936, 2560] (tie)
├─ 36 × Transformer Block:
│   ├─ GQA Attention: 32 Q / 8 KV, head_dim 128, QK-Norm
│   ├─ RMSNorm (input + post_attention)
│   ├─ SwiGLU MLP: gate/up 9728, down 2560×9728
│   └─ RoPE 1M + YaRN (32K→128K)
├─ norm [2560]
  │
  ▼
Output logits 151936 → softmax → ข้อความ (+ <think> reasoning ถ้า enable_thinking)
```

## 5. บทเรียนจากการศึกษาทั้ง 13 ไฟล์

1. **Sharded = แจกจ่าย** — index.json คือสัญญา — ขาด shard เดียวใช้ไม่ได้ แต่ resume ง่ายกว่าไฟล์เดียว
2. **config คือสัญญา** — ทุกไฟล์ต้องตรง config (ตรวจได้ด้วย diagnose-qwen3.py 63 checks)
3. **Thinking เป็น flag + token + template** — ควบคุมพฤติกรรมด้วย 3 ชั้น (enable_thinking, <think> pair, /think switch)
4. **YaRN ใน config ไม่ใช่ arch** — ขยาย context ด้วย rope_scaling ไม่ต้องแก้โครงสร้าง
5. **QK-Norm คือของใหม่ Qwen3** — ไม่มีใน Qwen2 — ช่วยให้ train ใหญ่ stable

## 6. ไฟล์รายงานย่อย

| รายงาน | ไฟล์ |
|--------|------|
| ทั้ง 3 shards + index | (รายงานในแชท — 63/63 PASS) |
| config.json | `REPORT-config.json.md` |
| generation_config.json | `REPORT-generation_config.json.md` |
| tokenizer (json/vocab/merges) | `REPORT-tokenizer.json.md` |
| tokenizer_config.json | `REPORT-tokenizer_config.json.md` |
| model.safetensors.index.json | `REPORT-index.json.md` |
| README.md | `REPORT-README.md.md` |
| .gitattributes | `REPORT-.gitattributes.md` |
| หลักการสร้าง 12 ข้อ | `REPORT-principles.md` |
| GGUF Q4_K_M | `REPORT-GGUF.md` (safetensors vs GGUF, 71% saved, 12 ข้อ) |

---

*รายงานสรุปโดย คำปัน (Kampun) — 2026-08-21 — BOI X-1 Knowledge Synthesis*
