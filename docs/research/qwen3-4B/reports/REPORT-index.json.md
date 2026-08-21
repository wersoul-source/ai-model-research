# รายงาน model.safetensors.index.json — Qwen3-4B

> ไฟล์: `/root/models/Qwen3-4B/model.safetensors.index.json` (32 K) | ตรวจ: 63/63 PASS

## เนื้อหาไฟล์

```json
{
  "metadata": {"total_size": 8044936192},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00003.safetensors",
    ...
    "model.norm.weight": "model-00003-of-00003.safetensors"
  }
}
```

## วิเคราะห์

- **total_size:** 8,044,936,192 bytes (7.49 GiB) — ตรงกับผลรวม 3 shards (3.96G + 3.99G + 99M)
- **weight_map:** 398 entries — ครอบคลุมทุก tensor (174 + 219 + 5)
- **Shard assignment:**
  - shard 1 (174 tensors): embed_tokens + layers 0-15 (บางส่วน)
  - shard 2 (219 tensors): layers 15-35 (ส่วนใหญ่)
  - shard 3 (5 tensors): layers 35 บางส่วน + norm
- **Mapping correct:** ทุก tensor ใน weight_map มีอยู่ใน header ของ shard ที่ระบุ (0 mismatched) — ตรวจด้วย header parse

## หลักการ sharded

- **Contract:** index.json คือสัญญา — loader ใช้ map นี้โหลด shard ที่ต้องการ (lazy loading)
- **Contiguity ต่อ shard:** แต่ละ shard ตรวจสอบ contiguity แยกกัน (no holes)
- **Resume:** ดาวน์โหลด shard ละไฟล์ — fail แล้ว resume ทีละ shard

## เทียบ Gemma 4 E4B

- Gemma: ไฟล์เดียว 16GB (2130 tensors) — ไม่มี index
- Qwen: 3 shards + index (398 tensors) — แบ่ง ~4GB/shard

*ตรวจด้วย diagnose-qwen3.py: index checks 4/4 PASS (total_size, count, shards, mapping)*
