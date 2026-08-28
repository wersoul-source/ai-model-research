# รายงาน config.json — Qwen3-4B

> ไฟล์: `/root/models/Qwen3-4B/config.json` (726 B) | ตรวจ: 63/63 PASS

## เนื้อหาไฟล์ (JSON)

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 2560,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 9728,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "rope_theta": 1000000,
  "rope_scaling": null,
  "tie_word_embeddings": true,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-06,
  "use_sliding_window": false
}
```

## วิเคราะห์

- **36 layers full attention** — ไม่ใช้ sliding window (Gemma ใช้ hybrid)
- **GQA 32/8, head_dim 128** — q_proj 4096 (=32×128), k/v 1024 (=8×128) — กึ่งหนึ่งของ Gemma
- **SwiGLU 9728** — ใกล้ Gemma 10240
- **RoPE 1M + null scaling** — default 32K, เพิ่ม YaRN factor 4 → 128K ใน README
- **QK-Norm** — ไม่ได้ระบุใน config แต่มีใน tensor (q_norm/k_norm) — เป็น implicit ใน arch
- **tie_word_embeddings true** — ประหยัด 0.39B

## เทียบ Gemma 4 E4B

| Qwen3-4B | Gemma 4 E4B |
|----------|-------------|
| 36 layers, full | 42 layers, hybrid |
| GQA 32/8 | GQA 8/2 |
| RoPE 1M | RoPE 10K/1M p-RoPE |
| 32K native | 128K native |

*ตรวจด้วย diagnose-qwen3.py: config checks 11/11 PASS*
