# รายงานการศึกษา: config.json — พิมพ์เขียวสถาปัตยกรรม Gemma 4 E4B

> Session: 2026-08-20-S39 | Skill: diagnose-gemma-4 | ขนาดไฟล์: 5.0 KB

## 1. ภาพรวม

`config.json` คือ **พิมพ์เขียว (blueprint)** ของโมเดล — บอก Transformers ว่าต้องสร้างโครงสร้างแบบไหนก่อนใส่น้ำหนักจาก `model.safetensors` มี 18 top-level keys ประกอบด้วย 3 sub-config (text/vision/audio) + special token IDs + ค่าทั่วไป

## 2. โครงสร้างหลัก

| ส่วน | เนื้อหา |
|------|---------|
| `architectures` | `["Gemma4ForConditionalGeneration"]` — คลาสที่ transformers ใช้โหลด |
| `text_config` | ค่าของ Language Model (42 layers) |
| `vision_config` | ค่าของ Vision Tower (16 layers) |
| `audio_config` | ค่าของ Audio Tower (12 layers) |
| token IDs | boi/boa/image/audio/eoi/eoa/video |
| `dtype` | `bfloat16` |
| `tie_word_embeddings` | `true` — ใช้ embedding เดียวกับ output layer |

## 3. text_config (Language Model) — หัวใจหลัก

| ค่า | ค่า | ความหมาย |
|-----|-----|----------|
| hidden_size | 2560 | มิติของ token embedding |
| intermediate_size | 10240 | ขนาด MLP (4× hidden) |
| num_hidden_layers | 42 | จำนวนชั้น |
| num_attention_heads | 8 | query heads |
| num_key_value_heads | 2 | KV heads (GQA — 8:2) |
| head_dim | 256 | ขนาด head (local) |
| global_head_dim | 512 | ขนาด head (global layers) |
| num_kv_shared_layers | 18 | ชั้นที่แชร์ KV cache (42.9%) |
| sliding_window | 512 | หน้าต่าง local attention |
| max_position_embeddings | 131072 | context 128K |
| vocab_size | 262144 | ขนาดคำศัพท์ |
| final_logit_softcapping | 30.0 | จำกัดค่า logit สุดท้าย |
| hidden_activation | gelu_pytorch_tanh | activation ของ MLP |
| hidden_size_per_layer_input | 256 | มิติ PLE ต่อ layer |

### layer_types (42 ชั้น) — รูปแบบ hybrid attention
```
sliding=35 ชั้น, full=7 ชั้น
pattern: [sliding×5, full] ซ้ำ 7 รอบ → ชั้นสุดท้าย (41) = full เสมอ
full อยู่ที่ชั้น: 5, 11, 17, 23, 29, 35, 41
```

### rope_parameters — การเข้ารหัสตำแหน่ง 2 แบบ
| ประเภท | rope_type | theta | partial_rotary_factor |
|--------|-----------|-------|----------------------|
| full_attention | proportional (p-RoPE) | 1,000,000 | 0.25 |
| sliding_attention | default (RoPE) | 10,000 | — |

## 4. vision_config (Vision Tower)

| ค่า | ค่า | ความหมาย |
|-----|-----|----------|
| hidden_size | 768 | มิติ |
| num_hidden_layers | 16 | จำนวนชั้น |
| num_attention_heads | 12 | heads |
| patch_size | 16 | ขนาด patch (ภาพแบ่งเป็น 16×16) |
| pooling_kernel_size | 3 | pooling หลัง encoder |
| position_embedding_size | 10240 | ขนาดตารางตำแหน่ง |
| default_output_length | 280 | soft tokens ต่อภาพ |
| rope_theta | 100.0 | ความถี่ RoPE |

## 5. audio_config (Audio Tower — Conformer)

| ค่า | ค่า | ความหมาย |
|-----|-----|----------|
| hidden_size | 1024 | มิติ |
| num_hidden_layers | 12 | จำนวนชั้น |
| num_attention_heads | 8 | heads |
| conv_kernel_size | 5 | kernel ของ conv |
| subsampling_conv_channels | [128, 32] | ลดความถี่ 2 ชั้น |
| output_proj_dims | 1536 | มิติ output (ก่อนเข้า LM) |
| attention_context_left | 13 | context ด้านซ้าย (causal) |
| attention_chunk_size | 12 | chunked attention |
| attention_logit_cap | 50.0 | จำกัด logit |

## 6. Special Token IDs

| Token | ID | ใช้เมื่อ |
|-------|-----|---------|
| boi (`<|image`) | 255999 | เริ่มภาพ |
| boa (`<|audio`) | 256000 | เริ่มเสียง |
| image (`<|image|>`) | 258880 | token ภาพ |
| audio (`<|audio|>`) | 258881 | token เสียง |
| eoi (`<image|>`) | 258882 | จบภาพ |
| eoa (`<audio|>`) | 258883 | จบเสียง |
| video (`<|video|>`) | 258884 | token วิดีโอ |

## 7. สรุป

- config.json ตรงกับ tensor จริงใน model.safetensors 100% (ตรวจด้วย diagnose-gemma4.py)
- เป็นไฟล์ที่ transformers อ่านก่อนเสมอ — ถ้าไฟล์นี้เสีย โหลดโมเดลไม่ได้ทั้งที่น้ำหนัก完好
- ค่าที่น่าสนใจที่สุด: `num_kv_shared_layers=18` + `global_head_dim=512` + `hidden_size_per_layer_input=256` — คือ 3 ค่าที่ทำให้โมเดลนี้ "ประหยัด" (KV sharing + PLE)