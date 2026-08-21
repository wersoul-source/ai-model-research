# รายงานสรุปรวม — การศึกษา Llama 3.1 8B Instruct (Original + GGUF)

> Session: 2026-08-21-S43 (GGUF) + S43A (original) | Skill: diagnose-llama3 v1.1.0 (21/21 + 39/39 PASS)
> ตรวจสอบจริงที่ `/root/models/Llama-3.1-8B/` (original 16.06 GB — GGUF ลบแล้วหลัง git ครบ)

## 1. ตารางสรุปทั้งหมด

| # | ไฟล์/ไฟล์เซ็ต | ขนาด | หน้าที่ | สถานะ |
|---|---------------|------|---------|-------|
| 1-4 | model-0000{1..4}-of-00004.safetensors | 16.06G | 291 tensors BF16 (82+104+100+5) | ✅ 39/39 PASS |
| 5 | model.safetensors.index.json | 24K | weight_map contract, total_size 16,060,522,496 | ✅ |
| 6 | config.json | 855 B | 32 layers, GQA 32/8, RoPE θ=500K, llama3 scaling | ✅ 15/15 |
| 7 | generation_config.json | 184 B | eos [128001,128008,128009] | ✅ |
| 8 | tokenizer.json | 8.7M | BPE 128,000 base + 256 added = 128,256 | ✅ |
| 9 | tokenizer_config.json | 50K | PreTrainedTokenizerFast | ✅ |
| 10 | special_tokens_map.json | 296 B | bos/eos | ✅ |
| — | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 4.92G | 292 tensors (Q4_K 193+Q6_K 33+F32 66) | ✅ 21/21 → ลบแล้ว (git ครบ) |

## 2. ตัวเลขสำคัญ (ตรวจจริงทุกค่า)

| ตัวเลข | ค่า | แหล่ง |
|--------|-----|-------|
| params | 8,030,261,248 (8.03B) | shard headers รวม |
| tensors safetensors / GGUF | 291 / 292 (+rope_freqs) | header parse |
| dtype | BF16 100% ↔ Q4_K/Q6_K/F32 | headers |
| layers × tensors/layer | 32 × 9 (ไม่มี QK-Norm) | tensor map |
| GQA | 32 Q / 8 KV, head_dim 128 | config + k_proj [1024,4096] |
| SwiGLU | 14336 | gate/up_proj shape |
| vocab | 128,256 = 128,000 + 256 added | tokenizer math ✓ |
| merges | 280,147 | tokenizer.json |
| context | 131072 native (llama3 scaling factor 8, 8K→128K) | config |
| rope θ | 500,000 | config + GGUF KV |
| tie embeddings | false (lm_head แยก = 13% params) | config + tensors |
| rms_eps | 1e-05 | config + GGUF FLOAT32 |
| compression GGUF | 69.4% saved | 16.06→4.92 GB |

## 3. เทียบข้ามตระกูล (Llama vs Qwen3)

| มิติ | Llama 3.1 8B | Qwen3-4B |
|------|--------------|----------|
| ปรัชญา | wide & shallow (32×4096) | deep & narrow (36×2560) |
| QK-Norm | ✗ (9 tensors/layer) | ✓ (11 tensors/layer) |
| tie embeddings | ✗ (+525M params) | ✓ |
| thinking mode | ✗ | ✓ `<think>` |
| context | 128K native | 32K→128K (YaRN) |
| license | llama3.1 community | Apache 2.0 |

## 4. Diagrams

| ไฟล์ | เนื้อหา |
|------|---------|
| file-relationships.mmd/svg | 10 ไฟล์ original + data flow |
| model-architecture.mmd/svg | block × 32 + GQA + SwiGLU + lm_head แยก |
| llama-vs-qwen.mmd/svg | cross-family comparison |

## 5. รายงานย่อย

| รายงาน | ไฟล์ |
|--------|------|
| Original safetensors (12 หลักการ) | `REPORT-llama3.1-8B-original.md` |
| GGUF Q4_K_M (12 หลักการ) | `REPORT-llama3.1-8B-GGUF.md` |

## 6. Traps ที่บันทึกไว้ (ใช้ซ้ำได้)

1. meta-llama gated 401 → mirror (NousResearch/bartowski)
2. NousResearch มี original/*.pth 16.1G junk → --include ให้ชัด
3. HF cache cp = double space → mv blob (readlink -f → rename)
4. GGUF float KV (rms_eps) ห้าม decode เป็น int
5. scalar >65535 ต้อง struct.unpack เต็ม ไม่ใช่ bytes()[0]

---

*รายงานสรุปโดย คำปัน (Kampun) — 2026-08-21 — BOI X-1 Knowledge Synthesis*
