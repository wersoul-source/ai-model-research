# รายงานการศึกษา: README.md — Model Card

> Session: 2026-08-20-S39 | Skill: diagnose-gemma-4 | ขนาดไฟล์: 27 KB (537 บรรทัด)

## 1. ภาพรวม

`README.md` คือ **model card** อย่างเป็นทางการของ Google DeepMind — เอกสารที่บอกทุกอย่างเกี่ยวกับโมเดล: สเปก, ความสามารถ, benchmark, วิธีใช้, ข้อจำกัด

## 2. โครงสร้างเนื้อหา (20 H1, 9 H2, 17 H3)

| ส่วน | เนื้อหา |
|------|---------|
| Frontmatter | library_name, license (apache-2.0), pipeline_tag (any-to-any) |
| แนะนำ Gemma 4 | 5 ขนาด (E2B/E4B/12B/26B-A4B/31B), ความสามารถใหม่ |
| Models Overview | Dense + MoE, hybrid attention, PLE |
| Benchmark Results | MMLU, AIME, LiveCodeBench, GPQA, vision, audio |
| Core Capabilities | Thinking, Long Context, Image/Video/Audio, Function Calling |
| Getting Started | โค้ดโหลด + ใช้โมเดล (text/image/audio/video) |
| Best Practices | sampling, thinking mode, modality order, token budget |
| Model Data | ชุดฝึก, การกรองข้อมูล |
| Ethics & Safety | การประเมินความปลอดภัย |
| Usage & Limitations | ข้อจำกัด, ความเสี่ยง |
| Citation | BibTeX |

## 3. ข้อมูลสำคัญที่ได้จาก README

### สเปก E4B (ตรงกับไฟล์จริง)
| Property | ค่า |
|----------|-----|
| Total Parameters | 4.5B effective (8B with embeddings) |
| Layers | 42 |
| Sliding Window | 512 tokens |
| Context Length | 128K tokens |
| Vocabulary | 262K |
| Modalities | Text, Image, Audio |
| Vision Encoder | ~150M |
| Audio Encoder | ~300M |

### Benchmark (E4B)
| Benchmark | คะแนน |
|-----------|-------|
| MMLU Pro | 69.4% |
| AIME 2026 (no tools) | 42.5% |
| LiveCodeBench v6 | 52.0% |
| GPQA Diamond | 58.6% |
| MMMU (vision) | 76.6% |
| MMMU Pro (vision) | 52.6% |
| CoVoST (audio) | 35.54 |

### วิธีใช้ (จาก README)
```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
```

### Best Practices ที่สำคัญ
1. **Sampling:** temperature=1.0, top_p=0.95, top_k=64 (ตรงกับ generation_config.json)
2. **Thinking:** ใส่ `<|think|>` ใน system prompt เพื่อเปิดโหมดคิด
3. **Modality order:** ภาพก่อนข้อความ, เสียงหลังข้อความ
4. **Token budget:** 70/140/280/560/1120 (ภาพ) — เลือกตามงาน
5. **Audio:** สูงสุด 30 วินาที, Video: สูงสุด 60 วินาที

## 4. สรุป

- README.md คือ "คู่มือมนุษย์" — อ่านแล้วรู้วิธีใช้โมเดลถูกต้อง
- ข้อมูลใน README ตรงกับไฟล์จริง 100% (ตรวจด้วย diagnose-gemma4.py)
- มีโค้ดตัวอย่างครบ (text/image/audio/video) — ใช้เป็น reference ในการพัฒนา BOI X-1 ได้