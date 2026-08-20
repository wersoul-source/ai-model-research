# รายงานการศึกษา: processor_config.json — ระบบประมวลผล Multimodal

> Session: 2026-08-20-S39 | Skill: diagnose-gemma-4 | ขนาดไฟล์: 1.6 KB

## 1. ภาพรวม

`processor_config.json` กำหนดวิธีแปลง **input ดิบ (ภาพ/เสียง/วิดีโอ/ข้อความ) → tokens** ที่โมเดลเข้าใจ ทำงานคู่กับ `tokenizer.json` ผ่าน `Gemma4Processor` — เป็น "หูตา" ของโมเดล

## 2. โครงสร้าง

```
processor_class: Gemma4Processor
├── feature_extractor  (เสียง) — Gemma4AudioFeatureExtractor
├── image_processor    (ภาพ)  — Gemma4ImageProcessor
├── video_processor    (วิดีโอ) — Gemma4VideoProcessor
└── ค่าร่วม: audio_ms_per_token, audio_seq_length, image_seq_length
```

## 3. Audio Feature Extractor (เสียง)

| ค่า | ค่า | ความหมาย |
|-----|-----|----------|
| sampling_rate | 16000 Hz | ความถี่ตัวอย่างเสียง |
| fft_length | 512 | ขนาด FFT window |
| hop_length | 160 | ระยะเลื่อน (10ms) |
| frame_length | 320 | ขนาด frame (20ms) |
| feature_size | 128 | จำนวน mel bins |
| min/max_frequency | 0 - 8000 Hz | ช่วงความถี่ |
| mel_floor | 0.001 | ค่าต่ำสุดของ mel |
| preemphasis | 0.0 | ไม่ใช้ pre-emphasis |

**การคำนวณ:**
- 30 วินาทีเสียง = 750 tokens (`audio_seq_length`)
- 1 token = 40ms (`audio_ms_per_token`)
- 16kHz ÷ 160 hop = 100 frames/วินาที → 128 mel bins/frame

## 4. Image Processor (ภาพ)

| ค่า | ค่า | ความหมาย |
|-----|-----|----------|
| patch_size | 16 | แบ่งภาพเป็น patch 16×16 |
| pooling_kernel_size | 3 | pooling หลัง vision encoder |
| image_seq_length | 280 | soft tokens ต่อภาพ |
| max_soft_tokens | 280 | งบสูงสุด (ปรับได้ 70-1120) |
| rescale_factor | 0.0039 (1/255) | ปรับ pixel 0-255 → 0-1 |
| do_normalize | false | ไม่ normalize (ค่า mean/std = 0/1) |
| resample | 3 | bicubic interpolation |

**การทำงาน:** ภาพ → แบ่ง patch 16×16 → vision tower → 280 soft tokens → ผ่าน `embed_vision` เข้า LM

## 5. Video Processor (วิดีโอ)

| ค่า | ค่า | ความหมาย |
|-----|-----|----------|
| num_frames | 32 | จำนวน frame สูงสุด |
| max_soft_tokens | 70 | tokens ต่อ frame (ประหยัดกว่า image) |
| do_normalize | true | normalize (ต่างจากภาพนิ่ง) |
| patch_size | 16 | เหมือน image |

**การทำงาน:** วิดีโอ → sample 32 frames → แต่ละ frame 70 tokens → รวม 2,240 tokens สูงสุด (60 วินาที @ 1fps)

## 6. สรุป

- ไฟล์นี้คือ "ตัวแปลงสัญญาณ" — ทำให้โมเดลเข้าใจโลกภายนอก (ภาพ/เสียง/วิดีโอ)
- จุดสำคัญ: **image 280 tokens แต่ video แค่ 70 tokens/frame** — เพราะวิดีโอมีหลาย frame ต้องประหยัดต่อ frame
- ค่าเหล่านี้ต้องตรงกับตอนฝึกโมเดล — เปลี่ยนแล้วผลลัพธ์จะเพี้ยน