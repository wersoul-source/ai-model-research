# รายงานการศึกษา: .gitattributes — กฎ Git LFS

> Session: 2026-08-20-S39 | Skill: diagnose-gemma-4 | ขนาดไฟล์: 1.5 KB (36 บรรทัด)

## 1. ภาพรวม

`.gitattributes` คือ **กฎของ Git** ที่บอกว่าไฟล์ประเภทไหนต้องเก็บผ่าน **Git LFS (Large File Storage)** — ไฟล์ใหญ่จะถูกเก็บ pointer ใน git และตัวจริงอยู่ที่เซิร์ฟเวอร์ LFS แยกต่างหาก

## 2. เนื้อหา (36 rules ทั้งหมด)

```
*.7z, *.arrow, *.bin, *.bz2, *.ckpt, *.ftz, *.gz, *.h5, *.joblib,
*.lfs.*, *.mlmodel, *.model, *.msgpack, *.npy, *.npz, *.onnx, *.ot,
*.parquet, *.pb, *.pickle, *.pkl, *.pt, *.pth, *.rar, *.safetensors,
saved_model/**, *.tar.*, *.tar, *.tflite, *.tgz, *.wasm, *.xz, *.zip,
*.zst, *tfevents*, tokenizer.json
```

ทุก rule มีรูปแบบเดียวกัน:
```
*.safetensors filter=lfs diff=lfs merge=lfs -text
```

## 3. วิเคราะห์

### ความหมายของแต่ละส่วน
| ส่วน | ความหมาย |
|------|----------|
| `*.safetensors` | ไฟล์ที่ match |
| `filter=lfs` | ใช้ LFS filter (เก็บ pointer แทนเนื้อหา) |
| `diff=lfs` | ใช้ LFS diff driver |
| `merge=lfs` | ใช้ LFS merge driver |
| `-text` | ไม่ treat เป็น text (binary) |

### กลุ่มไฟล์ที่ครอบคลุม
| กลุ่ม | นามสกุล | ตัวอย่าง |
|-------|---------|----------|
| โมเดล | .safetensors, .pt, .pth, .ckpt, .bin, .onnx | น้ำหนักโมเดล |
| ข้อมูล | .parquet, .npy, .npz, .arrow | dataset |
| บีบอัด | .zip, .tar, .gz, .7z, .xz, .zst | archive |
| อื่น | .pb, .pickle, .pkl, .msgpack | serialized |

### ไฟล์สำคัญของ Gemma ที่ถูก LFS
- `model.safetensors` (15.99 GB) → ผ่าน `*.safetensors`
- `tokenizer.json` (31 MB) → ผ่าน rule พิเศษ `tokenizer.json` (ไม่ match กับ rule อื่น)

## 4. ทำไมต้องมี

- **Git ธรรมดาไม่เหมาะกับไฟล์ใหญ่** — repo จะบวม, clone ช้า, memory เต็ม
- **LFS เก็บ pointer (~100 bytes)** ใน git แทนไฟล์จริง (15.99 GB) → repo เล็ก, clone เร็ว
- **HF Hub ใช้ LFS เป็นมาตรฐาน** — ทุกโมเดลบน Hub ต้องมีไฟล์นี้

## 5. สรุป

- .gitattributes คือ "กฎการจัดเก็บ" — ทำให้ repo จัดการไฟล์ใหญ่ได้
- ไฟล์นี้ไม่เกี่ยวกับการทำงานของโมเดล — เกี่ยวกับการ version control
- ถ้าไม่มีไฟล์นี้ → git จะพยายาม commit ไฟล์ 16GB ตรง ๆ → repo พัง