# รายงาน .gitattributes — Qwen3-4B

> ไฟล์: `/root/models/Qwen3-4B/.gitattributes` (1.6 K) | LFS rules

## เนื้อหาไฟล์

```
*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
... (32 lines)
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
tokenizer.json filter=lfs diff=lfs merge=lfs -text
```

## วิเคราะห์

- **เหมือน Gemma:** กฎ LFS เดียวกัน — `*.safetensors` + `tokenizer.json` เป็น LFS — เพราะไฟล์ใหญ่ (3.9G/shard + 11M tokenizer)
- **Sharded:** Qwen มี `model-0000X.safetensors` 3 ไฟล์ — แต่ละไฟล์ถูก LFS track แยกกัน — ทำให้ push/pull ทีละ shard

*ไฟล์นี้ไม่เกี่ยวกับ model behavior — แค่บอก git ว่าไฟล์ไหนเก็บใน LFS*
