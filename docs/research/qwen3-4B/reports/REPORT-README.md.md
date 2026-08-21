# รายงาน README.md — Qwen3-4B

> ไฟล์: `/root/models/Qwen3-4B/README.md` (16.9 K) | Source: https://huggingface.co/Qwen/Qwen3-4B

## สรุป README

- **Model:** Qwen3-4B — Causal LM, 4.0B (non-embedding 3.6B), 36 layers, GQA 32/8, context 32K → 131K YaRN
- **Highlights:** Thinking/non-thinking ในโมเดลเดียว, reasoning สูงกว่า Qwen2.5, agent/tool calling, 100+ ภาษา
- **Quickstart:** transformers ≥4.51.0, AutoModelForCausalLM + AutoTokenizer, `enable_thinking=True/False`
- **Thinking budget:** YaRN factor 4, vllm/sglang รองรับ
- **Best practices:** thinking temp 0.6/top_k20/top_p0.95, non-thinking 0.7/0.8/20, presence_penalty 0-2
- **Citation:** Qwen3 Technical Report arXiv 2505.09388

## ประเด็นสำคัญที่ README บอก

1. **Qwen2.5-VL → Qwen3 36T tokens** — ใช้ vision model ช่วยดึง text จาก PDF + synthetic data
2. **3-stage pretraining:** 30T (4K) → 5T reasoning (4K) → long context (32K) + ABF/YaRN/DCA
3. **4-stage posttraining:** Long-CoT cold start → Reasoning RL (GRPO) → Thinking Mode Fusion → General RL + Strong-to-Weak Distillation
4. **Evaluation:** Qwen3-4B ชนะ Qwen2.5-7B ใน ~ครึ่ง benchmarks โดยเฉพาะ STEM/coding

## เทียบ Gemma 4 E4B README

- Gemma README: multimodal (image/audio/video) + 256K context + PLE
- Qwen README: thinking mode + 100+ lang + YaRN + agent/tool calling — text specialist

*เนื้อหา README ตรงกับ config/shards จริง: 36 layers, 32/8 GQA, 4.02B, BF16*
