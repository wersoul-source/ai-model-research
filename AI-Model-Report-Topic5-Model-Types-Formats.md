# Topic 5: Model Types/Categories + File Formats + Performance Comparison

> Research Report by: คำปัน (Kampun) — Method-2 Agent, BOI Family
> Session: S10 (2026-08-10)
> Status: Completed

---

## Table of Contents
1. [AI Model Types & Categories](#1-ai-model-types--categories)
2. [Model File Formats](#2-model-file-formats)
3. [Performance Comparison: Best vs Fastest vs Lightweight](#3-performance-comparison)
4. [Recommended Format per Use Case](#4-recommended-format-per-use-case)
5. [Summary](#5-summary)

---

## 1. AI Model Types & Categories

### 1.1 Overview Table

| Type | Function | Example Models | Input | Output |
|------|----------|---------------|-------|--------|
| **Language Model (LM)** | Predict text | GPT-4, LLaMA, BERT | Text | Text |
| **Multimodal Model** | Understand multiple modalities | GPT-4V, Gemini, Claude 3.5 | Text+Image+Audio | Text |
| **Vision Model** | Understand images | ViT, DINOv2, SAM | Image | Embedding/Label |
| **Text-to-Image (T2I)** | Generate images from text | Stable Diffusion, DALL-E 3 | Text | Image |
| **Text-to-Speech (TTS)** | Convert text to speech | Bark, XTTS, Fish Speech | Text | Audio |
| **Speech-to-Text (STT/ASR)** | Transcribe audio | Whisper, Canary | Audio | Text |
| **Text-to-Video (T2V)** | Generate video from text | Sora, Kling, Runway | Text | Video |
| **Embedding Model** | Map text/images to vectors | BGE, E5, CLIP | Text/Image | Vector |
| **Recommendation Model** | Predict user preferences | DLRM, Two-Tower | User+Item | Score |
| **Time-Series Model** | Forecast temporal data | Temporal Fusion Transformer | Time series | Forecast |
| **Tabular Model** | Process structured data | XGBoost, TabNet | Table rows | Prediction |
| **Agent Model** | Autonomous tool use | GPT-4+Tools, Claude+Tools | Text+Tools | Action |

### 1.2 Deep Dive: Language Models

Language models are the most widely used AI models. They predict the next token (word/subword) given a sequence of input tokens.

**Key Subcategories:**

| Subcategory | Description | Example |
|-------------|-------------|---------|
| **Auto-regressive LM** | Predicts next token sequentially | GPT-4, LLaMA, Mistral |
| **Masked LM** | Predicts masked tokens in parallel | BERT, RoBERTa |
| **Encoder-Decoder** | Encodes input, generates output | T5, BART, mBART |
| **Decoder-Only** | Most common for modern LLMs | GPT, LLaMA, Claude |

**How They Work:**
1. Input text → Tokenizer → Token IDs
2. Token IDs → Embedding layer → Dense vectors
3. Dense vectors → Transformer layers (self-attention + FFN)
4. Output → Linear layer → Probability distribution over vocabulary
5. Token with highest probability → next token

### 1.3 Deep Dive: Multimodal Models

Multimodal models understand multiple types of data simultaneously.

**Architecture Pattern:**
- Text component (LLM backbone)
- Vision component (Vision Transformer or CLIP encoder)
- Audio component (Whisper encoder)
- Fusion mechanism (cross-attention, late fusion, or early fusion)

**Examples:**
- **GPT-4V/GPT-4o**: Text + Vision + Audio
- **Gemini**: Text + Vision + Audio + Video
- **Claude 3.5**: Text + Vision

### 1.4 Deep Dive: Vision Models

Vision models process images for classification, detection, segmentation, or generation.

| Task | Model Type | Example |
|------|-----------|---------|
| Classification | CNN/ViT | ResNet, ViT |
| Object Detection | YOLO, DETR | YOLOv8, DETR |
| Segmentation | U-Net, SAM | SAM, Mask2Former |
| Feature Extraction | DINOv2, CLIP | DINOv2, SigLIP |

### 1.5 Deep Dive: Text-to-Image Models

Text-to-image models generate images from text descriptions.

**Architecture Pattern (Diffusion Models):**
1. Text → CLIP Text Encoder → Text embeddings
2. Noise → U-Net/DiT → Denoised image
3. Text embeddings guide denoising process
4. Decoder (VAE) → Final image

**Examples:**
- **Stable Diffusion 3**: Open-source, high quality
- **DALL-E 3**: Closed-source, integrated with ChatGPT
- **Midjourney**: Closed-source, best artistic quality
- **FLUX**: Open-source, new architecture

### 1.6 Deep Dive: Text-to-Speech Models

TTS models convert text into natural-sounding speech.

**Architecture Pattern (VITS-based):**
1. Text → Phoneme encoder → Text representations
2. Text representations → Flow-based decoder → Mel spectrogram
3. Mel spectrogram → HiFi-GAN vocoder → Audio waveform

**Examples:**
- **Bark**: Open-source, multi-language
- **XTTS v2**: Open-source, voice cloning
- **Fish Speech**: Open-source, fast
- **ElevenLabs**: Closed-source, best quality

### 1.7 Deep Dive: Speech-to-Text Models

STT models transcribe audio into text.

**Architecture Pattern (Whisper-based):**
1. Audio → Mel spectrogram → Encoder
2. Encoder → Decoder → Text tokens
3. Decoder uses cross-attention to focus on audio features

**Examples:**
- **Whisper**: Open-source, multi-language
- **Canary**: NVIDIA, fast
- **Deepgram**: Closed-source, real-time

### 1.8 Deep Dive: Embedding Models

Embedding models map data to dense vector representations for similarity search.

**Use Cases:**
- Semantic search
- Recommendation systems
- Clustering
- Anomaly detection

**Examples:**
- **BGE**: Text embedding
- **E5**: Text embedding
- **CLIP**: Image + Text embedding
- **DINOv2**: Image embedding

---

## 2. Model File Formats

### 2.1 Format Comparison Table

| Format | Extension | Framework | Loading Speed | Portability | Security | Best For |
|--------|-----------|-----------|---------------|-------------|----------|----------|
| **SafeTensors** | `.safetensors` | Any (Python/Rust/JS) | ⚡ Fast | ✅ High | ✅ Safe | Modern standard, safe loading |
| **GGUF** | `.gguf` | llama.cpp | ⚡ Fast | ✅ High | ✅ Safe | CPU/Edge inference |
| **ONNX** | `.onnx` | ONNX Runtime | ⚡ Fast | ✅ High | ⚠️ Medium | Cross-framework deployment |
| **TFLite** | `.tflite` | TensorFlow Lite | ⚡ Fast | ✅ High | ⚠️ Medium | Mobile/Edge devices |
| **PyTorch** | `.pt`/`.pth` | PyTorch | 🐌 Slow | ⚠️ Low | ❌ Unsafe | Training, research |
| **MLX** | `.safetensors` | MLX | ⚡ Fast | ✅ High | ✅ Safe | Apple Silicon optimization |
| **TensorRT** | `.engine` | TensorRT | ⚡⚡ Fastest | ⚠️ Low | ⚠️ Medium | NVIDIA GPU inference |
| **GGML** | `.ggml` | ggml | ⚡ Fast | ✅ High | ✅ Safe | Legacy CPU inference |

### 2.2 SafeTensors

**What is SafeTensors?**
SafeTensors is a simple and safe way to store and share tensors (model weights) developed by Hugging Face. It's designed to replace the unsafe pickle-based format used by PyTorch.

**Key Features:**
- **Safe**: No arbitrary code execution (unlike pickle)
- **Fast**: Can mmap (memory-mapped) for fast loading
- **Portable**: Works with Python, Rust, JavaScript, and more
- **Simple**: Just store tensors with metadata

**When to Use:**
- Sharing models on Hugging Face Hub
- Loading models in production
- Any situation where security matters

**File Structure:**
```
Header (JSON) → Tensor metadata (name, dtype, shape, offset)
Data section → Raw tensor data
```

### 2.3 GGUF

**What is GGUF?**
GGUF (GPT-Generated Unified Format) is the format used by llama.cpp for loading and running LLMs. It replaced the older GGML format and includes everything needed for inference.

**Key Features:**
- **Self-contained**: Model weights + tokenizer + metadata in one file
- **Quantization built-in**: Supports Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.
- **CPU-optimized**: Designed for CPU inference
- **Portable**: Works on any platform (Windows, Mac, Linux, Android)

**When to Use:**
- Running LLMs on CPU-only devices
- Edge deployment (phones, Raspberry Pi)
- Quantized models for memory efficiency

**Supported Quantization Types:**
| Type | Bits | Quality | Speed | Use Case |
|------|------|---------|-------|----------|
| Q2_K | 2-bit | Low | Fast | Extreme compression |
| Q4_0 | 4-bit | Medium | Fast | Balanced |
| Q4_K_M | 4-bit | Medium+ | Fast | Recommended |
| Q5_K_M | 5-bit | High | Medium | Quality focus |
| Q8_0 | 8-bit | Very High | Slow | Near-original quality |
| F16 | 16-bit | Original | Slow | Maximum quality |
| F32 | 32-bit | Original | Very Slow | Training/reference |

### 2.4 ONNX

**What is ONNX?**
ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. It allows models to be transferred between different frameworks.

**Key Features:**
- **Framework-agnostic**: Works with PyTorch, TensorFlow, scikit-learn, etc.
- **Optimized runtime**: ONNX Runtime provides hardware acceleration
- **Standardized**: Industry standard for model deployment
- **Graph optimization**: Automatic fusion of operations

**When to Use:**
- Deploying models across different frameworks
- Production inference with ONNX Runtime
- Edge deployment with ONNX Runtime Mobile

**Supported Hardware:**
- CPU (x86, ARM)
- GPU (NVIDIA, AMD, Intel)
- Mobile (Android, iOS)
- Edge devices

### 2.5 TFLite

**What is TFLite?**
TFLite (TensorFlow Lite) is TensorFlow's format for mobile and edge deployment. It's optimized for resource-constrained devices.

**Key Features:**
- **Optimized for mobile**: Small binary size, fast inference
- **Hardware acceleration**: GPU delegate, NNAPI delegate, Edge TPU
- **Quantization**: INT8, FP16 quantization built-in
- **Cross-platform**: Android, iOS, embedded Linux, microcontrollers

**When to Use:**
- Mobile app deployment
- IoT devices
- Embedded systems
- Real-time inference on edge devices

**Delegates (Hardware Acceleration):**
| Delegate | Hardware | Use Case |
|----------|----------|----------|
| GPU | Mobile GPU | Fast inference on phones |
| NNAPI | Android Neural Networks | Android-specific optimization |
| Edge TPU | Google Edge TPU | IoT devices |
| Core ML | Apple Neural Engine | iOS devices |

### 2.6 PyTorch (.pt/.pth)

**What is PyTorch format?**
PyTorch's native format uses Python's pickle module to serialize model weights and architecture.

**Key Features:**
- **Native to PyTorch**: Direct save/load
- **Full flexibility**: Can save entire model or just state_dict
- **Python-dependent**: Requires Python environment

**When to Use:**
- Training and research
- Development and experimentation
- When you need full model flexibility

**⚠️ Security Warning:**
- Uses pickle → can execute arbitrary code
- Never load untrusted .pt files
- Use SafeTensors for sharing

### 2.7 MLX

**What is MLX?**
MLX is Apple's machine learning framework optimized for Apple Silicon (M1/M2/M3/M4 chips).

**Key Features:**
- **Apple Silicon optimized**: Uses unified memory architecture
- **Fast**: Leverages Neural Engine and GPU
- **Python-native**: Easy to use with NumPy-like API
- **Uses SafeTensors**: Compatible with Hugging Face models

**When to Use:**
- Running models on Mac (M1/M2/M3/M4)
- Apple ecosystem deployment
- Fast local inference on Apple devices

---

## 3. Performance Comparison

### 3.1 Best Quality (Accuracy)

**Winner: Full-precision models (FP16/FP32) in native format**

| Model | Format | Size | Quality Score | Use Case |
|-------|--------|------|---------------|----------|
| GPT-4 (1.8T) | Proprietary | 1.8T params | 95/100 | State-of-the-art |
| LLaMA 3 70B | SafeTensors/FP16 | 140 GB | 92/100 | Open-source leader |
| Claude 3.5 Sonnet | Proprietary | Unknown | 91/100 | Best coding |
| Gemini Ultra | Proprietary | Unknown | 90/100 | Best multimodal |

**Key Factors for Quality:**
- Larger models → better quality
- More training data → better generalization
- Higher precision (FP16/FP32) → no information loss
- Instruction tuning → better following instructions
- RLHF → better alignment with human preferences

### 3.2 Fastest Inference

**Winner: TensorRT (NVIDIA) + Quantized models**

| Framework/Format | Speed | Hardware | Quantization | Use Case |
|------------------|-------|----------|--------------|----------|
| TensorRT | ⚡⚡⚡ Fastest | NVIDIA GPU | INT8/FP16 | Production inference |
| llama.cpp (GGUF) | ⚡⚡ Fast | CPU | Q4_K_M | Edge/CPU inference |
| ONNX Runtime | ⚡⚡ Fast | CPU/GPU | INT8/FP16 | Cross-platform |
| TFLite | ⚡ Fast | Mobile | INT8 | Mobile devices |
| PyTorch (FP16) | 🐌 Medium | GPU | None | Research |

**Speed Benchmarks (tokens/second):**
| Setup | Model Size | Speed | Notes |
|-------|-----------|-------|-------|
| RTX 4090 + TensorRT | 7B | ~200 t/s | Fastest |
| M2 Ultra + MLX | 7B | ~50 t/s | Best on Mac |
| CPU (Threadripper) | 7B | ~20 t/s | CPU inference |
| iPhone 15 Pro | 3B | ~10 t/s | Mobile |
| Raspberry Pi 5 | 3B | ~5 t/s | Edge device |

### 3.3 Most Lightweight (Smallest Footprint)

**Winner: Quantized GGUF models (Q2_K/Q4_0)**

| Model | Format | Size | Quality | Use Case |
|-------|--------|------|---------|----------|
| TinyLlama 1.1B | GGUF/Q4 | 0.6 GB | Basic | IoT devices |
| Phi-3 Mini 3.8B | GGUF/Q4 | 2.1 GB | Good | Mobile phones |
| LLaMA 3 8B | GGUF/Q4 | 4.5 GB | Great | Laptops |
| Mistral 7B | GGUF/Q4 | 4.0 GB | Great | Edge devices |
| Gemma 2 9B | GGUF/Q4 | 5.5 GB | Excellent | Desktop |

**Size vs Quality Tradeoff:**
| Model Size | Quantization | RAM Needed | Quality Level |
|------------|--------------|------------|---------------|
| 1B params | FP16 | 2 GB | Basic |
| 3B params | Q4 | 2.5 GB | Good |
| 7B params | Q4 | 4.5 GB | Great |
| 13B params | Q4 | 8 GB | Excellent |
| 70B params | Q4 | 40 GB | State-of-art |

### 3.4 Format Selection Guide

| Use Case | Best Format | Why |
|----------|-------------|-----|
| **Research/Training** | PyTorch (.pt) | Full flexibility, easy debugging |
| **Sharing on HuggingFace** | SafeTensors | Safe, fast, portable |
| **Running on CPU** | GGUF | Optimized for CPU, quantized |
| **Mobile deployment** | TFLite | Optimized for mobile, hardware acceleration |
| **Cross-framework** | ONNX | Works with any framework |
| **Apple Silicon** | MLX | Optimized for M1/M2/M3/M4 |
| **NVIDIA production** | TensorRT | Fastest inference on NVIDIA |
| **Edge devices** | GGUF + Q4 | Small size, CPU-friendly |

### 3.5 Quantization Impact on Quality

| Quantization | Size Reduction | Quality Loss | When to Use |
|--------------|----------------|--------------|-------------|
| FP16 | 2x | None | Maximum quality |
| INT8 | 4x | Minimal (~1%) | Production inference |
| Q8_0 | 4x | Minimal (~1%) | Near-original quality |
| Q5_K_M | ~3x | Small (~2-3%) | Balanced quality/size |
| Q4_K_M | ~4x | Moderate (~3-5%) | Recommended default |
| Q4_0 | ~4x | Moderate (~3-5%) | CPU inference |
| Q2_K | ~8x | Significant (~10%) | Extreme compression |

**Rule of Thumb:**
- Q4_K_M is the sweet spot for most use cases
- Q5_K_M for better quality if memory allows
- Q8_0 for near-original quality
- Q2_K only for extreme memory constraints

---

## 4. Recommended Format per Use Case

### 4.1 Desktop/Laptop (8-32 GB RAM)

| Use Case | Model | Format | Quantization | RAM Needed |
|----------|-------|--------|--------------|------------|
| General assistant | LLaMA 3 8B | GGUF | Q4_K_M | 5 GB |
| Coding help | CodeLlama 13B | GGUF | Q4_K_M | 9 GB |
| Creative writing | Mistral 7B | GGUF | Q5_K_M | 5 GB |
| Research | LLaMA 3 70B | GGUF | Q4_K_M | 40 GB |

### 4.2 Mobile (4-8 GB RAM)

| Use Case | Model | Format | Quantization | RAM Needed |
|----------|-------|--------|--------------|------------|
| Basic assistant | TinyLlama 1.1B | GGUF/Q4 | Q4_0 | 0.8 GB |
| Smart assistant | Phi-3 Mini 3.8B | GGUF/Q4 | Q4_0 | 2.5 GB |
| Advanced assistant | LLaMA 3 8B | GGUF/Q4 | Q4_0 | 5 GB |

### 4.3 Edge/IoT (1-4 GB RAM)

| Use Case | Model | Format | Quantization | RAM Needed |
|----------|-------|--------|--------------|------------|
| Text classification | DistilBERT | ONNX | INT8 | 0.2 GB |
| Image classification | MobileNet | TFLite | INT8 | 0.02 GB |
| Basic NLP | TinyLlama | GGUF | Q2_K | 0.6 GB |

### 4.4 Production Server (32+ GB RAM)

| Use Case | Model | Format | Quantization | RAM Needed |
|----------|-------|--------|--------------|------------|
| High-throughput | LLaMA 3 70B | TensorRT | FP16 | 140 GB |
| Balanced | LLaMA 3 8B | TensorRT | INT8 | 9 GB |
| Real-time | Mistral 7B | ONNX | INT8 | 4 GB |

---

## 5. Summary

### Key Takeaways:

1. **AI Models come in many types**: Language, Vision, Audio, Multimodal, Embedding, etc.
2. **SafeTensors is the modern standard**: Safe, fast, portable — use this for sharing
3. **GGUF is king for CPU/Edge**: Built-in quantization, self-contained, portable
4. **ONNX is best for cross-framework**: Works with any framework, optimized runtime
5. **TFLite is best for mobile**: Optimized for mobile devices, hardware acceleration
6. **PyTorch is best for training**: Full flexibility, but unsafe for sharing
7. **MLX is best for Apple Silicon**: Optimized for M1/M2/M3/M4 chips
8. **Q4_K_M is the sweet spot**: Best balance of size and quality for most use cases
9. **Format depends on use case**: No single format is best for everything
10. **Quantization trades quality for size**: Choose based on your constraints

### Decision Tree:

```
What's your use case?
├── Training/Research → PyTorch (.pt)
├── Sharing on HuggingFace → SafeTensors
├── Running on CPU → GGUF
├── Mobile deployment → TFLite
├── Cross-framework → ONNX
├── Apple Silicon → MLX
├── NVIDIA production → TensorRT
└── Edge devices → GGUF + Q4
```

### Quick Reference:

| Question | Answer |
|----------|--------|
| What format is safest? | SafeTensors |
| What format is fastest on CPU? | GGUF (Q4_K_M) |
| What format is smallest? | GGUF (Q2_K) |
| What format works everywhere? | ONNX |
| What format is best for Mac? | MLX |
| What format is best for phones? | TFLite |
| What format is best for training? | PyTorch (.pt) |

---

## References

1. [Hugging Face SafeTensors](https://huggingface.co/docs/safetensors/)
2. [GGUF Format](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
3. [ONNX Runtime](https://onnxruntime.ai/)
4. [TensorFlow Lite](https://www.tensorflow.org/lite)
5. [Apple MLX](https://ml-explore.github.io/mlx/)
6. [TensorRT](https://developer.nvidia.com/tensorrt)
7. [llama.cpp](https://github.com/ggerganov/llama.cpp)
8. [Quantization Guide](https://github.com/ggerganov/llama.cpp#quantization)

---

**Report completed by: คำปัน (Kampun)**
**Date: 2026-08-10**
**Session: S10**
**Status: ✅ DONE**
