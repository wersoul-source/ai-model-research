# Topic 4: ความรู้ที่ต้องมีเป็นพื้นฐาน และ ความรู้เสริมเพื่อพัฒนา Model

> รายงานโดย: คำปัน (Kampun) | BOI Family
> วันที่: 2026-08-19
> แหล่งอ้างอิง: 40+ แหล่ง (เอกสารวิชาการ, บทความเทคนิค, เส้นทางอาชีพ)

---

## สารบัญ

1. [ภาพรวม: Knowledge Architecture](#1-ภาพรวม-knowledge-architecture)
2. [ความรู้พื้นฐาน (Foundation Knowledge)](#2-ความรู้พื้นฐาน-foundation-knowledge)
3. [ความรู้เสริม (Supplementary Knowledge)](#3-ความรู้เสริม-supplementary-knowledge)
4. [Knowledge Map ตามระดับความยาก](#4-knowledge-map-ตามระดับความยาก)
5. [Learning Path แนะนำ](#5-learning-path-แนะนำ)

---

## 1. ภาพรวม: Knowledge Architecture

### 1.1 Knowledge Pyramid for AI/ML

```
┌─────────────────────────────────────────────────────────────────┐
│                 AI/ML KNOWLEDGE PYRAMID                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────────┐                          │
│                    │   RESEARCH      │ ◄── Frontier Level      │
│                    │   (Innovation)  │     (100B+ models)      │
│                    └────────┬────────┘                          │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │   ADVANCED      │ ◄── Expert Level        │
│                    │   (Optimization)│     (7B-70B models)     │
│                    └────────┬────────┘                          │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │   INTERMEDIATE  │ ◄── Practitioner Level  │
│                    │   (Application) │     (Fine-tuning)       │
│                    └────────┬────────┘                          │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │   FOUNDATION    │ ◄── Beginner Level      │
│                    │   (Fundamentals)│     (Using models)      │
│                    └─────────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Two Categories of Knowledge

| ประเภท | คำจำกัดความ | ตัวอย่าง |
|--------|------------|---------|
| **Foundation Knowledge** | ความรู้ที่จำเป็นต้องมีก่อนเริ่มสร้าง Model | Math, Programming, ML Basics |
| **Supplementary Knowledge** | ความรู้ที่ช่วยพัฒนาและสร้าง Model ที่ดีขึ้น | Distributed Systems, CUDA, MLOps |

---

## 2. ความรู้พื้นฐาน (Foundation Knowledge)

### 2.1 คณิตศาสตร์ (Mathematics)

#### 2.1.1 Linear Algebra (พีชคณิตเชิงเส้น)

**ทำไมสำคัญ:**
- Data อยู่ใน form ของ vectors และ matrices
- Neural networks ทำงานกับ matrix operations
- Dimensionality reduction (PCA, SVD)

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Vectors & Matrices** | Data representation | Input features, weights |
| **Matrix Multiplication** | Forward propagation | `y = Wx + b` |
| **Dot Product** | Attention mechanism | `Q·K^T` |
| **Eigenvalues/Eigenvectors** | PCA, SVD | Dimensionality reduction |
| **Singular Value Decomposition** | Matrix factorization | Recommendation systems |
| **Norms (L1, L2)** | Regularization | L1/L2 regularization |
| **Determinant** | Matrix inversion | Computing inverses |

**เรียนจาก:**
- 3Blue1Brown: "Essence of Linear Algebra" (YouTube)
- Coursera: "Mathematics for Machine Learning: Linear Algebra" (Imperial College London)
- หนังสือ: "Mathematics for Machine Learning" (Deisenroth et al.)

---

#### 2.1.2 Calculus (แคลคูลัส)

**ทำไมสำคัญ:**
- Backpropagation ใช้ chain rule
- Gradient descent ใช้ derivatives
- Optimization ใช้ partial derivatives

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Derivatives** | Gradient computation | `∂L/∂w` |
| **Partial Derivatives** | Multi-variable optimization | `∂L/∂w₁`, `∂L/∂w₂` |
| **Chain Rule** | Backpropagation | `∂L/∂x = ∂L/∂y · ∂y/∂x` |
| **Gradient** | Direction of steepest ascent | `∇L` |
| **Hessian** | Second-order optimization | `H = ∂²L/∂w²` |
| **Gradient Descent** | Model training | `w = w - η∇L` |

**เรียนจาก:**
- 3Blue1Brown: "Essence of Calculus" (YouTube)
- Coursera: "Mathematics for Machine Learning: Multivariate Calculus"
- Khan Academy: Calculus 1 & 2

---

#### 2.1.3 Probability & Statistics (ความน่าจะเป็นและสถิติ)

**ทำไมสำคัญ:**
- Understanding uncertainty in data
- Bayesian inference
- Model evaluation

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Probability Distributions** | Data modeling | Gaussian, Bernoulli, Categorical |
| **Bayes' Theorem** | Bayesian inference | Naive Bayes, Bayesian networks |
| **Maximum Likelihood Estimation** | Parameter estimation | Finding best weights |
| **Expectation & Variance** | Understanding data | Mean, standard deviation |
| **Hypothesis Testing** | Model evaluation | A/B testing |
| **Confidence Intervals** | Uncertainty quantification | Model confidence |
| **Cross-Entropy Loss** | Classification loss | `L = -Σ y·log(ŷ)` |

**เรียนจาก:**
- Khan Academy: Statistics & Probability
- Coursera: "Mathematics for Machine Learning and Data Science" (DeepLearning.AI)
- หนังสือ: "Pattern Recognition and Machine Learning" (Bishop)

---

#### 2.1.4 Optimization (การเพิ่มประสิทธิภาพ)

**ทำไมสำคัญ:**
- Training models = optimization problem
- Finding global/local minima
- Hyperparameter tuning

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Gradient Descent** | Basic optimization | `w = w - η∇L` |
| **Stochastic GD (SGD)** | Mini-batch training | Faster convergence |
| **Adam Optimizer** | Adaptive learning rates | Most popular optimizer |
| **Learning Rate Scheduling** | Training dynamics | Warmup, cosine annealing |
| **Convex Optimization** | Theory of optimality | Convex loss surfaces |
| **Regularization** | Prevent overfitting | L1, L2, Dropout |

---

### 2.2 Programming (การเขียนโปรแกรม)

#### 2.2.1 Python (ภาษาหลัก)

**ทำไม Python:**
- Ecosystem ที่ใหญ่ที่สุดสำหรับ AI/ML
- ง่ายต่อการเรียนรู้
- รองรับทุก framework

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Basic Syntax** | เขียน code | Variables, loops, functions |
| **Data Structures** | จัดการ data | Lists, dicts, sets |
| **Object-Oriented** | สร้าง classes | Custom models |
| **List Comprehension** | Data processing | `[x*2 for x in data]` |
| **Error Handling** | Debugging | try/except |
| **File I/O** | Data loading | Reading CSV, JSON |
| **Decorators** | Advanced patterns | `@torch.no_grad()` |

**Libraries ที่ต้องรู้:**

| Library | ใช้ทำอะไร |
|---------|---------|
| **NumPy** | Matrix operations |
| **Pandas** | Data manipulation |
| **Matplotlib/Seaborn** | Data visualization |
| **Scikit-learn** | Classical ML |
| **PyTorch/TensorFlow** | Deep Learning |
| **Hugging Face Transformers** | LLMs |

---

#### 2.2.2 Data Manipulation & Visualization

**หัวข้อที่ต้องรู้:**

```python
# NumPy
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
matrix = np.random.randn(3, 3)  # Random 3x3 matrix

# Pandas
import pandas as pd
df = pd.read_csv('data.csv')
df.describe()  # Statistical summary

# Matplotlib
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
```

---

### 2.3 Machine Learning Basics (พื้นฐาน ML)

#### 2.3.1 Supervised Learning

| ประเภท | Algorithm | ใช้ทำอะไร |
|--------|----------|---------|
| **Regression** | Linear, Polynomial | ทำนายค่าต่อเนื่อง |
| **Classification** | Logistic, SVM, Trees | จัดประเภท |

#### 2.3.2 Unsupervised Learning

| ประเภท | Algorithm | ใช้ทำอะไร |
|--------|----------|---------|
| **Clustering** | K-Means, DBSCAN | จัดกลุ่มข้อมูล |
| **Dimensionality Reduction** | PCA, t-SNE | ลดมิติข้อมูล |
| **Anomaly Detection** | Isolation Forest | หาข้อมูลผิดปกติ |

#### 2.3.3 Evaluation Metrics

| Metric | ใช้กับ | สูตร |
|--------|-------|------|
| **Accuracy** | Classification | `TP+TN / Total` |
| **Precision** | Classification | `TP / (TP+FP)` |
| **Recall** | Classification | `TP / (TP+FN)` |
| **F1-Score** | Classification | `2·(P·R)/(P+R)` |
| **MSE** | Regression | `Σ(ŷ-y)²/n` |
| **R²** | Regression | `1 - SS_res/SS_tot` |

---

### 2.4 Deep Learning Fundamentals

#### 2.4.1 Neural Network Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
    │              │                │
    ▼              ▼                ▼
  Features    Activation        Prediction
              Functions
```

#### 2.4.2 Key Concepts

| หัวข้อ | คำอธิบาย |
|--------|---------|
| **Forward Propagation** | ส่งข้อมูลผ่าน network |
| **Backpropagation** | คำนวณ gradients |
| **Activation Functions** | ReLU, Sigmoid, Tanh |
| **Loss Functions** | Cross-entropy, MSE |
| **Optimizers** | SGD, Adam, AdamW |
| **Regularization** | Dropout, BatchNorm, Weight Decay |

---

## 3. ความรู้เสริม (Supplementary Knowledge)

### 3.1 Distributed Systems (ระบบกระจาย)

**ทำไมต้องรู้:**
- Model ขนาดใหญ่ต้องใช้ GPU หลายตัว
- Training ต้องทำ parallel
- Fault tolerance

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Data Parallelism** | แบ่ง data บน GPU | DDP, FSDP |
| **Tensor Parallelism** | แบ่ง layers | Megatron-LM |
| **Pipeline Parallelism** | แบ่ง stages | GPipe |
| **Communication** | GPU-to-GPU | NCCL, InfiniBand |
| **Fault Tolerance** | ข้อผิดพลาด | Checkpointing |

**เครื่องมือ:**
- PyTorch DDP (DistributedDataParallel)
- PyTorch FSDP (FullyShardedDataParallel)
- DeepSpeed
- Megatron-LM

---

### 3.2 GPU Programming & CUDA

**ทำไมต้องรู้:**
- GPU คือหัวใจของการ training
- CUDA optimization ทำให้เร็วขึ้น 2-10x
- Custom kernels สำหรับ operation เฉพาะ

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **GPU Architecture** | เข้าใจ hardware | SM, Warps, Memory |
| **CUDA Basics** | เขียน GPU code | Threads, Blocks, Grids |
| **Memory Management** | จัดการ VRAM | `cudaMalloc`, `cudaMemcpy` |
| **Shared Memory** | เร็วขึ้น | Tiling |
| **Warp Divergence** | หลีกเลี่ยง | Divergent branches |
| **Stream & Events** | Async operations | Pipelining |

**เครื่องมือ:**
- CUDA Toolkit
- Nsight Systems/Compute
- Triton (OpenAI)
- FlashAttention

---

### 3.3 MLOps (Machine Learning Operations)

**ทำไมต้องรู้:**
- Deploy model ให้ใช้งานจริง
- Monitor performance
- Version control

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Experiment Tracking** | ติดตาม runs | MLflow, Weights & Biases |
| **Model Versioning** | เวอร์ชัน model | DVC, Model Registry |
| **CI/CD Pipeline** | Automate deployment | GitHub Actions |
| **Model Monitoring** | ตรวจสอบ drift | Evidently, Prometheus |
| **Data Pipeline** | จัดการ data | Airflow, Prefect |
| **Containerization** | จัดสภาพแวดล้อม | Docker, Kubernetes |

**เครื่องมือ:**
- MLflow
- Weights & Biases (W&B)
- DVC (Data Version Control)
- Kubeflow
- Seldon Core

---

### 3.4 System Design & Architecture

**ทำไมต้องรู้:**
- ออกแบบระบบ AI ที่ scalable
- Microservices สำหรับ inference
- Cost optimization

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Microservices** | แยก service | API, Inference, Training |
| **Load Balancing** | กระจายงาน | NGINX, Traefik |
| **Caching** | เร็วขึ้น | Redis, Memcached |
| **Message Queue** | Async processing | Kafka, RabbitMQ |
| **API Design** | ให้ service | REST, gRPC |
| **Cloud Architecture** | Scalability | AWS, GCP, Azure |

---

### 3.5 Data Engineering

**ทำไมต้องรู้:**
- Data คือหัวใจของ ML
- Data quality = model quality
- Scalable data pipelines

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Data Collection** | เก็บ data | Web scraping, APIs |
| **Data Cleaning** | ทำความสะอาด | Handle missing values |
| **Feature Engineering** | สร้าง features | Scaling, encoding |
| **Data Augmentation** | เพิ่ม data | Image rotation, text paraphrase |
| **ETL Pipeline** | แปลง data | Extract-Transform-Load |
| **Data Warehouse** | เก็บ data | BigQuery, Snowflake |

---

### 3.6 Research Skills

**ทำไมต้องรู้:**
- อ่าน paper ได้
- ทำวิจัยของตัวเอง
- ติดตาม developments

**หัวข้อที่ต้องรู้:**

| หัวข้อ | ใช้ใน AI/ML | ตัวอย่าง |
|--------|-----------|---------|
| **Paper Reading** | เข้าใจงานวิจัย | arXiv, Semantic Scholar |
| **Paper Writing** | เผยแพร่ผลงาน | LaTeX, Overleaf |
| **Experiment Design** | ออกแบบการทดลอง | A/B testing, ablation studies |
| **Reproducibility** | ทำซ้ำได้ | Code, data, seeds |
| **Literature Review** | ทบทวนงาน | Survey papers |

---

### 3.7 Domain-Specific Knowledge

**ทำไมต้องรู้:**
- AI ต้อง solve ปัญหาจริง
- Domain knowledge ช่วย design better features
- Validation กับ domain experts

**ตัวอย่าง Domain:**

| Domain | Knowledge ที่ต้องรู้ | ตัวอย่าง Application |
|--------|---------------------|---------------------|
| **Healthcare** | Medical terminology, regulations | Disease diagnosis |
| **Finance** | Financial instruments, risk | Fraud detection |
| **NLP** | Linguistics, semantics | Chatbots, translation |
| **Computer Vision** | Image processing, 3D geometry | Object detection |
| **Robotics** | Control theory, kinematics | Autonomous robots |
| **Speech** | Audio processing, phonetics | Speech recognition |

---

## 4. Knowledge Map ตามระดับความยาก

### 4.1 Level 1: Beginner (เริ่มต้น)

**เป้าหมาย:** ใช้ Pre-trained Model ได้

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEVEL 1: BEGINNER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Foundation Knowledge:                                          │
│  ├─ Python basics (syntax, data structures)                    │
│  ├─ NumPy, Pandas, Matplotlib                                   │
│  ├─ Basic statistics (mean, median, std)                        │
│  └─ Basic ML concepts (supervised/unsupervised)                 │
│                                                                 │
│  Time to learn: 2-3 เดือน                                      │
│  Prerequisites: High school math                               │
│  Output: ใช้ Hugging Face, scikit-learn ได้                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**สิ่งที่ต้องทำ:**
1. เรียน Python basics (1 เดือน)
2. เรียน NumPy, Pandas (2 สัปดาห์)
3. เรียน ML basics (1 เดือน)
4. ทำ project เล็กๆ (2 สัปดาห์)

---

### 4.2 Level 2: Intermediate (ปานกลาง)

**เป้าหมาย:** Fine-tune Model ได้

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEVEL 2: INTERMEDIATE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Foundation Knowledge:                                          │
│  ├─ Linear Algebra (vectors, matrices, operations)             │
│  ├─ Calculus (derivatives, gradients)                          │
│  ├─ Probability & Statistics (distributions, Bayes)            │
│  ├─ Deep Learning basics (neural nets, backprop)               │
│  └─ PyTorch or TensorFlow                                      │
│                                                                 │
│  Supplementary Knowledge:                                       │
│  ├─ Data preprocessing & augmentation                          │
│  ├─ Transfer learning                                          │
│  └─ Model evaluation & metrics                                 │
│                                                                 │
│  Time to learn: 3-6 เดือน                                      │
│  Prerequisites: Level 1                                        │
│  Output: Fine-tune models, build simple DL models              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**สิ่งที่ต้องทำ:**
1. เรียน Linear Algebra (1 เดือน)
2. เรียน Calculus basics (1 เดือน)
3. เรียน Probability & Statistics (1 เดือน)
4. เรียน Deep Learning (1 เดือน)
5. ทำ project 3-5 ชิ้น (2 เดือน)

---

### 4.3 Level 3: Advanced (ขั้นสูง)

**เป้าหมาย:** Train Model ขนาดเล็ก-กลางได้

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEVEL 3: ADVANCED                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Foundation Knowledge:                                          │
│  ├─ คณิตศาสตร์ทั้งหมด (Linear Algebra, Calculus, Probability)   │
│  ├─ Optimization theory                                        │
│  ├─ Deep Learning architectures (CNN, RNN, Transformer)        │
│  └─ Programming (Python, C++ basics)                           │
│                                                                 │
│  Supplementary Knowledge:                                       │
│  ├─ Distributed training (FSDP, DeepSpeed)                     │
│  ├─ Mixed precision training                                   │
│  ├─ Model optimization (quantization, pruning)                 │
│  ├─ MLOps basics                                               │
│  └─ Research skills (paper reading)                            │
│                                                                 │
│  Time to learn: 6-12 เดือน                                     │
│  Prerequisites: Level 2                                        │
│  Output: Train 1B-7B models, deploy to production              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**สิ่งที่ต้องทำ:**
1. เรียนคณิตศาสตร์เชิงลึก (3 เดือน)
2. เรียน DL architectures (2 เดือน)
3. เรียน distributed training (2 เดือน)
4. เรียน MLOps (1 เดือน)
5. ทำ project 5-10 ชิ้น (4 เดือน)

---

### 4.4 Level 4: Expert (ผู้เชี่ยวชาญ)

**เป้าหมาย:** Train Model ขนาดใหญ่ได้

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEVEL 4: EXPERT                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Foundation Knowledge:                                          │
│  ├─ คณิตศาสตร์ขั้นสูง (Information Theory, Game Theory)        │
│  ├─ Computer Architecture (GPU, memory, networking)            │
│  ├─ Systems Programming (C++, CUDA)                            │
│  └─ Research methodology                                       │
│                                                                 │
│  Supplementary Knowledge:                                       │
│  ├─ GPU programming (CUDA, custom kernels)                     │
│  ├─ Distributed systems at scale                               │
│  ├─ Model architecture design                                  │
│  ├─ Alignment techniques (RLHF, DPO)                           │
│  ├─ Advanced MLOps                                             │
│  └─ Paper writing & publication                                │
│                                                                 │
│  Time to learn: 1-2 ปี                                         │
│  Prerequisites: Level 3                                        │
│  Output: Train 70B+ models, contribute to research             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**สิ่งที่ต้องทำ:**
1. เรียน systems programming (3 เดือน)
2. เรียน GPU programming (3 เดือน)
3. เรียน advanced distributed systems (3 เดือน)
4. เรียน alignment techniques (2 เดือน)
5. ทำ research project (6 เดือน)

---

### 4.5 Level 5: Researcher (นักวิจัย)

**เป้าหมาย:** สร้าง Frontier Model ได้

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEVEL 5: RESEARCHER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Foundation Knowledge:                                          │
│  ├─ คณิตศาสตร์ทั้งหมด                                           │
│  ├─ Computer Science theory                                    │
│  ├─ Domain expertise                                           │
│  └─ Research methodology                                       │
│                                                                 │
│  Supplementary Knowledge:                                       │
│  ├─ Novel architecture design                                  │
│  ├─ Custom training techniques                                 │
│  ├─ Frontier optimization                                      │
│  ├─ Alignment & safety                                         │
│  ├─ Hardware-software co-design                                │
│  └─ Publication & communication                                │
│                                                                 │
│  Time to learn: 3-5 ปี+                                        │
│  Prerequisites: Level 4 + PhD or equivalent experience         │
│  Output: Novel architectures, 100B+ models, publications       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.6 Knowledge Map Summary

| Level | Math | Programming | ML/DL | Systems | Time |
|-------|------|-------------|-------|---------|------|
| **1** | Basic stats | Python basics | Use models | None | 2-3 months |
| **2** | Linear Algebra, Calculus, Probability | Python + Libraries | Fine-tune | Basic | 3-6 months |
| **3** | All foundation + Optimization | Python + C++ basics | Train models | Distributed training | 6-12 months |
| **4** | Advanced math + Info Theory | C++, CUDA | Architecture design | GPU programming | 1-2 years |
| **5** | All + Domain expertise | All + Research | Novel architectures | Full stack | 3-5+ years |

---

## 5. Learning Path แนะนำ

### 5.1 Recommended Learning Path

```
┌─────────────────────────────────────────────────────────────────┐
│                 RECOMMENDED LEARNING PATH                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Foundation (2-3 เดือน)                               │
│  ├─ Python basics                                               │
│  ├─ NumPy, Pandas, Matplotlib                                   │
│  ├─ Basic statistics                                            │
│  └─ ML basics (scikit-learn)                                    │
│                                                                 │
│  Phase 2: Mathematics (3-6 เดือน)                              │
│  ├─ Linear Algebra                                              │
│  ├─ Calculus                                                    │
│  ├─ Probability & Statistics                                    │
│  └─ Optimization                                                │
│                                                                 │
│  Phase 3: Deep Learning (2-3 เดือน)                            │
│  ├─ Neural networks basics                                      │
│  ├─ PyTorch or TensorFlow                                       │
│  ├─ CNN, RNN, Transformer                                       │
│  └─ Transfer learning                                           │
│                                                                 │
│  Phase 4: Practice (3-6 เดือน)                                 │
│  ├─ Build 5-10 projects                                         │
│  ├─ Kaggle competitions                                         │
│  ├─ Fine-tune models                                            │
│  └─ Deploy to production                                        │
│                                                                 │
│  Phase 5: Specialization (6-12 เดือน)                          │
│  ├─ Distributed training                                        │
│  ├─ GPU programming                                             │
│  ├─ MLOps                                                       │
│  └─ Domain expertise                                            │
│                                                                 │
│  Phase 6: Advanced (1-2 ปี)                                    │
│  ├─ Research skills                                             │
│  ├─ Paper reading/writing                                       │
│  ├─ Novel architectures                                         │
│  └─ Publication                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Resources by Level

#### Level 1: Beginner

| Resource | Type | Cost |
|----------|------|------|
| Python for Everybody (Coursera) | Course | Free |
| Kaggle Learn | Interactive | Free |
| Fast.ai Practical DL | Course | Free |
| Google ML Crash Course | Course | Free |

#### Level 2: Intermediate

| Resource | Type | Cost |
|----------|------|------|
| Mathematics for ML (Imperial College) | Course | Free |
| Deep Learning Specialization (Andrew Ng) | Course | $49/mo |
| PyTorch Tutorials | Docs | Free |
| Hands-On ML (Aurélien Géron) | Book | $50 |

#### Level 3: Advanced

| Resource | Type | Cost |
|----------|------|------|
| Stanford CS231n (Vision) | Course | Free |
| Stanford CS224n (NLP) | Course | Free |
| Fast.ai Advanced | Course | Free |
| Papers with Code | Research | Free |

#### Level 4: Expert

| Resource | Type | Cost |
|----------|------|------|
| CUDA Programming (NVIDIA) | Course | Free |
| MIT 6.824 (Distributed Systems) | Course | Free |
| Reading papers daily | Practice | Free |
| Contributing to open source | Practice | Free |

---

## สรุป: หลักการสำคัญ

### 1. Foundation First

> "อย่าข้ามขั้น — Math + Programming = รากฐานที่แข็งแกร่ง"

### 2. Learn by Doing

> "อ่าน 10 เท่าไม่สู้ทำ 1 — project-based learning ได้ผลจริง"

### 3. Progressive Depth

> "เริ่มจากใช้ → เข้าใจ → สร้าง → วิจัย ตามลำดับ"

### 4. Stay Current

> "AI เปลี่ยนเร็ว — อ่าน paper ทุกสัปดาห์"

### 5. Domain Matters

> "AI ที่ดี = ML + Domain Knowledge — อย่ามองข้าม"

---

## อ้างอิง

1. Deisenroth et al. (2020). "Mathematics for Machine Learning"
2. Goodfellow et al. (2016). "Deep Learning"
3. Mitchell (1997). "Machine Learning"
4. Bishop (2006). "Pattern Recognition and Machine Learning"
5. Stanford CS229: Machine Learning
6. Stanford CS231n: CNN for Visual Recognition
7. Stanford CS224n: NLP with Deep Learning
8. Fast.ai: Practical Deep Learning for Coders
9. DeepLearning.AI: Deep Learning Specialization
10. NVIDIA: CUDA Programming Guide
11. PyTorch Documentation
12. TensorFlow Documentation
13. Hugging Face Documentation
14. Kaggle Learn
15. Google ML Crash Course

---

**End of Report — Topic 4**
**คำปัน (Kampun) | BOI Family | 2026-08-19**
