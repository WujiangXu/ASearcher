# ASearcher 安装指南

本指南帮助你在本地服务器上使用conda安装和配置ASearcher环境。

## 快速安装

### 方法1：使用environment.yml（推荐）

```bash
# 1. 创建conda环境
conda env create -f environment.yml

# 2. 激活环境
conda activate asearcher

# 3. 安装NLTK数据（用于句子分词）
python -c "import nltk; nltk.download('punkt')"

# 4. 验证安装
python -c "import torch; import transformers; import areal; print('✓ Installation successful!')"
```

### 方法2：手动安装

```bash
# 1. 创建Python 3.10环境
conda create -n asearcher python=3.10 -y
conda activate asearcher

# 2. 安装PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. 安装核心依赖
pip install areal transformers>=4.35.0 datasets>=2.14.0 accelerate

# 4. 安装其他依赖（见下文完整列表）
pip install -r requirements.txt
```

## 详细依赖说明

### 核心依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `python` | 3.10 | Python运行环境 |
| `pytorch` | >=2.1.0 | 深度学习框架 |
| `areal` | latest | ASearcher的异步RL框架 |
| `transformers` | >=4.35.0 | HuggingFace模型库 |
| `datasets` | >=2.14.0 | 数据集加载 |

### 可选依赖

#### 1. 用于概念编码（离线预处理）

```bash
pip install sonar-space nltk
python -c "import nltk; nltk.download('punkt')"
```

#### 2. 用于QA生成（OpenAI API）

```bash
pip install openai>=1.0.0
export OPENAI_API_KEY='your-api-key'
```

#### 3. 用于本地LLM服务（SGLang）

```bash
pip install sglang[all]>=0.1.0
```

#### 4. 用于向量检索（FAISS）

```bash
# CPU版本
pip install faiss-cpu

# 或GPU版本（更快）
conda install -c conda-forge faiss-gpu
```

## 完整requirements.txt

创建 `requirements.txt` 文件：

```txt
# Core RL Framework
areal

# Transformers & NLP
transformers>=4.35.0
tokenizers>=0.15.0
datasets>=2.14.0
accelerate>=0.24.0
sentencepiece
protobuf

# Tensor & Data Processing
tensordict>=0.2.0
torchdata>=0.7.0

# Vector Search
faiss-cpu>=1.7.4

# API & Web
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
aiohttp>=3.9.0
requests>=2.31.0

# SONAR (for concept encoding)
sonar-space
nltk>=3.8.0

# OpenAI API (for QA synthesis)
openai>=1.0.0

# Evaluation
evaluate>=0.4.0

# Utilities
tqdm>=4.66.0
pyyaml>=6.0
python-dotenv
numpy>=1.24.0
scipy
```

然后安装：

```bash
pip install -r requirements.txt
```

## 验证安装

### 1. 基础验证

```bash
conda activate asearcher

# 检查Python版本
python --version  # 应该是 Python 3.10.x

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查核心库
python -c "import areal, transformers, datasets; print('✓ Core libraries OK')"
```

### 2. 检查OpenAI API（如果需要用GPT生成数据）

```bash
export OPENAI_API_KEY='your-api-key'
python qa_synthesis/openai_client.py
```

### 3. 检查SONAR（如果需要概念编码）

```bash
bash preprocessing/quick_test.sh
```

### 4. 完整集成测试

```bash
# 测试预处理管道（需要GPU）
python preprocessing/test_preprocessing.py

# 测试agent
cd evaluation/
python test_agent.py  # 如果有测试脚本
```

## 常见问题

### 问题1: CUDA版本不匹配

```bash
# 检查你的CUDA版本
nvidia-smi

# 如果是CUDA 11.8，改用：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 如果是CUDA 12.4+，改用：
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 问题2: AReaL安装失败

AReaL是ASearcher的核心框架。如果 `pip install areal` 失败：

```bash
# 从源码安装
git clone https://github.com/inclusionAI/AReaL.git
cd AReaL
pip install -e .
```

官方文档：https://inclusionai.github.io/AReaL/tutorial/installation.html

### 问题3: SONAR安装问题

```bash
# 如果sonar-space安装失败，尝试：
pip install --upgrade pip
pip install sonar-space --no-cache-dir

# 或从conda-forge安装依赖
conda install -c conda-forge fairseq2
pip install sonar-space
```

### 问题4: FAISS导入错误

```bash
# 卸载旧版本
pip uninstall faiss faiss-cpu faiss-gpu -y

# CPU版本
pip install faiss-cpu

# 或GPU版本
conda install -c conda-forge faiss-gpu
```

### 问题5: 内存不足

如果GPU内存不足：

1. 减小batch size
2. 使用gradient checkpointing
3. 使用更小的模型（7B而不是32B）
4. 使用CPU offload

## 环境变量配置

创建 `.env` 文件：

```bash
# API Keys
SERPER_API_KEY=your_serper_api_key
JINA_API_KEY=your_jina_api_key
OPENAI_API_KEY=your_openai_api_key  # 如果使用OpenAI

# Model Paths
MODEL_PATH=/path/to/your/models
DATA_DIR=/path/to/your/data

# Training Config
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TOKENIZERS_PARALLELISM=false
```

加载环境变量：

```bash
# 在运行脚本前
export $(cat .env | xargs)

# 或使用python-dotenv
pip install python-dotenv
```

## 针对不同使用场景的最小安装

### 场景1: 只做评估（不训练）

```bash
conda create -n asearcher-eval python=3.10 -y
conda activate asearcher-eval
pip install torch transformers datasets vllm fastapi uvicorn aiohttp requests tqdm
```

### 场景2: 只做QA生成（用OpenAI API）

```bash
conda create -n asearcher-qa python=3.10 -y
conda activate asearcher-qa
pip install openai aiohttp requests tqdm transformers
```

### 场景3: 离线预处理（概念编码）

```bash
conda create -n asearcher-prep python=3.10 -y
conda activate asearcher-prep
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install sonar-space nltk tqdm numpy
python -c "import nltk; nltk.download('punkt')"
```

### 场景4: 完整训练环境

使用上面的 `environment.yml` 完整安装。

## 下一步

安装完成后，参考以下文档：

1. **QA生成**: `qa_synthesis/OPENAI_USAGE.md` - 使用OpenAI API生成训练数据
2. **预处理**: `preprocessing/GETTING_STARTED.md` - 离线概念编码
3. **训练**: `docs/training.md` - RL训练指南
4. **评估**: `docs/evaluation.md` - 模型评估

## 获取帮助

- ASearcher GitHub: https://github.com/inclusionAI/ASearcher
- AReaL 文档: https://inclusionai.github.io/AReaL/
- HuggingFace: https://huggingface.co/collections/inclusionAI/asearcher-6891d8acad5ebc3a1e1fb2d1

祝训练顺利！🚀
