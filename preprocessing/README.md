# Offline Concept Preprocessing for ASearcher

This directory contains tools for preprocessing datasets with SONAR concept embeddings **offline**, enabling 10-100× faster RL training by eliminating encoding overhead during training.

## 📋 Overview

**Problem**: Online encoding during RL training is slow (5-27 min per batch) and causes GPU memory conflicts.

**Solution**: Preprocess all datasets once with SONAR embeddings, then use preprocessed data for training.

**Benefits**:
- ✅ 10-100× faster training (no encoding overhead)
- ✅ No GPU memory conflicts (SONAR not loaded during training)
- ✅ Deterministic and reproducible
- ✅ Easy to debug and validate

## 🧪 Testing (Recommended First Step)

Before preprocessing real data, verify your setup works correctly:

### Quick Smoke Test (30 seconds)

```bash
cd preprocessing/

# Test on CPU (no GPU needed)
bash quick_test.sh cpu

# Or test on GPU
bash quick_test.sh cuda:0
```

**Expected output**:
```
✓ Python found
✓ SONAR installed
✓ NLTK installed
✓ NumPy installed
✓ SONAR encoder loaded successfully
✓ Encoding successful, shape: (2, 1024)
✓ ALL CHECKS PASSED!
```

### Full Integration Test (2-3 minutes)

```bash
# Test complete workflow with sample data
python test_preprocessing.py --device cpu

# Or on GPU
python test_preprocessing.py --device cuda:0

# Keep test data for inspection
python test_preprocessing.py --device cpu --keep
```

**What it tests**:
1. ✅ Creates sample test data (3 questions)
2. ✅ Runs preprocessing with SONAR
3. ✅ Validates embedding quality
4. ✅ Tests loading processed data
5. ✅ Tests concept operations (coherence, similarity, selection)
6. ✅ Cleans up test files

**Expected output**:
```
============================================================
TEST SUMMARY
============================================================
Preprocessing                 ✓ PASS
Validation                    ✓ PASS
Data Loading                  ✓ PASS
Concept Operations            ✓ PASS
============================================================
✓ ALL TESTS PASSED!
```

If all tests pass, you're ready to process real datasets!

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install SONAR
pip install sonar-space

# Install other dependencies
pip install nltk tqdm numpy

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. Prepare Your Data

Organize your datasets in this structure:
```
ASearcher/
├── data/
│   ├── raw/              # Original datasets
│   │   ├── gaia_train.json
│   │   ├── gaia_val.json
│   │   └── ...
│   └── processed/        # Preprocessed datasets (will be created)
└── preprocessing/        # This directory
```

### 3. Preprocess a Single Dataset

```bash
cd preprocessing/

python preprocess_concepts.py \
    --input ../data/raw/gaia_train.json \
    --output ../data/processed/gaia_train_with_concepts.pkl \
    --device cuda:0
```

**Expected output**:
```
Loading SONAR encoder: text_sonar_basic_encoder on cuda:0
✓ SONAR loaded successfully (batch_size=32)
Loading dataset from ../data/raw/gaia_train.json
✓ Loaded 500 items

Processing 500 items...
100%|████████████████████| 500/500 [02:15<00:00,  3.68it/s]

✓ Processing complete:
  - Items: 500
  - Webpages: 5000
  - Total sentences: 120000
  - Avg sentences/page: 24.0

Saving to ../data/processed/gaia_train_with_concepts.pkl
✓ Saved successfully
  - File size: 3.20 GB
```

### 4. Validate Preprocessed Data

```bash
python validate_concepts.py \
    --dataset ../data/processed/gaia_train_with_concepts.pkl
```

**Expected output**:
```
============================================================
Validating: ../data/processed/gaia_train_with_concepts.pkl
============================================================

1. Validating Embedding Shapes
✓ All embeddings have correct shape (1024-dim)

2. Computing Coherence Statistics
Coherence Statistics:
  - Average coherence: 0.782
  - Median: 0.791
  - Range: [0.245, 0.985]

3. Validating Coherence Thresholds
✓ avg_coherence: 0.782 > 0.5 (PASS)
✓ min_coherence: 0.245 > 0.1 (PASS)

4. Checking Data Completeness
✓ All items have concept embeddings

============================================================
VALIDATION SUMMARY
============================================================
Embedding Shapes                ✓ PASS
Coherence Statistics            ✓ PASS
Coherence Thresholds            ✓ PASS
Data Completeness               ✓ PASS
============================================================
✓ ALL CHECKS PASSED!

This dataset is ready for RL training.
```

### 5. Batch Process Multiple Datasets

Edit `batch_preprocess.sh` to add your datasets:

```bash
# Edit DATASETS array in batch_preprocess.sh
DATASETS=(
    "data/raw/gaia_train.json:data/processed/gaia_train_with_concepts.pkl"
    "data/raw/gaia_val.json:data/processed/gaia_val_with_concepts.pkl"
    "data/raw/xbench.json:data/processed/xbench_with_concepts.pkl"
)
```

Then run:

```bash
bash batch_preprocess.sh cuda:0
```

## 📚 Detailed Usage

### `preprocess_concepts.py`

Main preprocessing script that encodes text into SONAR concept embeddings.

**Arguments**:
- `--input, -i`: Input dataset file (.json, .jsonl, or .pkl)
- `--output, -o`: Output path (.pkl recommended)
- `--model`: SONAR model name (default: `text_sonar_basic_encoder`)
- `--device`: Device for encoding (default: `cuda:0`)
  - `cuda:0`, `cuda:1`, etc. for GPU
  - `cpu` for CPU (slower but no GPU needed)
- `--batch-size`: Batch size for encoding (default: 32)
  - Larger = faster but more GPU memory
  - Try 64 or 128 if you have enough memory
- `--dataset-format`: Format hint (`auto`, `gaia`, `hotpotqa`, `xbench`)

**Examples**:

```bash
# Process on specific GPU
python preprocess_concepts.py \
    -i data/raw/gaia_train.json \
    -o data/processed/gaia_train_with_concepts.pkl \
    --device cuda:1

# Process on CPU (no GPU needed)
python preprocess_concepts.py \
    -i data/raw/hotpotqa.json \
    -o data/processed/hotpotqa_with_concepts.pkl \
    --device cpu

# Use larger batch size for faster processing
python preprocess_concepts.py \
    -i data/raw/xbench.json \
    -o data/processed/xbench_with_concepts.pkl \
    --batch-size 64
```

### `validate_concepts.py`

Validation script to check preprocessing quality.

**Arguments**:
- `--dataset, -d`: Path to preprocessed dataset (.pkl)

**What it checks**:
1. **Embedding Shapes**: All embeddings are 1024-dim
2. **Coherence Statistics**: Average similarity between consecutive sentences
3. **Coherence Thresholds**: Coherence > 0.5 (good quality)
4. **Data Completeness**: All items have concept embeddings

**Example**:

```bash
python validate_concepts.py -d data/processed/gaia_train_with_concepts.pkl
```

### `batch_preprocess.sh`

Batch processing script for multiple datasets.

**Usage**:

```bash
# Process on GPU 0
bash batch_preprocess.sh cuda:0

# Process on GPU 1
bash batch_preprocess.sh cuda:1

# Process on CPU
bash batch_preprocess.sh cpu
```

**Configuration**: Edit the `DATASETS` array in the script:

```bash
DATASETS=(
    "input1.json:output1.pkl"
    "input2.json:output2.pkl"
    # Add more...
)
```

## 📊 Dataset Format

### Input Format

The preprocessing script supports multiple dataset formats:

**Format 1: GAIA/xBench** (list of questions with webpages):
```json
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "webpages": [
            {
                "url": "https://example.com/france",
                "text": "France is a country in Europe. Its capital is Paris..."
            }
        ]
    }
]
```

**Format 2: HotpotQA** (with context):
```json
[
    {
        "question": "...",
        "answer": "...",
        "context": [
            ["Title 1", "Text content 1..."],
            ["Title 2", "Text content 2..."]
        ]
    }
]
```

**Format 3: With search results**:
```json
[
    {
        "question": "...",
        "search_results": [
            {
                "title": "...",
                "snippet": "Snippet text...",
                "url": "..."
            }
        ]
    }
]
```

### Output Format

After preprocessing, each webpage/result/context is augmented with:

```python
{
    # Original fields preserved
    "url": "...",
    "text": "...",

    # Added by preprocessing
    "sentences": ["Sentence 1.", "Sentence 2.", ...],  # List[str]
    "concept_embeddings": np.ndarray,  # Shape: [n_sentences, 1024]
    "num_sentences": 42,  # int
    "embedding_dim": 1024  # int
}
```

## 🔧 Integration with Training

### Loading Preprocessed Data

```python
import pickle
import numpy as np

# Load preprocessed dataset
with open('data/processed/gaia_train_with_concepts.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Access concept embeddings (no SONAR needed!)
for item in train_data:
    question = item['question']
    answer = item['answer']

    for webpage in item['webpages']:
        # Concept embeddings are pre-computed
        embeddings = webpage['concept_embeddings']  # np.ndarray [n_sentences, 1024]
        sentences = webpage['sentences']  # List[str]

        # Use for selector, segmentation, etc.
        selected_indices = selector(embeddings, question)
        # ...
```

### Example DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class ConceptDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'question': item['question'],
            'answer': item['answer'],
            'webpages': [
                {
                    'url': wp['url'],
                    'sentences': wp['sentences'],
                    'concept_embeddings': wp['concept_embeddings'],  # Pre-computed!
                }
                for wp in item.get('webpages', [])
            ]
        }

# Usage
train_dataset = ConceptDataset('data/processed/gaia_train_with_concepts.pkl')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for batch in train_loader:
    # No SONAR encoding needed during training!
    # Embeddings are already computed
    pass
```

## ⚙️ Advanced Usage

### Processing Very Large Datasets

For datasets that don't fit in memory:

```python
# Process in chunks
from preprocess_concepts import ConceptPreprocessor
import pickle

preprocessor = ConceptPreprocessor(device='cuda:0')

chunk_size = 1000
output_dir = Path('data/processed/chunks')
output_dir.mkdir(parents=True, exist_ok=True)

for chunk_idx in range(0, len(dataset), chunk_size):
    chunk = dataset[chunk_idx:chunk_idx + chunk_size]

    # Process chunk
    processed_chunk = preprocessor.process_dataset(chunk)

    # Save chunk
    chunk_path = output_dir / f'chunk_{chunk_idx:06d}.pkl'
    with open(chunk_path, 'wb') as f:
        pickle.dump(processed_chunk, f)

    print(f"Saved chunk {chunk_idx // chunk_size}")
```

### Custom Language Support

SONAR supports 200+ languages. Specify language during preprocessing:

```python
# In preprocess_concepts.py, modify encode_text():
concept_data = self.encode_text(webpage['text'], lang='zho_Hans')  # Chinese
concept_data = self.encode_text(webpage['text'], lang='spa_Latn')  # Spanish
concept_data = self.encode_text(webpage['text'], lang='fra_Latn')  # French

# Common language codes:
# - eng_Latn: English
# - zho_Hans: Chinese (Simplified)
# - spa_Latn: Spanish
# - fra_Latn: French
# - deu_Latn: German
# - jpn_Jpan: Japanese
# See SONAR docs for full list
```

## 📈 Performance Expectations

### Preprocessing Time

Approximate times on V100/A100 GPU:

| Dataset | Questions | Webpages | Sentences | Time | Output Size |
|---------|-----------|----------|-----------|------|-------------|
| GAIA train | 500 | 5,000 | 120,000 | ~1 hour | 3.2 GB |
| GAIA val | 125 | 1,250 | 30,000 | ~15 min | 0.8 GB |
| xBench | 1,000 | 10,000 | 240,000 | ~2 hours | 5.1 GB |
| HotpotQA | 5,000 | 50,000 | 1,200,000 | ~4 hours | 20 GB |

### Training Speedup

Without preprocessing (online encoding):
- Encoding overhead: **5-27 min per batch**
- Total for 1000 batches: **83-450 hours**

With preprocessing (offline):
- Encoding overhead: **0 min per batch** ✅
- Total saved: **83-450 hours**
- **ROI: 9-50× return on investment**

### GPU Memory Savings

During RL training:

| Component | Without Preprocessing | With Preprocessing |
|-----------|----------------------|-------------------|
| QwQ-32B (INT4) | 20 GB | 20 GB |
| SONAR encoder | 1.5 GB | 0 GB ✅ |
| Selector | 0.5 GB | 0.5 GB |
| **Total** | **22 GB** | **20.5 GB** |

## 🐛 Troubleshooting

### Issue: SONAR Installation Failed

```bash
# Solution 1: Use conda
conda create -n sonar python=3.10
conda activate sonar
pip install sonar-space

# Solution 2: Docker
# See docs/offline_concept_encoding_strategy.md for Docker setup
```

### Issue: CUDA Out of Memory

```bash
# Solution: Reduce batch size
python preprocess_concepts.py \
    --batch-size 16  # Default is 32

# Or use CPU (slower but no GPU needed)
python preprocess_concepts.py \
    --device cpu
```

### Issue: Validation Failed (Low Coherence)

This may indicate:
- Poor quality input text (very fragmented)
- Text in unsupported language
- Encoding errors

Check a few examples manually:
```python
import pickle
with open('data/processed/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Inspect first webpage
webpage = data[0]['webpages'][0]
print("Sentences:", webpage['sentences'][:5])
print("Embeddings shape:", webpage['concept_embeddings'].shape)
print("Coherence:", ...)  # Compute manually
```

### Issue: File Size Too Large

Concept embeddings use ~4 KB per sentence. If file size is a concern:

```python
# Option 1: Use float16 instead of float32
embeddings = embeddings.astype(np.float16)  # Halves size

# Option 2: Store chunks separately (see Advanced Usage)
```

## 📖 References

- [SONAR Paper](https://arxiv.org/abs/2308.11466)
- [SONAR GitHub](https://github.com/facebookresearch/SONAR)
- [Offline Encoding Strategy](../docs/offline_concept_encoding_strategy.md)
- [Concept Modeling Integration](../docs/concept_modeling_integration.md)

## 🤝 Contributing

When adding new dataset formats:

1. Add format detection in `process_dataset()`
2. Add test case in `validate_concepts.py`
3. Update this README with format example
4. Test on sample data before full preprocessing

## ⚖️ License

Same as ASearcher main project.
