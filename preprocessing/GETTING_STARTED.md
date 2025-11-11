# Getting Started with Offline Concept Preprocessing

Quick guide to get you up and running with SONAR concept preprocessing for ASearcher.

## ⚡ TL;DR (5-minute setup)

```bash
# 1. Install dependencies
pip install sonar-space nltk tqdm numpy
python -c "import nltk; nltk.download('punkt')"

# 2. Run quick test
cd preprocessing/
bash quick_test.sh cpu

# 3. If test passes, run full test
python test_preprocessing.py --device cpu

# 4. If all tests pass, process your data!
python preprocess_concepts.py \
    --input ../data/raw/your_data.json \
    --output ../data/processed/your_data_with_concepts.pkl
```

---

## 📝 Step-by-Step Guide

### Step 1: Install Dependencies (2 minutes)

```bash
# Install SONAR (Meta's sentence encoder)
pip install sonar-space

# Install supporting libraries
pip install nltk tqdm numpy

# Download NLTK sentence tokenizer
python -c "import nltk; nltk.download('punkt')"
```

**Verify installation**:
```bash
python -c "from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline; print('✓ SONAR installed')"
```

### Step 2: Run Quick Test (30 seconds)

```bash
cd preprocessing/

# Test on CPU (no GPU needed for testing)
bash quick_test.sh cpu
```

**Expected output**:
```
========================================
Quick Smoke Test
========================================
✓ Python found: Python 3.10.x
✓ SONAR installed
✓ NLTK installed
✓ NumPy installed
✓ NLTK punkt tokenizer found
✓ SONAR encoder loaded successfully
✓ Encoding successful, shape: (2, 1024)
✓ Embeddings have correct dimension (1024)
========================================
✓ ALL CHECKS PASSED!
========================================
```

**If it fails**:
- SONAR not installed → `pip install sonar-space`
- NLTK not installed → `pip install nltk`
- GPU error → Use `cpu` instead of `cuda:0`

### Step 3: Run Full Integration Test (2-3 minutes)

```bash
# Test complete preprocessing workflow
python test_preprocessing.py --device cpu
```

**What it does**:
1. Creates 3 sample questions with webpages
2. Runs SONAR encoding
3. Validates embedding quality
4. Tests loading and using processed data
5. Tests concept operations (coherence, similarity)
6. Cleans up test files

**Expected output**:
```
============================================================
Creating Sample Test Data
============================================================
✓ Created 3 sample questions
  1. What is the capital of France?
     - Webpages: 2
  2. Who wrote Pride and Prejudice?
     - Webpages: 2
  3. What is the speed of light?
     - Webpages: 1

============================================================
Running Preprocessing
============================================================
Loading SONAR encoder: text_sonar_basic_encoder on cpu
✓ SONAR loaded successfully (batch_size=8)

Processing 3 items...
100%|████████████████████| 3/3 [00:15<00:00,  5.12s/it]

✓ Processing complete:
  - Items: 3
  - Webpages: 5
  - Total sentences: 42
  - Avg sentences/page: 8.4

============================================================
Running Validation
============================================================
1. Validating Embedding Shapes
✓ All embeddings have correct shape (1024-dim)

2. Computing Coherence Statistics
Avg coherence: 0.782

3. Validating Coherence Thresholds
✓ avg_coherence: 0.782 > 0.5 (PASS)
✓ min_coherence: 0.245 > 0.1 (PASS)

4. Checking Data Completeness
✓ All items have concept embeddings

============================================================
Testing Data Loading
============================================================
✓ Loaded 3 items
✓ All structure checks passed

============================================================
Testing Concept Operations
============================================================
Test 1: Computing coherence scores
  - Webpage: France - Wikipedia
    Coherence: 0.812
    Sentences: 8

Test 2: Computing query-concept similarities
  Question: What is the capital of France?
    Webpage: France - Wikipedia
    Top-3 similarities: ['0.923', '0.891', '0.856']

Test 3: Mock segment selection (top-k by similarity)
  Selected top-3 sentences (by mock similarity):
    [2] (sim=0.923): The capital and largest city is Paris...
    [3] (sim=0.891): Paris is known for the Eiffel Tower...
    [4] (sim=0.856): France has a population of about 67 million...

✓ All concept operations completed successfully

============================================================
TEST SUMMARY
============================================================
Preprocessing                 ✓ PASS
Validation                    ✓ PASS
Data Loading                  ✓ PASS
Concept Operations            ✓ PASS
============================================================
✓ ALL TESTS PASSED!

The preprocessing pipeline is working correctly.
You can now use it on real datasets.
```

**If it fails**:
- Check error messages for specific issues
- Try `--device cpu` if GPU fails
- Check SONAR installation: `pip install --upgrade sonar-space`
- Report issue with full error log

### Step 4: Prepare Your Dataset

Organize your data:
```
ASearcher/
├── data/
│   ├── raw/              # Put your original datasets here
│   │   ├── gaia_train.json
│   │   ├── gaia_val.json
│   │   └── your_data.json
│   └── processed/        # Preprocessed data will go here (auto-created)
└── preprocessing/
```

Your dataset should be in one of these formats:

**Format 1: GAIA/xBench style** (recommended):
```json
[
  {
    "question": "What is X?",
    "answer": "Y",
    "webpages": [
      {
        "url": "https://...",
        "text": "Long text content..."
      }
    ]
  }
]
```

**Format 2: HotpotQA style**:
```json
[
  {
    "question": "...",
    "answer": "...",
    "context": [
      ["Title 1", "Text 1"],
      ["Title 2", "Text 2"]
    ]
  }
]
```

### Step 5: Preprocess Your Dataset

```bash
cd preprocessing/

# Single dataset
python preprocess_concepts.py \
    --input ../data/raw/gaia_train.json \
    --output ../data/processed/gaia_train_with_concepts.pkl \
    --device cuda:0

# Batch process multiple datasets
# (Edit batch_preprocess.sh to add your datasets)
bash batch_preprocess.sh cuda:0
```

**Expected time**:
- Small dataset (100 questions): ~10 minutes
- GAIA train (500 questions): ~1 hour
- Large dataset (5000 questions): ~4 hours

**Output**:
```
Loading SONAR encoder: text_sonar_basic_encoder on cuda:0
✓ SONAR loaded successfully (batch_size=32)
Loading dataset from ../data/raw/gaia_train.json
✓ Loaded 500 items

Processing 500 items...
100%|████████████████████| 500/500 [58:23<00:00,  7.0s/it]

✓ Processing complete:
  - Items: 500
  - Webpages: 5000
  - Total sentences: 120000
  - Avg sentences/page: 24.0

Saving to ../data/processed/gaia_train_with_concepts.pkl
✓ Saved successfully
  - File size: 3.20 GB
```

### Step 6: Validate Processed Data

```bash
python validate_concepts.py \
    --dataset ../data/processed/gaia_train_with_concepts.pkl
```

**Expected output**:
```
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

### Step 7: Use in Training

```python
import pickle

# Load preprocessed data
with open('data/processed/gaia_train_with_concepts.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Use in training (no SONAR needed!)
for episode in train_data:
    question = episode['question']
    answer = episode['answer']

    for webpage in episode['webpages']:
        # Concept embeddings are pre-computed!
        embeddings = webpage['concept_embeddings']  # Shape: [n_sentences, 1024]
        sentences = webpage['sentences']  # List[str]

        # Use for your selector/segmentation
        selected_indices = your_selector(embeddings, question)
        # ... rest of your training code
```

**Benefits**:
- ✅ No SONAR loading during training
- ✅ 10-100× faster (no encoding overhead)
- ✅ 1.5 GB less GPU memory
- ✅ Deterministic and reproducible

---

## 🐛 Troubleshooting

### Issue: SONAR installation fails

**Solution 1**: Use conda environment
```bash
conda create -n sonar python=3.10
conda activate sonar
pip install sonar-space
```

**Solution 2**: Check Python version
```bash
python --version  # Should be 3.8+
```

**Solution 3**: Update pip
```bash
pip install --upgrade pip
pip install sonar-space
```

### Issue: CUDA out of memory during preprocessing

**Solution 1**: Reduce batch size
```bash
python preprocess_concepts.py \
    --batch-size 16  # Default is 32
```

**Solution 2**: Use CPU (slower but works)
```bash
python preprocess_concepts.py \
    --device cpu
```

### Issue: Test fails with "NLTK punkt not found"

**Solution**:
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Issue: Validation fails with low coherence

This may indicate:
- Poor quality input text (very fragmented)
- Wrong language (SONAR supports 200+ languages but English works best)

**Check manually**:
```python
import pickle
with open('data/processed/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Inspect first webpage
webpage = data[0]['webpages'][0]
print("Sentences:", webpage['sentences'][:5])
print("Num sentences:", webpage['num_sentences'])
print("Embeddings shape:", webpage['concept_embeddings'].shape)
```

---

## ❓ FAQ

### Q: How long does preprocessing take?

A: Depends on dataset size:
- 100 questions: ~10 minutes
- 500 questions (GAIA train): ~1 hour
- 5000 questions (HotpotQA): ~4 hours

Run overnight for large datasets.

### Q: How much disk space do I need?

A: ~4 KB per sentence:
- GAIA train (120k sentences): ~3.2 GB
- All typical datasets: ~11-20 GB total

Very manageable on any modern machine.

### Q: Can I process datasets in other languages?

A: Yes! SONAR supports 200+ languages. Specify language:
```python
# In preprocess_concepts.py, modify encode_text():
concept_data = self.encode_text(text, lang='zho_Hans')  # Chinese
concept_data = self.encode_text(text, lang='spa_Latn')  # Spanish
```

### Q: Do I need to preprocess again if I update the dataset?

A: Only for new/changed items. You can:
1. Process only new items
2. Merge with existing processed data
3. Or reprocess everything (recommended for simplicity)

### Q: Can I use this for other projects?

A: Yes! This preprocessing approach works for any task that needs:
- Sentence-level semantic embeddings
- Concept-based text representation
- Efficient repeated encoding

---

## 📚 Next Steps

After preprocessing completes successfully:

1. **Develop RL Selector** (`concept_selector.py`)
   - Uses preprocessed embeddings
   - Trains on task success
   - No SONAR needed

2. **Develop Segmenter** (`concept_segmenter.py`)
   - Predicts boundaries
   - Uses concept gradients
   - Trains with boundary quality rewards

3. **Integration with ASearcher**
   - Modify `AgentMemory` to use preprocessed data
   - Train with GRPO
   - Evaluate on benchmarks

See main documentation for detailed implementation guides.

---

## 🤝 Getting Help

If you encounter issues:

1. Run `bash quick_test.sh cpu` to verify environment
2. Run `python test_preprocessing.py --device cpu` to test pipeline
3. Check error messages and consult Troubleshooting section above
4. Review full documentation in `README.md`
5. Open an issue with:
   - Error message
   - Test output
   - System info (OS, Python version, GPU)

---

## ✅ Success Checklist

Before processing production datasets, verify:

- [ ] `quick_test.sh` passes
- [ ] `test_preprocessing.py` passes
- [ ] Can load and inspect processed test data
- [ ] Understand dataset format requirements
- [ ] Have enough disk space (~20 GB)
- [ ] Know expected processing time

If all checked, you're ready to go! 🚀
