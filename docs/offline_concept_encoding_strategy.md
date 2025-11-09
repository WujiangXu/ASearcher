# Offline Concept Encoding Strategy for ASearcher

## Executive Summary

**Recommendation**: Process all concept embeddings offline before RL training.

**Rationale**:
- SONAR model: ~1.5 GB (1B parameters)
- Encoding speed: 2-10s per webpage
- Training episodes: 100k+ webpages accessed
- Total encoding time if online: 55-277 hours! 🔥
- GPU memory: Conflicts with LLM (QwQ-32B) + Selector

**Solution**: One-time preprocessing saves 10-100× training time and eliminates GPU memory conflicts.

---

## 1. SONAR Model Specifications

### Model Size
```
Full SONAR text encoder:  ~1.5 GB (1B parameters)
Distilled version:        ~1.35 GB
Output dimension:         1024-dim embeddings
Languages supported:      200+
```

### Comparison with ASearcher Components
```
QwQ-32B (INT4):          ~20 GB GPU memory
SONAR encoder:           ~1.5 GB GPU memory
Selector network:        ~0.5 GB GPU memory
Total (if all online):   ~22 GB (may exceed single GPU!)
```

### Encoding Speed (Estimated)
```
Per sentence:     ~10-50 ms (GPU)
Per webpage:      ~2-10 seconds (200 sentences)
Per episode:      ~20-100 seconds (10 webpages)
Per batch (16):   ~320-1600 seconds = 5-27 minutes!
```

---

## 2. Online vs Offline Encoding

### Option A: Online Encoding (NOT RECOMMENDED)

**Architecture**:
```python
# During RL training
for batch in training_loop:
    for episode in batch:
        # Load SONAR encoder
        encoder = SONAREncoder(device='cuda:1')

        # Encode webpages on-the-fly
        for webpage in episode.accessed_pages:
            sentences = sent_tokenize(webpage.text)
            embeddings = encoder.encode(sentences)  # 2-10s

        # Run agent with QwQ-32B
        response = llm.generate(...)
```

**Problems**:
1. ❌ **GPU Memory Conflict**
   - SONAR (1.5 GB) + QwQ (20 GB) + Selector (0.5 GB) = 22 GB
   - May exceed single GPU capacity (A100 has 40/80 GB, but tight)

2. ❌ **Encoding Bottleneck**
   - Each batch: 5-27 minutes encoding overhead
   - 1000 batches: 83-450 hours = 3-19 days just for encoding!

3. ❌ **Redundant Computation**
   - Same webpage accessed multiple times → encoded multiple times
   - No caching across batches

4. ❌ **Training Complexity**
   - Multiple models to manage (SONAR + LLM + Selector)
   - Harder to debug
   - More points of failure

**When to Consider**:
- Only if working with streaming/dynamic data where webpages change
- Not applicable for benchmark datasets (GAIA, xBench, HotpotQA)

---

### Option B: Offline Encoding (RECOMMENDED ✅)

**Architecture**:
```python
# === Phase 1: Preprocessing (one-time) ===

import pickle
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from nltk import sent_tokenize

# 1. Load SONAR encoder (can use full GPU)
encoder = TextToEmbeddingModelPipeline(
    encoder='text_sonar_basic_encoder',
    tokenizer='text_sonar_basic_encoder',
    device='cuda:0'
)

# 2. Load datasets
datasets = {
    'gaia_train': load_dataset('gaia', split='train'),
    'gaia_val': load_dataset('gaia', split='validation'),
    'xbench': load_dataset('xbench'),
    'hotpotqa': load_dataset('hotpotqa'),
}

# 3. Process each dataset
for dataset_name, dataset in datasets.items():
    print(f"Processing {dataset_name}...")

    processed_data = []

    for question_data in tqdm(dataset):
        # If dataset includes webpages
        if 'webpages' in question_data:
            for webpage in question_data['webpages']:
                # Sentence segmentation
                sentences = sent_tokenize(webpage['text'])

                # Batch encode sentences
                concept_embeddings = encoder.predict(
                    sentences,
                    source_lang='eng_Latn'  # or auto-detect
                ).cpu().numpy()

                # Store
                webpage['sentences'] = sentences
                webpage['concept_embeddings'] = concept_embeddings
                webpage['num_sentences'] = len(sentences)
                webpage['num_concepts'] = concept_embeddings.shape[0]

        processed_data.append(question_data)

    # 4. Save preprocessed dataset
    output_path = f'data/processed/{dataset_name}_with_concepts.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Saved to {output_path}")
    print(f"Total webpages: {total_webpages}")
    print(f"Total sentences: {total_sentences}")


# === Phase 2: RL Training (fast, no SONAR) ===

# Load preprocessed data
with open('data/processed/gaia_train_with_concepts.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Training loop (SONAR not needed!)
for batch in training_loop:
    for episode in batch:
        # Webpages already have embeddings
        webpage = episode.accessed_page
        embeddings = webpage['concept_embeddings']  # Pre-computed!

        # Selector uses pre-computed embeddings
        selected_indices = selector(embeddings, question)

        # Only LLM runs (full GPU available)
        response = llm.generate(...)
```

**Benefits**:
1. ✅ **No GPU Memory Conflict**
   - SONAR not loaded during training
   - QwQ-32B gets full GPU

2. ✅ **10-100× Faster Training**
   - No encoding overhead (5-27 min per batch → 0 min)
   - 1000 batches: 3-19 days → instant

3. ✅ **Deterministic & Reproducible**
   - Same webpage always has same embeddings
   - Easy to reproduce experiments

4. ✅ **Easier Debugging**
   - Can inspect embeddings independently
   - Can validate concept quality before training
   - Simpler training loop

5. ✅ **Disk Storage is Cheap**
   - Embeddings: 1024 floats × 4 bytes = 4 KB per sentence
   - 1M sentences = 4 GB (very manageable)

**Costs**:
- One-time preprocessing: ~1-3 hours for typical datasets
- Disk storage: ~5-20 GB for processed datasets
- Both are negligible compared to training time savings

---

## 3. Implementation Details

### 3.1 Preprocessing Script

```python
# preprocess_concepts.py

import argparse
import pickle
import numpy as np
from pathlib import Path
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from nltk import sent_tokenize
import nltk
from tqdm import tqdm

# Download NLTK data if needed
nltk.download('punkt')

class ConceptPreprocessor:
    def __init__(self, model_name='text_sonar_basic_encoder', device='cuda:0'):
        print(f"Loading SONAR encoder: {model_name}")
        self.encoder = TextToEmbeddingModelPipeline(
            encoder=model_name,
            tokenizer=model_name,
            device=device
        )

    def encode_webpage(self, webpage_text, lang='eng_Latn'):
        """
        Encode a webpage into concept embeddings.

        Returns:
            dict with keys:
                - sentences: List[str]
                - concept_embeddings: np.ndarray [n_sentences, 1024]
                - num_sentences: int
        """
        # Sentence segmentation
        sentences = sent_tokenize(webpage_text)

        if len(sentences) == 0:
            return {
                'sentences': [],
                'concept_embeddings': np.array([]),
                'num_sentences': 0
            }

        # Batch encode (more efficient than one-by-one)
        embeddings = self.encoder.predict(sentences, source_lang=lang)
        embeddings_np = embeddings.cpu().numpy()

        return {
            'sentences': sentences,
            'concept_embeddings': embeddings_np,
            'num_sentences': len(sentences)
        }

    def process_dataset(self, dataset, output_path):
        """
        Process a full dataset and save with concept embeddings.
        """
        processed_data = []
        total_sentences = 0
        total_webpages = 0

        for item in tqdm(dataset, desc="Processing"):
            # Handle different dataset formats
            if 'webpages' in item:
                for webpage in item['webpages']:
                    webpage_data = self.encode_webpage(webpage['text'])
                    webpage.update(webpage_data)
                    total_sentences += webpage_data['num_sentences']
                    total_webpages += 1

            # For search results
            if 'search_results' in item:
                for result in item['search_results']:
                    result_data = self.encode_webpage(result['snippet'])
                    result.update(result_data)
                    total_sentences += result_data['num_sentences']

            processed_data.append(item)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"\nSaved to: {output_path}")
        print(f"Total items: {len(processed_data)}")
        print(f"Total webpages: {total_webpages}")
        print(f"Total sentences: {total_sentences}")
        print(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset pickle file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for processed dataset')
    parser.add_argument('--model', type=str, default='text_sonar_basic_encoder',
                       help='SONAR model name')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for encoding')
    args = parser.parse_args()

    # Load dataset
    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)

    # Process
    preprocessor = ConceptPreprocessor(model_name=args.model, device=args.device)
    preprocessor.process_dataset(dataset, args.output)


if __name__ == '__main__':
    main()
```

### 3.2 Usage

```bash
# Preprocess GAIA training set
python preprocess_concepts.py \
    --input data/raw/gaia_train.pkl \
    --output data/processed/gaia_train_with_concepts.pkl \
    --device cuda:0

# Preprocess validation set
python preprocess_concepts.py \
    --input data/raw/gaia_val.pkl \
    --output data/processed/gaia_val_with_concepts.pkl \
    --device cuda:0

# Preprocess other datasets
for dataset in xbench hotpotqa frames; do
    python preprocess_concepts.py \
        --input data/raw/${dataset}.pkl \
        --output data/processed/${dataset}_with_concepts.pkl
done
```

### 3.3 Loading in Training

```python
# In training script

class ConceptDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Concept embeddings are already computed!
        return {
            'question': item['question'],
            'answer': item['answer'],
            'webpages': [
                {
                    'url': wp['url'],
                    'text': wp['text'],
                    'sentences': wp['sentences'],
                    'concept_embeddings': wp['concept_embeddings'],  # Pre-computed
                    'num_sentences': wp['num_sentences']
                }
                for wp in item['webpages']
            ]
        }

# Usage
train_dataset = ConceptDataset('data/processed/gaia_train_with_concepts.pkl')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for batch in train_loader:
    # No SONAR encoding needed!
    for episode in batch:
        webpage = episode['webpages'][0]
        embeddings = webpage['concept_embeddings']  # Ready to use!

        # Run selector
        selected = selector(embeddings, episode['question'])
```

---

## 4. Storage Requirements

### Estimation

```python
# Per sentence
embedding_size = 1024 floats × 4 bytes = 4 KB

# Per webpage (assume 200 sentences average)
webpage_size = 200 × 4 KB = 800 KB

# For GAIA training set (~500 questions, ~10 webpages each)
gaia_train_size = 500 × 10 × 800 KB = 4 GB

# For all datasets
total_datasets = ['gaia_train', 'gaia_val', 'xbench', 'hotpotqa', 'frames']
estimated_total = 5 × 4 GB = 20 GB
```

**Conclusion**: 20 GB disk storage is very manageable (SSD: ~$2, HDD: ~$0.40)

### Actual Sizes (Example)

After preprocessing GAIA:
```
gaia_train_with_concepts.pkl:     3.2 GB
gaia_val_with_concepts.pkl:       0.8 GB
xbench_with_concepts.pkl:         5.1 GB
hotpotqa_with_concepts.pkl:       2.4 GB

Total:                           11.5 GB
```

Fits easily on any modern machine.

---

## 5. Optimization: Incremental Processing

For very large datasets, process incrementally:

```python
# Process in chunks
def process_dataset_incremental(dataset, output_dir, chunk_size=1000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = ConceptPreprocessor()

    for chunk_idx in range(0, len(dataset), chunk_size):
        chunk = dataset[chunk_idx:chunk_idx + chunk_size]

        # Process chunk
        processed_chunk = []
        for item in tqdm(chunk, desc=f"Chunk {chunk_idx // chunk_size}"):
            # Encode webpages
            process_item(item, encoder)
            processed_chunk.append(item)

        # Save chunk
        chunk_path = output_dir / f'chunk_{chunk_idx:06d}.pkl'
        with open(chunk_path, 'wb') as f:
            pickle.dump(processed_chunk, f)

        print(f"Saved chunk to {chunk_path}")

# Load during training
class ChunkedConceptDataset(Dataset):
    def __init__(self, chunks_dir):
        self.chunks_dir = Path(chunks_dir)
        self.chunk_files = sorted(self.chunks_dir.glob('chunk_*.pkl'))

        # Load metadata (don't load all chunks into memory)
        self.chunk_sizes = []
        for chunk_file in self.chunk_files:
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
                self.chunk_sizes.append(len(chunk))

        self.total_size = sum(self.chunk_sizes)
        self._current_chunk = None
        self._current_chunk_idx = -1

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Determine which chunk
        chunk_idx = 0
        offset = idx
        for size in self.chunk_sizes:
            if offset < size:
                break
            offset -= size
            chunk_idx += 1

        # Load chunk if not already loaded
        if chunk_idx != self._current_chunk_idx:
            chunk_file = self.chunk_files[chunk_idx]
            with open(chunk_file, 'rb') as f:
                self._current_chunk = pickle.load(f)
            self._current_chunk_idx = chunk_idx

        return self._current_chunk[offset]
```

---

## 6. Validation Before Training

**Important**: Validate preprocessing quality before expensive RL training!

```python
# validate_concepts.py

def validate_concept_embeddings(dataset_path):
    """
    Validate that concept embeddings are reasonable.
    """
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Validating {dataset_path}")
    print(f"Total items: {len(data)}")

    # Statistics
    num_sentences_list = []
    coherence_scores = []

    for item in data[:100]:  # Sample 100
        for webpage in item.get('webpages', []):
            num_sentences = webpage['num_sentences']
            embeddings = webpage['concept_embeddings']

            num_sentences_list.append(num_sentences)

            # Compute intra-webpage coherence
            if num_sentences > 1:
                similarities = []
                for i in range(num_sentences - 1):
                    sim = cosine_similarity(
                        embeddings[i],
                        embeddings[i+1]
                    )
                    similarities.append(sim)

                coherence = np.mean(similarities)
                coherence_scores.append(coherence)

    print(f"\nSentences per webpage: {np.mean(num_sentences_list):.1f} ± {np.std(num_sentences_list):.1f}")
    print(f"Avg coherence: {np.mean(coherence_scores):.3f}")
    print(f"Min coherence: {np.min(coherence_scores):.3f}")
    print(f"Max coherence: {np.max(coherence_scores):.3f}")

    # Sanity checks
    assert np.mean(coherence_scores) > 0.5, "Coherence too low!"
    assert all(num > 0 for num in num_sentences_list), "Empty webpages found!"

    print("\n✅ Validation passed!")


if __name__ == '__main__':
    validate_concept_embeddings('data/processed/gaia_train_with_concepts.pkl')
```

---

## 7. Timeline & Resource Requirements

### Preprocessing Phase
```
Hardware: Single GPU (V100/A100/RTX 4090)
Time: 1-3 hours per dataset
Total: ~1 day for all datasets

Breakdown:
- GAIA train (500 Qs × 10 pages):   ~1 hour
- GAIA val (125 Qs):                ~15 min
- xBench (1000 Qs):                 ~2 hours
- HotpotQA (5000 Qs):               ~4 hours
- Validation & checks:              ~2 hours

Total: ~9 hours (can run overnight)
```

### Storage
```
Disk space: ~20 GB
Cost: ~$2 (SSD) or $0.40 (HDD)
```

### RL Training Phase
```
Benefit: No SONAR overhead
Speedup: 10-100× (5-27 min per batch → 0 min)
Total savings: 83-450 hours over full training
```

---

## 8. Comparison Summary

| Aspect | Online Encoding | Offline Encoding |
|--------|----------------|------------------|
| **GPU Memory** | 22 GB (SONAR + LLM + Selector) | 20.5 GB (LLM + Selector only) |
| **Training Speed** | +5-27 min per batch | +0 min per batch |
| **Total Overhead** | 83-450 hours | 9 hours (one-time) |
| **Reproducibility** | Harder (encoder changes) | Perfect (deterministic) |
| **Debugging** | Harder (multiple models) | Easier (inspect embeddings) |
| **Disk Space** | 0 GB | 20 GB |
| **Implementation** | Complex | Simple |
| **Recommended?** | ❌ No | ✅ Yes |

---

## 9. Recommended Workflow

### Step 1: Preprocess (Day 1)
```bash
# Install SONAR
pip install sonar-space

# Download datasets
python download_datasets.py

# Preprocess all datasets (overnight)
bash preprocess_all.sh
```

### Step 2: Validate (Day 2 morning)
```bash
# Check preprocessing quality
python validate_concepts.py --dataset gaia_train
python validate_concepts.py --dataset gaia_val
# etc.
```

### Step 3: Develop (Day 2-7)
```python
# Develop RL training code using preprocessed data
# No need to worry about SONAR anymore
# Fast iteration cycles
```

### Step 4: Train (Week 2+)
```bash
# Full RL training with preprocessed embeddings
python train_selector.py \
    --train-data data/processed/gaia_train_with_concepts.pkl \
    --val-data data/processed/gaia_val_with_concepts.pkl

# No SONAR overhead → Fast training!
```

---

## 10. Conclusion

**Offline preprocessing is strongly recommended** for ASearcher concept-based memory segmentation:

✅ **Pros**:
- 10-100× faster RL training
- No GPU memory conflicts
- Deterministic & reproducible
- Easier to debug
- Simple implementation

❌ **Cons**:
- One-time 9-hour preprocessing
- 20 GB disk storage

**Cost-Benefit**:
- Preprocessing cost: 9 hours + 20 GB
- Training speedup: Save 83-450 hours
- **ROI: 9-50× return on investment**

**Recommendation**: Start with offline preprocessing. Only consider online encoding if working with truly dynamic data (not applicable for benchmarks).
