#!/usr/bin/env python3
"""
Offline SONAR Concept Preprocessing for ASearcher

This script preprocesses datasets by encoding text into SONAR concept embeddings.
The preprocessed data can then be used for RL training without needing to load
SONAR during training, saving 10-100× time and GPU memory.

Usage:
    python preprocess_concepts.py \
        --input data/raw/gaia_train.json \
        --output data/processed/gaia_train_with_concepts.pkl \
        --device cuda:0

Author: ASearcher Team
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
except ImportError:
    print("ERROR: SONAR not installed. Install with: pip install sonar-space")
    sys.exit(1)

try:
    from nltk.tokenize import sent_tokenize
    import nltk
except ImportError:
    print("ERROR: NLTK not installed. Install with: pip install nltk")
    sys.exit(1)


class ConceptPreprocessor:
    """
    Preprocessor that encodes text into SONAR concept embeddings.
    """

    def __init__(
        self,
        model_name: str = 'text_sonar_basic_encoder',
        device: str = 'cuda:0',
        batch_size: int = 32
    ):
        """
        Initialize SONAR encoder.

        Args:
            model_name: SONAR model name (default: text_sonar_basic_encoder)
            device: Device to run encoding on (cuda:0, cuda:1, or cpu)
            batch_size: Batch size for encoding (larger = faster but more memory)
        """
        print(f"Loading SONAR encoder: {model_name} on {device}")

        try:
            self.encoder = TextToEmbeddingModelPipeline(
                encoder=model_name,
                tokenizer=model_name,
                device=device
            )
        except Exception as e:
            print(f"ERROR loading SONAR: {e}")
            print("\nTroubleshooting:")
            print("1. Install SONAR: pip install sonar-space")
            print("2. Download models: python -c 'from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline; TextToEmbeddingModelPipeline(encoder=\"text_sonar_basic_encoder\", tokenizer=\"text_sonar_basic_encoder\")'")
            sys.exit(1)

        self.batch_size = batch_size
        print(f"✓ SONAR loaded successfully (batch_size={batch_size})")

        # Download NLTK punkt tokenizer if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

    def encode_text(
        self,
        text: str,
        lang: str = 'eng_Latn'
    ) -> Dict[str, Any]:
        """
        Encode text into concept embeddings.

        Args:
            text: Input text (webpage, search result, etc.)
            lang: Language code (default: eng_Latn for English)

        Returns:
            Dictionary with:
                - sentences: List[str] - sentence segmentation
                - concept_embeddings: np.ndarray [n_sentences, 1024]
                - num_sentences: int
                - embedding_dim: int (1024)
        """
        if not text or not text.strip():
            return {
                'sentences': [],
                'concept_embeddings': np.array([]),
                'num_sentences': 0,
                'embedding_dim': 0
            }

        # Sentence segmentation
        sentences = sent_tokenize(text)

        if len(sentences) == 0:
            return {
                'sentences': [],
                'concept_embeddings': np.array([]),
                'num_sentences': 0,
                'embedding_dim': 0
            }

        # Batch encode for efficiency
        try:
            embeddings = self.encoder.predict(sentences, source_lang=lang)
            embeddings_np = embeddings.cpu().numpy()
        except Exception as e:
            print(f"WARNING: Encoding failed for text (length={len(text)}): {e}")
            # Return empty embeddings on failure
            return {
                'sentences': sentences,
                'concept_embeddings': np.array([]),
                'num_sentences': len(sentences),
                'embedding_dim': 0
            }

        return {
            'sentences': sentences,
            'concept_embeddings': embeddings_np,
            'num_sentences': len(sentences),
            'embedding_dim': embeddings_np.shape[1]
        }

    def process_webpage(self, webpage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single webpage.

        Args:
            webpage: Dict with 'text' key (and optionally 'url', 'title', etc.)

        Returns:
            Webpage dict augmented with concept embeddings
        """
        if 'text' not in webpage:
            print("WARNING: Webpage missing 'text' field, skipping")
            return webpage

        # Encode text
        concept_data = self.encode_text(webpage['text'])

        # Augment webpage with concept data
        webpage.update(concept_data)

        return webpage

    def process_search_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a search result snippet.

        Args:
            result: Dict with 'snippet' or 'text' key

        Returns:
            Result dict augmented with concept embeddings
        """
        text = result.get('snippet') or result.get('text') or ''

        if not text:
            print("WARNING: Search result missing text, skipping")
            return result

        # Encode text
        concept_data = self.encode_text(text)

        # Augment result
        result.update(concept_data)

        return result

    def process_dataset(
        self,
        dataset: List[Dict[str, Any]],
        dataset_format: str = 'auto'
    ) -> List[Dict[str, Any]]:
        """
        Process entire dataset.

        Args:
            dataset: List of question/episode data
            dataset_format: Format hint ('gaia', 'hotpotqa', 'auto')

        Returns:
            Processed dataset with concept embeddings
        """
        processed_data = []
        total_sentences = 0
        total_webpages = 0
        total_search_results = 0

        print(f"\nProcessing {len(dataset)} items...")

        for item in tqdm(dataset, desc="Processing"):
            # Process webpages
            if 'webpages' in item:
                for webpage in item['webpages']:
                    self.process_webpage(webpage)
                    total_webpages += 1
                    if webpage.get('num_sentences'):
                        total_sentences += webpage['num_sentences']

            # Process search results
            if 'search_results' in item:
                for result in item['search_results']:
                    self.process_search_result(result)
                    total_search_results += 1
                    if result.get('num_sentences'):
                        total_sentences += result['num_sentences']

            # Process contexts (for HotpotQA format)
            if 'context' in item and isinstance(item['context'], list):
                for context_item in item['context']:
                    if isinstance(context_item, list) and len(context_item) == 2:
                        title, text = context_item
                        # Treat as webpage
                        webpage = {'text': text, 'title': title}
                        self.process_webpage(webpage)
                        context_item.append(webpage)  # Add processed data
                        total_webpages += 1
                        if webpage.get('num_sentences'):
                            total_sentences += webpage['num_sentences']

            processed_data.append(item)

        print(f"\n✓ Processing complete:")
        print(f"  - Items: {len(processed_data)}")
        print(f"  - Webpages: {total_webpages}")
        print(f"  - Search results: {total_search_results}")
        print(f"  - Total sentences: {total_sentences}")
        print(f"  - Avg sentences/page: {total_sentences / max(total_webpages, 1):.1f}")

        return processed_data


def load_dataset(input_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from JSON or pickle file."""
    print(f"Loading dataset from {input_path}")

    if input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            data = json.load(f)
    elif input_path.suffix == '.jsonl':
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif input_path.suffix == '.pkl':
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Handle different dataset structures
    if isinstance(data, dict):
        # Might be wrapped in a dict like {'data': [...]}
        if 'data' in data:
            data = data['data']
        elif 'questions' in data:
            data = data['questions']
        else:
            # Convert to list
            data = [data]

    print(f"✓ Loaded {len(data)} items")
    return data


def save_dataset(data: List[Dict[str, Any]], output_path: Path):
    """Save processed dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}")

    if output_path.suffix == '.pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif output_path.suffix == '.json':
        # Note: numpy arrays need special handling for JSON
        print("WARNING: Saving as JSON will lose numpy arrays. Use .pkl instead.")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

    file_size_gb = output_path.stat().st_size / 1e9
    print(f"✓ Saved successfully")
    print(f"  - File size: {file_size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess datasets with SONAR concept embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process GAIA training set
  python preprocess_concepts.py \\
      --input data/raw/gaia_train.json \\
      --output data/processed/gaia_train_with_concepts.pkl

  # Process with specific device
  python preprocess_concepts.py \\
      --input data/raw/xbench.json \\
      --output data/processed/xbench_with_concepts.pkl \\
      --device cuda:1

  # Process on CPU (slower but no GPU needed)
  python preprocess_concepts.py \\
      --input data/raw/hotpotqa.json \\
      --output data/processed/hotpotqa_with_concepts.pkl \\
      --device cpu
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input dataset file (.json, .jsonl, or .pkl)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for processed dataset (.pkl recommended)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='text_sonar_basic_encoder',
        help='SONAR model name (default: text_sonar_basic_encoder)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for encoding (cuda:0, cuda:1, or cpu)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )
    parser.add_argument(
        '--dataset-format',
        type=str,
        default='auto',
        choices=['auto', 'gaia', 'hotpotqa', 'xbench'],
        help='Dataset format hint (default: auto)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(input_path)

    # Initialize preprocessor
    preprocessor = ConceptPreprocessor(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )

    # Process
    processed_dataset = preprocessor.process_dataset(
        dataset,
        dataset_format=args.dataset_format
    )

    # Save
    save_dataset(processed_dataset, output_path)

    print(f"\n{'='*60}")
    print("✓ Preprocessing complete!")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"\nYou can now use this preprocessed data for RL training")
    print(f"without needing to load SONAR, saving 10-100× time!")


if __name__ == '__main__':
    main()
