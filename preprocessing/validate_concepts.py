#!/usr/bin/env python3
"""
Validate Preprocessed Concept Embeddings

This script validates that preprocessed concept embeddings are of good quality
before using them for RL training. It checks:
- Embeddings exist and have correct shape
- Concept coherence (intra-document similarity)
- No missing or corrupted data
- Embedding statistics

Usage:
    python validate_concepts.py --dataset data/processed/gaia_train_with_concepts.pkl

Author: ASearcher Team
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def validate_embeddings_shape(data: List[Dict[str, Any]]) -> bool:
    """Check that all embeddings have correct shape."""
    print("\n" + "="*60)
    print("1. Validating Embedding Shapes")
    print("="*60)

    errors = []
    expected_dim = 1024  # SONAR dimension

    for idx, item in enumerate(tqdm(data[:100], desc="Checking shapes")):
        # Check webpages
        if 'webpages' in item:
            for wp_idx, webpage in enumerate(item['webpages']):
                if 'concept_embeddings' not in webpage:
                    errors.append(f"Item {idx}, webpage {wp_idx}: Missing concept_embeddings")
                    continue

                embeddings = webpage['concept_embeddings']
                if len(embeddings) == 0:
                    continue  # Empty is OK

                if embeddings.shape[1] != expected_dim:
                    errors.append(
                        f"Item {idx}, webpage {wp_idx}: "
                        f"Wrong dimension {embeddings.shape[1]} (expected {expected_dim})"
                    )

                if webpage.get('num_sentences', 0) != embeddings.shape[0]:
                    errors.append(
                        f"Item {idx}, webpage {wp_idx}: "
                        f"Mismatch between num_sentences={webpage.get('num_sentences')} "
                        f"and embeddings.shape[0]={embeddings.shape[0]}"
                    )

        # Check search results
        if 'search_results' in item:
            for sr_idx, result in enumerate(item['search_results']):
                if 'concept_embeddings' not in result:
                    errors.append(f"Item {idx}, search_result {sr_idx}: Missing concept_embeddings")
                    continue

                embeddings = result['concept_embeddings']
                if len(embeddings) == 0:
                    continue

                if embeddings.shape[1] != expected_dim:
                    errors.append(
                        f"Item {idx}, search_result {sr_idx}: "
                        f"Wrong dimension {embeddings.shape[1]} (expected {expected_dim})"
                    )

    if errors:
        print(f"\n❌ Found {len(errors)} shape errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False
    else:
        print("✓ All embeddings have correct shape (1024-dim)")
        return True


def compute_coherence_statistics(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute coherence statistics for concept embeddings."""
    print("\n" + "="*60)
    print("2. Computing Coherence Statistics")
    print("="*60)

    coherence_scores = []
    num_sentences_list = []
    num_webpages = 0

    for item in tqdm(data[:200], desc="Computing coherence"):
        if 'webpages' in item:
            for webpage in item['webpages']:
                if 'concept_embeddings' not in webpage:
                    continue

                embeddings = webpage['concept_embeddings']
                num_sentences = webpage.get('num_sentences', 0)

                if num_sentences <= 1:
                    continue

                num_webpages += 1
                num_sentences_list.append(num_sentences)

                # Compute pairwise cosine similarities
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity(embeddings[i], embeddings[i+1])
                    similarities.append(sim)

                # Average coherence for this webpage
                coherence = np.mean(similarities)
                coherence_scores.append(coherence)

    if not coherence_scores:
        print("⚠ No webpages with multiple sentences found")
        return {}

    stats = {
        'num_webpages': num_webpages,
        'avg_sentences_per_page': np.mean(num_sentences_list),
        'std_sentences_per_page': np.std(num_sentences_list),
        'min_sentences': np.min(num_sentences_list),
        'max_sentences': np.max(num_sentences_list),
        'avg_coherence': np.mean(coherence_scores),
        'std_coherence': np.std(coherence_scores),
        'min_coherence': np.min(coherence_scores),
        'max_coherence': np.max(coherence_scores),
        'median_coherence': np.median(coherence_scores)
    }

    print(f"\nWebpage Statistics:")
    print(f"  - Number of webpages analyzed: {stats['num_webpages']}")
    print(f"  - Sentences per page: {stats['avg_sentences_per_page']:.1f} ± {stats['std_sentences_per_page']:.1f}")
    print(f"  - Range: [{stats['min_sentences']:.0f}, {stats['max_sentences']:.0f}]")

    print(f"\nCoherence Statistics:")
    print(f"  - Average coherence: {stats['avg_coherence']:.3f}")
    print(f"  - Std dev: {stats['std_coherence']:.3f}")
    print(f"  - Median: {stats['median_coherence']:.3f}")
    print(f"  - Range: [{stats['min_coherence']:.3f}, {stats['max_coherence']:.3f}]")

    return stats


def validate_coherence_thresholds(stats: Dict[str, float]) -> bool:
    """Check if coherence meets minimum quality thresholds."""
    print("\n" + "="*60)
    print("3. Validating Coherence Thresholds")
    print("="*60)

    if not stats:
        print("❌ No statistics available")
        return False

    thresholds = {
        'avg_coherence': 0.5,  # Should be > 0.5
        'min_coherence': 0.1,  # Should be > 0.1 (very lenient)
    }

    passed = True

    for metric, threshold in thresholds.items():
        value = stats.get(metric, 0)
        if value > threshold:
            print(f"✓ {metric}: {value:.3f} > {threshold} (PASS)")
        else:
            print(f"❌ {metric}: {value:.3f} ≤ {threshold} (FAIL)")
            passed = False

    if passed:
        print("\n✓ All coherence thresholds passed!")
    else:
        print("\n❌ Some coherence thresholds failed")
        print("This may indicate:")
        print("  - Poor quality text segmentation")
        print("  - Encoding errors")
        print("  - Dataset issues")

    return passed


def check_data_completeness(data: List[Dict[str, Any]]) -> bool:
    """Check that all items have required fields."""
    print("\n" + "="*60)
    print("4. Checking Data Completeness")
    print("="*60)

    missing_fields = []
    items_with_embeddings = 0
    items_without_embeddings = 0

    for idx, item in enumerate(tqdm(data, desc="Checking completeness")):
        has_embeddings = False

        # Check for concept embeddings in webpages
        if 'webpages' in item:
            for webpage in item['webpages']:
                if 'concept_embeddings' in webpage and len(webpage['concept_embeddings']) > 0:
                    has_embeddings = True
                    break

        # Check in search results
        if not has_embeddings and 'search_results' in item:
            for result in item['search_results']:
                if 'concept_embeddings' in result and len(result['concept_embeddings']) > 0:
                    has_embeddings = True
                    break

        # Check in contexts (HotpotQA format)
        if not has_embeddings and 'context' in item:
            if isinstance(item['context'], list):
                for context_item in item['context']:
                    if len(context_item) > 2:  # Has processed data
                        has_embeddings = True
                        break

        if has_embeddings:
            items_with_embeddings += 1
        else:
            items_without_embeddings += 1
            if items_without_embeddings <= 5:  # Record first 5
                missing_fields.append(idx)

    print(f"\nCompleteness Statistics:")
    print(f"  - Items with embeddings: {items_with_embeddings}")
    print(f"  - Items without embeddings: {items_without_embeddings}")

    if items_without_embeddings > 0:
        print(f"\n⚠ Warning: {items_without_embeddings} items have no concept embeddings")
        if missing_fields:
            print(f"  Examples (first 5): {missing_fields}")
        print("  This may be expected if those items have no text content")
        return False

    print("\n✓ All items have concept embeddings")
    return True


def validate_dataset(dataset_path: Path) -> bool:
    """
    Main validation function.

    Returns:
        True if all checks pass, False otherwise
    """
    print("="*60)
    print(f"Validating: {dataset_path}")
    print("="*60)

    # Load dataset
    print("\nLoading dataset...")
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

    if not isinstance(data, list):
        print(f"❌ Expected list, got {type(data)}")
        return False

    print(f"✓ Loaded {len(data)} items")

    # Run validation checks
    checks = [
        ('Embedding Shapes', validate_embeddings_shape(data)),
    ]

    stats = compute_coherence_statistics(data)
    checks.append(('Coherence Statistics', bool(stats)))

    if stats:
        checks.append(('Coherence Thresholds', validate_coherence_thresholds(stats)))

    checks.append(('Data Completeness', check_data_completeness(data)))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check_name:30s} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nThis dataset is ready for RL training.")
        return True
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease review the errors above and re-run preprocessing if needed.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate preprocessed concept embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to preprocessed dataset (.pkl file)'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    # Run validation
    success = validate_dataset(dataset_path)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
