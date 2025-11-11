#!/usr/bin/env python3
"""
Test Script for Offline Concept Preprocessing

This script tests the complete preprocessing workflow:
1. Generate sample test data
2. Run preprocessing
3. Validate results
4. Test loading and using processed data
5. Clean up

Usage:
    python test_preprocessing.py [--device cuda:0|cpu]

Author: ASearcher Team
"""

import argparse
import json
import pickle
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def create_sample_data() -> List[Dict[str, Any]]:
    """
    Create sample test data for preprocessing.

    Returns:
        List of sample questions with webpages
    """
    print("\n" + "="*60)
    print("Creating Sample Test Data")
    print("="*60)

    sample_data = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "webpages": [
                {
                    "url": "https://example.com/france",
                    "title": "France - Wikipedia",
                    "text": """France is a country located in Western Europe.
                    It is officially known as the French Republic.
                    The capital and largest city is Paris, which is located in the north-central part of the country.
                    Paris is known for the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.
                    France has a population of about 67 million people.
                    The official language is French, and the currency is the Euro.
                    France is a member of the European Union and NATO.
                    The country is known for its wine, cheese, and cuisine."""
                },
                {
                    "url": "https://example.com/paris",
                    "title": "Paris - Wikipedia",
                    "text": """Paris is the capital and most populous city of France.
                    With an estimated population of 2.2 million residents, it is the fourth-largest city in the European Union.
                    The city is a major European center for finance, diplomacy, commerce, fashion, science, and the arts.
                    Paris is home to iconic landmarks such as the Eiffel Tower, which was built in 1889.
                    The Louvre is the world's most visited art museum.
                    Paris is divided into 20 arrondissements (districts)."""
                }
            ]
        },
        {
            "question": "Who wrote Pride and Prejudice?",
            "answer": "Jane Austen",
            "webpages": [
                {
                    "url": "https://example.com/pride-and-prejudice",
                    "title": "Pride and Prejudice",
                    "text": """Pride and Prejudice is a romantic novel by Jane Austen.
                    It was first published in 1813.
                    The novel follows the character development of Elizabeth Bennet.
                    The story is set in rural England in the early 19th century.
                    It is one of the most famous works in English literature.
                    The novel has been adapted into numerous films, television series, and stage productions."""
                },
                {
                    "url": "https://example.com/jane-austen",
                    "title": "Jane Austen",
                    "text": """Jane Austen was an English novelist known for her romantic fiction.
                    She was born on December 16, 1775, in Steventon, Hampshire, England.
                    Austen died on July 18, 1817, at the age of 41.
                    Her major works include Pride and Prejudice, Sense and Sensibility, and Emma.
                    She is one of the most widely read writers in English literature.
                    Her novels are known for their wit, social commentary, and insight into human nature."""
                }
            ]
        },
        {
            "question": "What is the speed of light?",
            "answer": "299,792,458 meters per second",
            "webpages": [
                {
                    "url": "https://example.com/speed-of-light",
                    "title": "Speed of Light",
                    "text": """The speed of light in vacuum, commonly denoted c, is a universal physical constant.
                    It is exactly 299,792,458 meters per second (approximately 300,000 km/s or 186,000 mi/s).
                    According to the special theory of relativity, c is the upper limit for the speed at which conventional matter or energy can travel through space.
                    The speed of light is the same for all observers, regardless of their motion or the motion of the light source.
                    This is one of the fundamental postulates of Einstein's theory of special relativity.
                    Light from the Sun takes approximately 8 minutes and 20 seconds to reach Earth."""
                }
            ]
        }
    ]

    print(f"✓ Created {len(sample_data)} sample questions")
    for i, item in enumerate(sample_data, 1):
        print(f"  {i}. {item['question']}")
        print(f"     - Webpages: {len(item['webpages'])}")

    return sample_data


def setup_test_directories():
    """Create test directories."""
    print("\n" + "="*60)
    print("Setting Up Test Directories")
    print("="*60)

    test_dir = Path("test_data")
    raw_dir = test_dir / "raw"
    processed_dir = test_dir / "processed"

    # Clean up if exists
    if test_dir.exists():
        print(f"⚠ Cleaning up existing test directory: {test_dir}")
        shutil.rmtree(test_dir)

    # Create directories
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    print(f"✓ Created {test_dir}/")
    print(f"  - {raw_dir}/")
    print(f"  - {processed_dir}/")

    return test_dir, raw_dir, processed_dir


def save_test_data(data: List[Dict[str, Any]], output_path: Path):
    """Save test data as JSON."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved test data to {output_path}")


def run_preprocessing(raw_path: Path, processed_path: Path, device: str) -> bool:
    """
    Run preprocessing script.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("Running Preprocessing")
    print("="*60)

    import subprocess

    cmd = [
        sys.executable,
        "preprocess_concepts.py",
        "--input", str(raw_path),
        "--output", str(processed_path),
        "--device", device,
        "--batch-size", "8"  # Small batch for testing
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,  # Show output
            text=True
        )

        if result.returncode == 0:
            print("\n✓ Preprocessing completed successfully")
            return True
        else:
            print(f"\n❌ Preprocessing failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ Error running preprocessing: {e}")
        return False


def run_validation(processed_path: Path) -> bool:
    """
    Run validation script.

    Returns:
        True if validation passed, False otherwise
    """
    print("\n" + "="*60)
    print("Running Validation")
    print("="*60)

    import subprocess

    cmd = [
        sys.executable,
        "validate_concepts.py",
        "--dataset", str(processed_path)
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print("\n✓ Validation passed")
            return True
        else:
            print(f"\n❌ Validation failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ Error running validation: {e}")
        return False


def test_loading_processed_data(processed_path: Path) -> bool:
    """
    Test loading and using processed data.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)

    try:
        # Load processed data
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Loaded {len(data)} items")

        # Check structure
        errors = []

        for idx, item in enumerate(data):
            if 'question' not in item:
                errors.append(f"Item {idx}: Missing 'question' field")

            if 'webpages' not in item:
                errors.append(f"Item {idx}: Missing 'webpages' field")
                continue

            for wp_idx, webpage in enumerate(item['webpages']):
                # Check required fields
                required_fields = ['concept_embeddings', 'sentences', 'num_sentences']
                for field in required_fields:
                    if field not in webpage:
                        errors.append(f"Item {idx}, webpage {wp_idx}: Missing '{field}' field")

                # Check embeddings
                if 'concept_embeddings' in webpage:
                    embeddings = webpage['concept_embeddings']

                    if len(embeddings) == 0:
                        continue  # Empty is OK

                    if not isinstance(embeddings, np.ndarray):
                        errors.append(f"Item {idx}, webpage {wp_idx}: embeddings not numpy array")

                    if embeddings.shape[1] != 1024:
                        errors.append(f"Item {idx}, webpage {wp_idx}: Wrong embedding dim {embeddings.shape[1]}")

                    if webpage['num_sentences'] != embeddings.shape[0]:
                        errors.append(f"Item {idx}, webpage {wp_idx}: Sentence count mismatch")

        if errors:
            print(f"\n❌ Found {len(errors)} errors:")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
            return False

        # Print sample statistics
        print("\nSample Statistics:")
        for i, item in enumerate(data[:3], 1):
            print(f"\n  Question {i}: {item['question']}")
            for j, webpage in enumerate(item['webpages'], 1):
                num_sents = webpage.get('num_sentences', 0)
                emb_shape = webpage.get('concept_embeddings', np.array([])).shape
                print(f"    Webpage {j}: {num_sents} sentences, embeddings shape {emb_shape}")

        print("\n✓ All structure checks passed")
        return True

    except Exception as e:
        print(f"\n❌ Error loading processed data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concept_operations(processed_path: Path) -> bool:
    """
    Test typical operations on concept embeddings.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("Testing Concept Operations")
    print("="*60)

    try:
        # Load data
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)

        # Test 1: Compute coherence
        print("\nTest 1: Computing coherence scores")
        for item in data[:2]:
            for webpage in item['webpages'][:1]:
                embeddings = webpage['concept_embeddings']

                if len(embeddings) <= 1:
                    continue

                # Compute pairwise similarities
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = np.dot(embeddings[i], embeddings[i+1]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
                    )
                    similarities.append(sim)

                avg_coherence = np.mean(similarities)
                print(f"  - Webpage: {webpage.get('title', 'Untitled')}")
                print(f"    Coherence: {avg_coherence:.3f}")
                print(f"    Sentences: {len(embeddings)}")

        # Test 2: Concept similarity to query
        print("\nTest 2: Computing query-concept similarities")
        for item in data[:1]:
            question = item['question']
            print(f"  Question: {question}")

            # Simple mock: use first sentence as "query embedding"
            if item['webpages'] and len(item['webpages'][0]['concept_embeddings']) > 0:
                query_emb = item['webpages'][0]['concept_embeddings'][0]

                for webpage in item['webpages'][:2]:
                    embeddings = webpage['concept_embeddings']

                    if len(embeddings) == 0:
                        continue

                    # Compute similarity to each sentence
                    similarities = []
                    for emb in embeddings:
                        sim = np.dot(query_emb, emb) / (
                            np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8
                        )
                        similarities.append(sim)

                    top_k = sorted(similarities, reverse=True)[:3]
                    print(f"    Webpage: {webpage.get('title', 'Untitled')}")
                    print(f"    Top-3 similarities: {[f'{s:.3f}' for s in top_k]}")

        # Test 3: Mock segment selection
        print("\nTest 3: Mock segment selection (top-k by similarity)")
        for item in data[:1]:
            if not item['webpages'] or len(item['webpages'][0]['concept_embeddings']) == 0:
                continue

            query_emb = item['webpages'][0]['concept_embeddings'][0]

            for webpage in item['webpages'][:1]:
                embeddings = webpage['concept_embeddings']
                sentences = webpage['sentences']

                if len(embeddings) == 0:
                    continue

                # Compute all similarities
                similarities = np.array([
                    np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
                    for emb in embeddings
                ])

                # Select top-3
                top_k_indices = np.argsort(similarities)[-3:][::-1]

                print(f"  Selected top-3 sentences (by mock similarity):")
                for idx in top_k_indices:
                    print(f"    [{idx}] (sim={similarities[idx]:.3f}): {sentences[idx][:80]}...")

        print("\n✓ All concept operations completed successfully")
        return True

    except Exception as e:
        print(f"\n❌ Error in concept operations: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_data(test_dir: Path, keep: bool = False):
    """Clean up test data."""
    print("\n" + "="*60)
    print("Cleanup")
    print("="*60)

    if keep:
        print(f"✓ Keeping test data in {test_dir}/")
        print("  You can inspect the files manually:")
        print(f"    - Raw: {test_dir}/raw/test_sample.json")
        print(f"    - Processed: {test_dir}/processed/test_sample_with_concepts.pkl")
    else:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"✓ Cleaned up test directory: {test_dir}")
        else:
            print(f"⚠ Test directory not found: {test_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Test offline concept preprocessing workflow'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for testing (cuda:0, cuda:1, or cpu). Default: cpu'
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep test data after completion (for manual inspection)'
    )

    args = parser.parse_args()

    print("="*60)
    print("OFFLINE CONCEPT PREPROCESSING TEST")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Keep test data: {args.keep}")

    # Track test results
    tests = []

    try:
        # Step 1: Setup
        sample_data = create_sample_data()
        test_dir, raw_dir, processed_dir = setup_test_directories()

        raw_path = raw_dir / "test_sample.json"
        processed_path = processed_dir / "test_sample_with_concepts.pkl"

        save_test_data(sample_data, raw_path)

        # Step 2: Preprocessing
        success = run_preprocessing(raw_path, processed_path, args.device)
        tests.append(("Preprocessing", success))

        if not success:
            print("\n❌ Preprocessing failed, skipping remaining tests")
            cleanup_test_data(test_dir, keep=args.keep)
            sys.exit(1)

        # Step 3: Validation
        success = run_validation(processed_path)
        tests.append(("Validation", success))

        # Step 4: Loading test
        success = test_loading_processed_data(processed_path)
        tests.append(("Data Loading", success))

        # Step 5: Concept operations
        success = test_concept_operations(processed_path)
        tests.append(("Concept Operations", success))

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        cleanup_test_data(test_dir, keep=False)
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(test_dir, keep=False)
        sys.exit(1)

    # Cleanup
    cleanup_test_data(test_dir, keep=args.keep)

    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in tests:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe preprocessing pipeline is working correctly.")
        print("You can now use it on real datasets.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check the errors above and fix any issues.")
        print("Common issues:")
        print("  1. SONAR not installed: pip install sonar-space")
        print("  2. NLTK data missing: python -c 'import nltk; nltk.download(\"punkt\")'")
        print("  3. GPU not available: use --device cpu")
        sys.exit(1)


if __name__ == '__main__':
    main()
