#!/bin/bash
#
# Quick Smoke Test for Preprocessing Pipeline
#
# This is a minimal test to verify SONAR installation and basic functionality.
# For comprehensive testing, use: python test_preprocessing.py
#
# Usage:
#   bash quick_test.sh [device]
#
# Example:
#   bash quick_test.sh cpu
#   bash quick_test.sh cuda:0

set -e  # Exit on error

DEVICE="${1:-cpu}"

echo "========================================"
echo "Quick Smoke Test"
echo "========================================"
echo "Device: $DEVICE"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ ERROR: Python not found"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Check if required packages are installed
echo ""
echo "Checking dependencies..."

python -c "import sonar" 2>/dev/null && echo "✓ SONAR installed" || {
    echo "❌ SONAR not installed"
    echo "Install with: pip install sonar-space"
    exit 1
}

python -c "import nltk" 2>/dev/null && echo "✓ NLTK installed" || {
    echo "❌ NLTK not installed"
    echo "Install with: pip install nltk"
    exit 1
}

python -c "import numpy" 2>/dev/null && echo "✓ NumPy installed" || {
    echo "❌ NumPy not installed"
    echo "Install with: pip install numpy"
    exit 1
}

# Check NLTK data
echo ""
echo "Checking NLTK data..."
python -c "import nltk; nltk.data.find('tokenizers/punkt')" 2>/dev/null && echo "✓ NLTK punkt tokenizer found" || {
    echo "⚠ NLTK punkt tokenizer not found"
    echo "Downloading..."
    python -c "import nltk; nltk.download('punkt', quiet=True)"
    echo "✓ Downloaded punkt tokenizer"
}

# Test SONAR loading
echo ""
echo "Testing SONAR loading on $DEVICE..."
python -c "
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
print('Loading SONAR encoder...')
encoder = TextToEmbeddingModelPipeline(
    encoder='text_sonar_basic_encoder',
    tokenizer='text_sonar_basic_encoder',
    device='$DEVICE'
)
print('✓ SONAR encoder loaded successfully')

# Test encoding
test_sentences = ['This is a test sentence.', 'Another test sentence.']
print('Testing encoding...')
embeddings = encoder.predict(test_sentences, source_lang='eng_Latn')
print(f'✓ Encoding successful, shape: {embeddings.shape}')

if embeddings.shape[1] != 1024:
    print(f'❌ ERROR: Expected 1024-dim embeddings, got {embeddings.shape[1]}')
    exit(1)

print('✓ Embeddings have correct dimension (1024)')
" || {
    echo "❌ SONAR test failed"
    exit 1
}

# All checks passed
echo ""
echo "========================================"
echo "✓ ALL CHECKS PASSED!"
echo "========================================"
echo ""
echo "Your environment is ready for preprocessing."
echo ""
echo "Next steps:"
echo "  1. Run full test: python test_preprocessing.py --device $DEVICE"
echo "  2. Process real data: python preprocess_concepts.py --input your_data.json --output output.pkl --device $DEVICE"
echo ""
