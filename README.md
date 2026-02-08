# llama.cpp - QFuse Branch

This branch extends llama.cpp with **QFuse (Q4_K_F)** quantization support, an outlier-aware quantization method that provides improved accuracy for 4-bit quantized models. This branch also includes **AWQ (Activation-aware Weight Quantization)** channel protection support for enhanced quantization quality.

## What is QFuse (Q4_K_F)?

**QFuse** is a new 4-bit quantization format (`Q4_K_F`) that uses an outlier-aware approach to preserve important weight channels during quantization. Unlike standard Q4_K quantization methods, QFuse:

- **Outlier-aware quantization**: Identifies and preserves critical weight channels that significantly impact model accuracy
- **Improved accuracy**: Better perplexity and quality compared to standard Q4_K_M quantization
- **AWQ integration**: Supports Activation-aware Weight Quantization channel protection for even better results
- **Optimized backends**: Includes optimized CPU and Metal (Apple Silicon) implementations

## Key Features

### 1. Q4_K_F Quantization Format
- New quantization type: `GGML_TYPE_Q4_K_F`
- Outlier-aware quantization that preserves important channels
- Similar model size to Q4_K_M (~4.58 GiB for Llama-3-8B) with improved accuracy

### 2. AWQ Channel Protection
- **Activation-aware Weight Quantization** support for Q4_K_F
- Protects a configurable fraction of channels (default: 1%) that are most important for model accuracy
- Uses importance matrix (imatrix) when available for better channel selection
- Stores protected channel indices and weights as metadata in GGUF format

### 3. Backend Optimizations

#### CPU Backend
- Optimized matrix multiplication kernels for Q4_K_F
- Tile-based processing (128-element tiles) for efficient computation
- AWQ correction kernels for handling protected channels

#### Metal Backend (Apple Silicon)
- Custom Metal shaders for Q4_K_F operations
- Optimized matrix multiplication with outlier handling
- Flash attention support with Q4_K_F tensors
- Efficient memory management for AWQ metadata

### 4. Additional Improvements
- Enhanced quantization tool with AWQ options
- GSM8K evaluation script for testing model quality
- Number grammar for structured number generation
- Improved GGUF metadata handling for quantization information

## Differences from Main Branch

This branch adds the following features not present in the main llama.cpp repository:

1. **Q4_K_F Quantization Type**: New `GGML_TYPE_Q4_K_F` quantization format
2. **AWQ Support**: Full AWQ channel protection implementation with metadata storage
3. **AWQ Header**: New `ggml/include/ggml-awq.h` header for AWQ tensor metadata
4. **Enhanced Quantization**: Improved quantization logic in `llama-quant.cpp` with AWQ channel selection
5. **Backend Optimizations**: CPU and Metal backend optimizations for Q4_K_F operations
6. **Evaluation Tools**: GSM8K evaluation script for quantitative testing
7. **Grammar Support**: Number grammar for structured outputs

## Usage

### Quantizing Models with Q4_K_F

Basic quantization to Q4_K_F:
```bash
./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_F.gguf Q4_K_F
```

### Quantizing with AWQ Channel Protection

For best results, use AWQ channel protection:
```bash
# With default 1% channel protection
./llama-quantize --awq ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_F-awq.gguf Q4_K_F

# With custom channel protection ratio (e.g., 2%)
./llama-quantize --awq --awq-ratio 0.02 ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_F-awq.gguf Q4_K_F

# With importance matrix for better channel selection
./llama-quantize --awq --awq-ratio 0.01 --imatrix imatrix.gguf ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_F-awq.gguf Q4_K_F
```

### Running Quantized Models

Models quantized with Q4_K_F work seamlessly with all llama.cpp tools:
```bash
# Using llama-cli
./llama-cli -m ./models/mymodel/ggml-model-Q4_K_F.gguf -cnv -p "You are a helpful assistant"

# Using llama-server
./llama-server -m ./models/mymodel/ggml-model-Q4_K_F.gguf --port 8080
```

## AWQ Channel Protection Details

AWQ (Activation-aware Weight Quantization) protects a small fraction of weight channels from quantization by:

1. **Channel Selection**: Identifies the most important channels based on:
   - Weight magnitude (absolute values)
   - Importance matrix values (if provided via `--imatrix`)
   - Combined scoring: `score = |weight| * |imatrix_value|`

2. **Protection Ratio**: Configurable via `--awq-ratio` (default: 0.01 = 1%)
   - Higher ratios preserve more channels but increase model size
   - Lower ratios reduce model size but may impact accuracy

3. **Metadata Storage**: Protected channel information is stored in GGUF format:
   - Channel indices for each tensor
   - Protected channel weights (F16 format)
   - Expert count for MoE models

4. **Runtime Correction**: During inference, protected channels are handled with full precision, providing accuracy improvements while maintaining most of the quantization benefits.

## Performance Characteristics

### Model Sizes (LFM2.5-1.2B-Instruct)

| Format | Size | Notes |
|--------|------|-------|
| F16 (baseline) | 2.34 GB | Original full precision |
| Q4_K_M | 730.9 MB | Standard 4-bit quantization |
| Q4_K_F | 787.1 MB | Outlier-aware, improved accuracy |
| Q4_K_F + AWQ | 804.8 MB | With 1% channel protection |
| Q5_K_S | 825.3 MB | 5-bit quantization (small) |
| Q5_K_M | 843.4 MB | 5-bit quantization (medium) |

### Perplexity vs F16 (baseline)

Tested on LFM2.5-1.2B-Instruct model:

| Metric | F16 | Q4_K_M | Δ vs F16 | % vs F16 | Q4_K_F | Δ vs F16 | % vs F16 | Q5_K_M | Δ vs F16 | % vs F16 | Q5_K_S | Δ vs F16 | % vs F16 |
|--------|-----|--------|----------|----------|--------|----------|----------|--------|----------|----------|--------|----------|----------|
| Last (chunk 50) | 24.5452 | 26.7186 | 2.1734 | 8.85% | 25.5672 | 1.0220 | 4.16% | 25.4721 | 0.9269 | 3.78% | 25.8600 | 1.3148 | 5.36% |
| Mean (all 50) | 28.0810 | 30.2387 | 2.1577 | 7.68% | 28.7362 | 0.6552 | 2.33% | 29.1374 | 1.0564 | 3.76% | 29.4770 | 1.3960 | 4.97% |
| Mean (last 10) | 24.5120 | 26.7509 | 2.2390 | 9.13% | 25.4779 | 0.9659 | 3.94% | 25.4378 | 0.9258 | 3.78% | 25.8056 | 1.2936 | 5.28% |
| Mean (last 20) | 24.7377 | 26.9298 | 2.1921 | 8.86% | 25.6638 | 0.9261 | 3.74% | 25.6669 | 0.9292 | 3.76% | 26.0244 | 1.2867 | 5.20% |

**Key Observations:**
- **Q4_K_F** significantly outperforms **Q4_K_M** across all metrics, with only ~7.7% size increase
- **Q4_K_F** achieves perplexity within **2.33-4.16%** of F16 baseline, compared to **7.68-9.13%** for Q4_K_M
- **Q4_K_F** performs comparably to **Q5_K_M** while using less memory (787.1 MB vs 843.4 MB)

### GSM8K Evaluation (100 questions)

Tested on LFM2.5-1.2B-Instruct model:

| Format | Score | Accuracy |
|--------|-------|----------|
| F16 (baseline) | 68/100 | 0.6800 |
| Q4_K_F + AWQ | 68/100 | 0.6800 |
| Q4_K_F | 67/100 | 0.6700 |
| Q4_K_M | 64/100 | 0.6400 |

**Key Observations:**
- **Q4_K_F + AWQ** matches F16 baseline performance (68/100)
- **Q4_K_F** achieves 98.5% of F16 accuracy (67/100 vs 68/100)
- **Q4_K_M** shows 6% accuracy drop compared to F16 (64/100 vs 68/100)

## Technical Details

### File Changes Summary

- **29 files changed**: 2,162 insertions, 252 deletions
- **New files**:
  - `ggml/include/ggml-awq.h`: AWQ tensor metadata structures
  - `scripts/gsm8k_eval.py`: Evaluation script for GSM8K dataset
  - `grammars/number.gbnf`: Number grammar for structured outputs

- **Modified files**:
  - Quantization: `src/llama-quant.cpp`, `tools/quantize/quantize.cpp`
  - CPU backend: `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/quants.c`
  - Metal backend: `ggml/src/ggml-metal/ggml-metal-ops.cpp`, `ggml/src/ggml-metal/ggml-metal.metal`
  - Model loading: `src/llama-model.cpp`, `src/llama-model-loader.cpp`
  - GGUF support: `gguf-py/gguf/constants.py`, `gguf-py/gguf/quants.py`

### Quantization Type Constants

- `GGML_TYPE_Q4_K_F = 40`: New quantization type identifier
- `LLAMA_FTYPE_MOSTLY_Q4_K_F`: File type for Q4_K_F quantized models

## Building

Build instructions are the same as the main llama.cpp repository:

```bash
# Standard build
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Or use the provided build scripts
./scripts/build.sh
```

## Testing

Use the included GSM8K evaluation script to test model quality:

```bash
python3 scripts/gsm8k_eval.py --model ./models/mymodel/ggml-model-Q4_K_F.gguf --dataset ./models/datasets/gsm8k/test.jsonl
```

## Contributing

This branch is maintained as a fork. For contributions:

1. Open issues or pull requests on this repository
2. Ensure compatibility with the main llama.cpp codebase
3. Include tests for new quantization features

## License

Same as main llama.cpp: MIT License

## References

- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [llama.cpp Main Repository](https://github.com/ggml-org/llama.cpp)
- [Quantization Documentation](./tools/quantize/README.md)
