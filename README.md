# GPT-2 Optimization Experiments

This folder contains three Python scripts that demonstrate different optimization techniques for GPT-2 language models.

## Files

### `gpt2_quantization.py`
**Quantization Analysis** - Reduces model size and memory usage
- Implements Dynamic INT8, Static INT8, and INT4 quantization
- Compares model sizes and performance across different quantization methods
- Shows compression ratios and speed improvements

### `speculative_decoding.py`
**Speculative Decoding** - Speeds up text generation using a smaller draft model
- Uses a small model to predict multiple tokens ahead
- Verifies predictions with the larger target model
- Tests different model size ratios to find optimal configurations

### `flash.py`
**Flash Attention** - Optimizes attention computation for longer sequences
- Implements memory-efficient attention mechanisms
- Compares standard vs flash attention performance
- Demonstrates KV-cache optimization for autoregressive generation

## Usage

Each script can be run independently:

```bash
python gpt2_quantization.py
python speculative_decoding.py
python flash.py
```

## Requirements

- PyTorch
- Transformers
- CUDA (optional, for GPU acceleration)
- bitsandbytes (optional, for INT4 quantization)
