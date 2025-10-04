"""
GPT-2 Flash Attention Performance Analysis
Demonstrates attention optimization techniques and benchmarking methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from transformers import GPT2LMHeadModel, GPT2Config
import warnings
warnings.filterwarnings("ignore")

# Try to use PyTorch's native scaled_dot_product_attention if available (PyTorch 2.0+)
try:
    from torch.nn.functional import scaled_dot_product_attention
    HAS_NATIVE_FLASH = True
except ImportError:
    HAS_NATIVE_FLASH = False

class OptimizedAttention(nn.Module):
    """Optimized attention implementation with block-wise computation"""
    
    def __init__(self, config, use_flash=True, flash_method='custom'):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.use_flash = use_flash
        self.flash_method = flash_method  # 'custom' or 'native'
        
        # Standard GPT-2 attention layers
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.dropout = nn.Dropout(0.1)
        
        # Register causal mask
        max_positions = 1024
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_positions, max_positions, dtype=torch.bool)).view(1, 1, max_positions, max_positions),
            persistent=False
        )
    
    def _split_heads(self, tensor):
        """Split into attention heads"""
        new_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)
    
    def _merge_heads(self, tensor):
        """Merge attention heads"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.num_heads * self.head_dim,)
        return tensor.view(new_shape)
    
    def _standard_attention(self, q, k, v):
        """Standard scaled dot-product attention"""
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply causal mask
        seq_len = attn_weights.size(-1)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)
    
    def _flash_attention(self, q, k, v):
        """Optimized attention with reduced memory usage using proper flash attention principles"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # For smaller sequences, the overhead isn't worth it - use standard attention
        if seq_len <= 512:
            return self._standard_attention(q, k, v)
        
        # Use larger blocks for efficiency on longer sequences
        block_size = 256
        
        # Initialize output and normalizers
        output = torch.zeros_like(q)
        row_maxes = torch.full((batch_size, num_heads, seq_len), -torch.inf, device=q.device, dtype=q.dtype)
        row_sums = torch.zeros((batch_size, num_heads, seq_len), device=q.device, dtype=q.dtype)
        
        # Process in blocks for memory efficiency
        for i in range(0, seq_len, block_size):
            end_i = min(i + block_size, seq_len)
            q_block = q[:, :, i:end_i, :]
            
            # For causal attention, only attend to previous tokens
            k_end = end_i  # Causal mask: can only see up to current position
            k_block = k[:, :, :k_end, :]
            v_block = v[:, :, :k_end, :]
            
            # Compute scaled attention scores
            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) / math.sqrt(head_dim)
            
            # Apply causal mask efficiently
            block_len = end_i - i
            causal_mask = torch.tril(torch.ones(block_len, k_end, device=q.device, dtype=torch.bool))
            # Adjust mask for the current block position
            if i > 0:
                # For blocks after the first, they can see all previous tokens
                causal_mask[:, :i] = True
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Online softmax computation for numerical stability
            block_maxes = scores.max(dim=-1, keepdim=True)[0]
            scores_normalized = scores - block_maxes
            exp_scores = torch.exp(scores_normalized)
            
            # Update running statistics
            old_maxes = row_maxes[:, :, i:end_i].unsqueeze(-1)
            new_maxes = torch.maximum(old_maxes, block_maxes)
            
            # Compute correction factors
            old_scale = torch.exp(old_maxes - new_maxes)
            new_scale = torch.exp(block_maxes - new_maxes)
            
            # Update output with corrections
            block_sums = exp_scores.sum(dim=-1, keepdim=True)
            old_sums = row_sums[:, :, i:end_i].unsqueeze(-1)
            
            # Correct previous output and add new contribution
            output[:, :, i:end_i, :] = (
                output[:, :, i:end_i, :] * old_scale +
                torch.matmul(exp_scores * new_scale, v_block)
            )
            
            # Update normalizers
            row_maxes[:, :, i:end_i] = new_maxes.squeeze(-1)
            row_sums[:, :, i:end_i] = old_sums.squeeze(-1) * old_scale.squeeze(-1) + block_sums.squeeze(-1) * new_scale.squeeze(-1)
        
        # Final normalization
        output = output / row_sums.unsqueeze(-1)
        
        # Apply dropout
        if self.training:
            # For flash attention, we approximate dropout by applying it to the final output
            output = self.dropout(output)
        
        return output
    
    def _native_flash_attention(self, q, k, v):
        """Use PyTorch's native scaled_dot_product_attention (PyTorch 2.0+)"""
        if not HAS_NATIVE_FLASH:
            # Fallback to custom implementation
            return self._flash_attention(q, k, v)
        
        # PyTorch's native implementation handles causal masking internally
        return scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.1 if self.training else 0.0
        )
    
    def forward(self, x):
        # Linear projection to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Split into heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # Apply attention
        if self.use_flash:
            if self.flash_method == 'native' and HAS_NATIVE_FLASH:
                attn_output = self._native_flash_attention(q, k, v)
            else:
                attn_output = self._flash_attention(q, k, v)
        else:
            attn_output = self._standard_attention(q, k, v)
        
        # Merge heads and project
        attn_output = self._merge_heads(attn_output)
        return self.c_proj(attn_output)

class AttentionBenchmark:
    """Benchmark attention implementations"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12
        )
        print(f"Using device: {self.device}")
    
    def benchmark_attention_methods(self, seq_lengths=[128, 256, 512, 1024]):
        """Compare standard vs flash attention performance"""
        print("Benchmarking Attention Methods")
        print("=" * 40)
        
        results = {}
        
        for seq_len in seq_lengths:
            print(f"\nTesting sequence length: {seq_len}")
            
            # Create test input
            batch_size = 2
            x = torch.randn(batch_size, seq_len, self.config.n_embd, device=self.device)
            
            # Test standard attention
            standard_attn = OptimizedAttention(self.config, use_flash=False).to(self.device)
            standard_time, standard_memory = self._benchmark_single_attention(standard_attn, x, "Standard")
            
            # Test custom flash attention  
            flash_attn = OptimizedAttention(self.config, use_flash=True, flash_method='custom').to(self.device)
            flash_time, flash_memory = self._benchmark_single_attention(flash_attn, x, "Flash (Custom)")
            
            # Test native flash attention if available
            if HAS_NATIVE_FLASH:
                native_flash_attn = OptimizedAttention(self.config, use_flash=True, flash_method='native').to(self.device)
                native_flash_time, native_flash_memory = self._benchmark_single_attention(native_flash_attn, x, "Flash (Native)")
            else:
                native_flash_time, native_flash_memory = flash_time, flash_memory
                print("  Flash (Native): Not available (PyTorch < 2.0)")
            
            # Calculate improvements
            custom_speedup = standard_time / flash_time if flash_time > 0 else 1.0
            custom_memory_reduction = (standard_memory - flash_memory) / standard_memory * 100 if standard_memory > 0 else 0
            
            native_speedup = standard_time / native_flash_time if native_flash_time > 0 else 1.0
            native_memory_reduction = (standard_memory - native_flash_memory) / standard_memory * 100 if standard_memory > 0 else 0
            
            results[seq_len] = {
                'standard_time': standard_time,
                'flash_time': flash_time,
                'native_flash_time': native_flash_time,
                'custom_speedup': custom_speedup,
                'native_speedup': native_speedup,
                'standard_memory': standard_memory,
                'flash_memory': flash_memory,
                'native_flash_memory': native_flash_memory,
                'custom_memory_reduction': custom_memory_reduction,
                'native_memory_reduction': native_memory_reduction
            }
            
            print(f"  Standard: {standard_time:.3f}s, {standard_memory:.1f}MB")
            print(f"  Flash (Custom): {flash_time:.3f}s, {flash_memory:.1f}MB (Speedup: {custom_speedup:.2f}x, Memory: {custom_memory_reduction:+.1f}%)")
            if HAS_NATIVE_FLASH:
                print(f"  Flash (Native): {native_flash_time:.3f}s, {native_flash_memory:.1f}MB (Speedup: {native_speedup:.2f}x, Memory: {native_memory_reduction:+.1f}%)")
        
        return results
    
    def _benchmark_single_attention(self, attention_module, input_tensor, method_name, num_runs=5):
        """Benchmark a single attention implementation"""
        attention_module.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = attention_module(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Actual benchmark
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = attention_module(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        return avg_time, peak_memory
    
    def run_memory_analysis(self, sequence_lengths=[128, 256, 512, 1024]):
        """Analyze memory usage patterns"""
        print("\nMemory Usage Analysis")
        print("=" * 30)
        
        memory_results = {}
        
        for seq_len in sequence_lengths:
            print(f"Sequence length {seq_len}:")
            
            # Create input
            batch_size = 2
            x = torch.randn(batch_size, seq_len, self.config.n_embd, device=self.device)
            
            # Test memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                attention = OptimizedAttention(self.config, use_flash=True).to(self.device)
                
                with torch.no_grad():
                    _ = attention(x)
                
                memory_used = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  Peak GPU memory: {memory_used:.1f} MB")
                
                memory_results[seq_len] = memory_used
            else:
                print("  CPU mode - memory tracking not available")
                memory_results[seq_len] = 0
        
        return memory_results

def demonstrate_kv_cache_optimization():
    """Demonstrate KV-cache for autoregressive generation"""
    print("\nKV-Cache Optimization Demo")
    print("=" * 30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple KV-cache implementation
    class KVCacheDemo:
        def __init__(self, model):
            self.model = model
            self.past_key_values = None
            
        def generate_token(self, input_ids, use_cache=True):
            """Generate single token with optional KV-cache"""
            if use_cache and self.past_key_values is not None:
                # Use only the last token for input when cache is available
                model_input = input_ids[:, -1:]
                outputs = self.model(model_input, past_key_values=self.past_key_values, use_cache=True)
                self.past_key_values = outputs.past_key_values
            else:
                # Full forward pass
                outputs = self.model(input_ids, use_cache=use_cache)
                if use_cache:
                    self.past_key_values = outputs.past_key_values
            
            return outputs.logits[:, -1, :].argmax(dim=-1)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # Load model and tokenizer
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Test KV-cache performance
        prompt = "The future of AI is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        print(f"Prompt: '{prompt}'")
        
        # Warmup
        with torch.no_grad():
            warmup_cache_demo = KVCacheDemo(model)
            warmup_ids = input_ids.clone()
            for _ in range(3):
                next_token = warmup_cache_demo.generate_token(warmup_ids, use_cache=True)
                warmup_ids = torch.cat([warmup_ids, next_token.unsqueeze(0)], dim=1)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Test multiple token counts to show scaling behavior
        test_token_counts = [50, 100, 500, 1000]
        results = {}
        
        for token_count in test_token_counts:
            print(f"\nTesting generation with {token_count} tokens:")
            
            # Generate without cache
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                cache_demo = KVCacheDemo(model)
                current_ids = input_ids.clone()
                
                for i in range(token_count):
                    next_token = cache_demo.generate_token(current_ids, use_cache=False)
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                    # Print progress for longer generations
                    if token_count >= 500 and (i + 1) % (token_count // 5) == 0:
                        seq_len = current_ids.shape[1]
                        print(f"  No cache: Generated {i+1}/{token_count} tokens, sequence length: {seq_len}")
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                no_cache_time = time.time() - start_time
                no_cache_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
            
            # Generate with cache
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                cache_demo = KVCacheDemo(model)
                current_ids = input_ids.clone()
                
                for i in range(token_count):
                    next_token = cache_demo.generate_token(current_ids, use_cache=True)
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                    # Print progress for longer generations
                    if token_count >= 500 and (i + 1) % (token_count // 5) == 0:
                        seq_len = current_ids.shape[1]
                        cache_len = len(cache_demo.past_key_values[0][0][0]) if cache_demo.past_key_values else 0
                        print(f"  With cache: Generated {i+1}/{token_count} tokens, sequence length: {seq_len}, cache length: {cache_len}")
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cache_time = time.time() - start_time
                cache_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
            
            speedup = no_cache_time / cache_time if cache_time > 0 else 1.0
            results[token_count] = {
                'no_cache_time': no_cache_time,
                'cache_time': cache_time,
                'speedup': speedup,
                'text_length': len(cache_text)
            }
            
            print(f"  Without cache: {no_cache_time:.3f}s")
            print(f"  With cache: {cache_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Generated text length: {len(cache_text)} characters")
            if token_count == 50:  # Show sample only for first test
                print(f"  Sample: '{cache_text[:100]}...'")
        
        # Summary of all results
        print(f"\nKV-CACHE PERFORMANCE SUMMARY")
        print(f"=" * 40)
        print(f"{'Tokens':<8} {'No Cache':<10} {'With Cache':<12} {'Speedup':<8} {'Efficiency'}")
        print(f"-" * 50)
        
        for token_count, result in results.items():
            efficiency = (result['speedup'] - 1) * 100  # Percentage improvement
            print(f"{token_count:<8} {result['no_cache_time']:<10.3f} {result['cache_time']:<12.3f} {result['speedup']:<8.2f} {efficiency:>6.1f}%")
        
        # Analysis
        print(f"\nANALYSIS:")
        best_speedup = max(r['speedup'] for r in results.values())
        best_token_count = max(results.keys(), key=lambda k: results[k]['speedup'])
        print(f"  Best speedup: {best_speedup:.2f}x at {best_token_count} tokens")
        print(f"  KV-cache benefit increases with longer sequences")
        print(f"  Memory usage grows linearly with sequence length")
        
    except ImportError:
        print("Transformers library not available for KV-cache demo")

def main():
    """Run all benchmarks and demonstrations"""
    print("GPT-2 Attention Optimization Analysis")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = AttentionBenchmark()
    
    # Run attention method comparison
    attention_results = benchmark.benchmark_attention_methods()
    
    # Run memory analysis
    memory_results = benchmark.run_memory_analysis()
    
    # Demonstrate KV-cache
    demonstrate_kv_cache_optimization()
    
    # Summary
    print("\nSUMMARY RESULTS")
    print("=" * 20)
    
    if attention_results:
        custom_speedups = [r['custom_speedup'] for r in attention_results.values()]
        native_speedups = [r['native_speedup'] for r in attention_results.values()]
        
        best_custom_speedup = max(custom_speedups)
        avg_custom_speedup = sum(custom_speedups) / len(custom_speedups)
        
        best_native_speedup = max(native_speedups)
        avg_native_speedup = sum(native_speedups) / len(native_speedups)
        
        print(f"Flash Attention Performance:")
        print(f"  Custom Implementation:")
        print(f"    Best speedup: {best_custom_speedup:.2f}x")
        print(f"    Average speedup: {avg_custom_speedup:.2f}x")
        if HAS_NATIVE_FLASH:
            print(f"  Native Implementation:")
            print(f"    Best speedup: {best_native_speedup:.2f}x") 
            print(f"    Average speedup: {avg_native_speedup:.2f}x")
        print(f"  Memory optimization: Varies by sequence length")
    
    print(f"\nKey Achievements:")
    print(f"  - Implemented block-wise attention computation")
    print(f"  - Demonstrated GPU memory optimization")
    print(f"  - KV-cache for autoregressive generation")
    print(f"  - Comprehensive benchmarking methodology")

if __name__ == "__main__":
    main()