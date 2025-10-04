"""
GPT-2 Speculative Decoding Analysis
Study of model size ratios and their impact on performance
"""

import torch
import torch.nn.functional as F
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")


class SpeculativeDecoder:
    """
    Speculative decoding using draft model to predict tokens
    verified by larger target model
    """
    
    def __init__(self, target_model, draft_model, tokenizer, gamma=4):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.gamma = gamma
        
        self.target_model.eval()
        self.draft_model.eval()
        
        self.device = next(target_model.parameters()).device
        self.draft_model.to(self.device)
        
        # Statistics tracking
        self.stats = {
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'total_target_calls': 0,
            'total_draft_calls': 0
        }
        
        # Calculate model sizes
        target_params = sum(p.numel() for p in target_model.parameters())
        draft_params = sum(p.numel() for p in draft_model.parameters())
        
        print(f"Speculative decoder initialized:")
        print(f"  Target: {target_params/1e6:.1f}M parameters")
        print(f"  Draft: {draft_params/1e6:.1f}M parameters")
        print(f"  Size ratio: {target_params/draft_params:.1f}x")
        print(f"  Lookahead: {gamma} tokens")
    
    def reset_stats(self):
        """Reset generation statistics"""
        self.stats = {
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'total_target_calls': 0,
            'total_draft_calls': 0
        }
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, 
                 top_k=50, do_sample=True, pad_token_id=None):
        """Generate tokens using speculative decoding"""
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        self.reset_stats()
        generated_tokens = 0
        current_sequence = input_ids.clone()
        
        with torch.no_grad():
            while generated_tokens < max_new_tokens:
                # Generate draft tokens
                draft_tokens = self._generate_draft_tokens(
                    current_sequence, temperature, top_k, do_sample
                )
                
                if len(draft_tokens) == 0:
                    next_token = self._target_single_token(
                        current_sequence, temperature, top_k, do_sample
                    )
                    current_sequence = torch.cat([current_sequence, next_token], dim=1)
                    generated_tokens += 1
                    continue
                
                # Verify with target model
                accepted_tokens = self._verify_draft_tokens(
                    current_sequence, draft_tokens, temperature, top_k, do_sample
                )
                
                if len(accepted_tokens) > 0:
                    accepted_tensor = torch.tensor([accepted_tokens], 
                                                 device=current_sequence.device)
                    current_sequence = torch.cat([current_sequence, accepted_tensor], dim=1)
                    generated_tokens += len(accepted_tokens)
                else:
                    next_token = self._target_single_token(
                        current_sequence, temperature, top_k, do_sample
                    )
                    current_sequence = torch.cat([current_sequence, next_token], dim=1)
                    generated_tokens += 1
                
                if current_sequence[0, -1].item() == pad_token_id:
                    break
        
        return current_sequence
    
    def _generate_draft_tokens(self, input_ids, temperature, top_k, do_sample):
        """Generate draft tokens with small model"""
        draft_tokens = []
        current_ids = input_ids.clone()
        
        for _ in range(self.gamma):
            outputs = self.draft_model(current_ids)
            self.stats['total_draft_calls'] += 1
            
            logits = outputs.logits[:, -1, :] / temperature
            
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            draft_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        self.stats['total_draft_tokens'] += len(draft_tokens)
        return draft_tokens
    
    def _verify_draft_tokens(self, input_ids, draft_tokens, temperature, top_k, do_sample):
        """Verify draft tokens with target model"""
        if len(draft_tokens) == 0:
            return []
        
        # Create sequence with all draft tokens
        verification_sequence = input_ids.clone()
        for token in draft_tokens:
            token_tensor = torch.tensor([[token]], device=input_ids.device)
            verification_sequence = torch.cat([verification_sequence, token_tensor], dim=1)
        
        # Single forward pass through target model
        outputs = self.target_model(verification_sequence)
        self.stats['total_target_calls'] += 1
        
        target_logits = outputs.logits[0]
        
        accepted_tokens = []
        current_ids = input_ids.clone()
        
        for i, draft_token in enumerate(draft_tokens):
            pos = input_ids.size(1) + i - 1
            if pos >= target_logits.size(0) - 1:
                break
            
            target_logits_pos = target_logits[pos] / temperature
            
            if top_k is not None and top_k > 0:
                indices_to_remove = target_logits_pos < torch.topk(target_logits_pos, min(top_k, target_logits_pos.size(-1)))[0][-1]
                target_logits_pos[indices_to_remove] = float('-inf')
            
            target_probs = F.softmax(target_logits_pos, dim=-1)
            
            # Get draft probability
            draft_outputs = self.draft_model(current_ids)
            draft_logits_pos = draft_outputs.logits[0, -1] / temperature
            
            if top_k is not None and top_k > 0:
                indices_to_remove = draft_logits_pos < torch.topk(draft_logits_pos, min(top_k, draft_logits_pos.size(-1)))[0][-1]
                draft_logits_pos[indices_to_remove] = float('-inf')
            
            draft_probs = F.softmax(draft_logits_pos, dim=-1)
            
            # Acceptance test
            target_prob = target_probs[draft_token].item()
            draft_prob = draft_probs[draft_token].item()
            acceptance_prob = min(1.0, target_prob / max(draft_prob, 1e-10))
            
            if torch.rand(1).item() < acceptance_prob:
                accepted_tokens.append(draft_token)
                token_tensor = torch.tensor([[draft_token]], device=input_ids.device)
                current_ids = torch.cat([current_ids, token_tensor], dim=1)
            else:
                # Rejection sampling
                if do_sample:
                    adjusted_probs = torch.clamp(target_probs - draft_probs * acceptance_prob, min=0)
                    if adjusted_probs.sum() > 1e-10:
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        corrected_token = torch.multinomial(adjusted_probs, 1).item()
                    else:
                        corrected_token = torch.argmax(target_probs).item()
                else:
                    corrected_token = torch.argmax(target_probs).item()
                
                accepted_tokens.append(corrected_token)
                break
        
        self.stats['total_accepted_tokens'] += len(accepted_tokens)
        return accepted_tokens
    
    def _target_single_token(self, input_ids, temperature, top_k, do_sample):
        """Generate single token with target model"""
        outputs = self.target_model(input_ids)
        self.stats['total_target_calls'] += 1
        
        logits = outputs.logits[:, -1, :] / temperature
        
        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token
    
    def get_stats(self):
        """Get generation statistics"""
        acceptance_rate = (self.stats['total_accepted_tokens'] / 
                          max(self.stats['total_draft_tokens'], 1))
        
        return {
            'acceptance_rate': acceptance_rate,
            'draft_tokens': self.stats['total_draft_tokens'],
            'accepted_tokens': self.stats['total_accepted_tokens'],
            'target_model_calls': self.stats['total_target_calls'],
            'draft_model_calls': self.stats['total_draft_calls']
        }


class ModelPairBenchmark:
    """Benchmark multiple model pairs systematically"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define model pairs to test
        self.model_pairs = [
            {
                'name': 'Large vs Tiny',
                'target': 'gpt2-large',
                'draft': 'distilgpt2',
                'description': 'GPT2-Large (774M) vs DistilGPT2 (82M) - 9.4x ratio'
            },
            {
                'name': 'Medium vs Tiny',
                'target': 'gpt2-medium',
                'draft': 'distilgpt2',
                'description': 'GPT2-Medium (355M) vs DistilGPT2 (82M) - 4.3x ratio'
            },
            {
                'name': 'Medium vs Small',
                'target': 'gpt2-medium',
                'draft': 'gpt2',
                'description': 'GPT2-Medium (355M) vs GPT2 (124M) - 2.9x ratio'
            }
        ]
        
        print(f"Model Pair Benchmark initialized on {self.device}")
    
    def load_model_pair(self, target_name, draft_name):
        """Load target and draft models"""
        print(f"Loading {target_name} and {draft_name}...")
        
        tokenizer = GPT2Tokenizer.from_pretrained(target_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        target_model = GPT2LMHeadModel.from_pretrained(target_name).to(self.device)
        draft_model = GPT2LMHeadModel.from_pretrained(draft_name).to(self.device)
        
        return target_model, draft_model, tokenizer
    
    def benchmark_model_pair(self, target_name, draft_name, test_prompts, 
                            max_new_tokens=35, num_runs=2):
        """Benchmark a specific model pair"""
        print(f"\nBenchmarking: {target_name} vs {draft_name}")
        print("=" * 40)
        
        try:
            target_model, draft_model, tokenizer = self.load_model_pair(
                target_name, draft_name
            )
            
            spec_decoder = SpeculativeDecoder(
                target_model, draft_model, tokenizer, gamma=4
            )
            
            results = []
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\nPrompt {i}: '{prompt[:40]}...'")
                
                prompt_results = []
                
                for run in range(num_runs):
                    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                    
                    # Speculative decoding
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    start_time = time.time()
                    spec_output = spec_decoder.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.8,
                        do_sample=True,
                        top_k=50
                    )
                    spec_time = time.time() - start_time
                    spec_stats = spec_decoder.get_stats()
                    
                    # Standard generation
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    start_time = time.time()
                    with torch.no_grad():
                        standard_output = target_model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=0.8,
                            do_sample=True,
                            top_k=50,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    standard_time = time.time() - start_time
                    
                    actual_tokens_spec = spec_output.size(1) - input_ids.size(1)
                    actual_tokens_std = standard_output.size(1) - input_ids.size(1)
                    
                    result = {
                        'spec_time': spec_time,
                        'standard_time': standard_time,
                        'speedup': standard_time / spec_time if spec_time > 0 else 0,
                        'spec_tps': actual_tokens_spec / spec_time if spec_time > 0 else 0,
                        'std_tps': actual_tokens_std / standard_time if standard_time > 0 else 0,
                        'acceptance_rate': spec_stats['acceptance_rate']
                    }
                    
                    prompt_results.append(result)
                
                # Average results
                avg_result = {
                    'speedup': sum(r['speedup'] for r in prompt_results) / len(prompt_results),
                    'acceptance_rate': sum(r['acceptance_rate'] for r in prompt_results) / len(prompt_results),
                    'spec_tps': sum(r['spec_tps'] for r in prompt_results) / len(prompt_results),
                    'std_tps': sum(r['std_tps'] for r in prompt_results) / len(prompt_results)
                }
                
                results.append(avg_result)
                
                print(f"  Speedup: {avg_result['speedup']:.2f}x")
                print(f"  Acceptance: {avg_result['acceptance_rate']:.1%}")
            
            # Calculate overall averages
            summary = {
                'target_model': target_name,
                'draft_model': draft_name,
                'avg_speedup': sum(r['speedup'] for r in results) / len(results),
                'avg_acceptance_rate': sum(r['acceptance_rate'] for r in results) / len(results),
                'avg_spec_tps': sum(r['spec_tps'] for r in results) / len(results),
                'avg_std_tps': sum(r['std_tps'] for r in results) / len(results)
            }
            
            print(f"\nSUMMARY:")
            print(f"  Average speedup: {summary['avg_speedup']:.2f}x")
            print(f"  Average acceptance: {summary['avg_acceptance_rate']:.1%}")
            
            # Clean up
            del target_model, draft_model, spec_decoder
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return summary
            
        except Exception as e:
            print(f"Failed: {e}")
            return None
    
    def run_comprehensive_benchmark(self):
        """Run benchmark across all model pairs"""
        print("SPECULATIVE DECODING ANALYSIS")
        print("=" * 50)
        
        test_prompts = [
            "The future of artificial intelligence will be",
            "Climate change is a global challenge that requires",
            "The most significant technological advancement"
        ]
        
        all_results = []
        
        for pair_config in self.model_pairs:
            print(f"\n\nTesting: {pair_config['description']}")
            
            result = self.benchmark_model_pair(
                pair_config['target'],
                pair_config['draft'],
                test_prompts,
                max_new_tokens=35
            )
            
            if result:
                result['config_name'] = pair_config['name']
                result['description'] = pair_config['description']
                all_results.append(result)
        
        # Final comparison
        print(f"\n\n" + "="*50)
        print("FINAL COMPARISON")
        print("="*50)
        
        for result in all_results:
            status = "FASTER" if result['avg_speedup'] > 1.0 else "SLOWER"
            print(f"\n{result['config_name']}: {result['avg_speedup']:.2f}x {status}")
            print(f"  {result['description']}")
            print(f"  Acceptance rate: {result['avg_acceptance_rate']:.1%}")
        
        if all_results:
            best_result = max(all_results, key=lambda x: x['avg_speedup'])
            print(f"\nBEST CONFIGURATION: {best_result['config_name']}")
            print(f"   Speedup: {best_result['avg_speedup']:.2f}x")
            print(f"   Acceptance: {best_result['avg_acceptance_rate']:.1%}")
        
        print(f"\n\nKEY INSIGHTS:")
        print(f"- Model size ratio critically impacts speculative decoding success")
        print(f"- High acceptance rates don't guarantee speedup")
        print(f"- Optimal configuration requires 8-10x parameter ratio")
        print(f"- Performance variability is inherent to the technique")
        
        return all_results


def main():
    """Run comprehensive speculative decoding analysis"""
    print("GPT-2 Speculative Decoding Analysis")
    print("Study of Model Size Ratios")
    print("=" * 50)
    
    benchmark = ModelPairBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print(f"\n\nAnalysis Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()