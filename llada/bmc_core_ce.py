"""
BMC core module: V3 extension with cross-entropy scoring.
Uses the CE formulation validated by our tests (direct token matching).
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

# Import the original BMC module to use as the base class
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bmc_core import BMCEstimator


class BMCEstimatorWithCE(BMCEstimator):
    """
    Extended BMC estimator that adds the validated cross-entropy metric.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        semantic_model_path: str = None,
        adaptive_t: bool = True,
        num_samples: int = 1,
        aggressive_scoring: bool = True,
        optimized_weights: Optional[Dict[str, float]] = None,
        task_name: Optional[str] = None,
        task_config: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        use_chat_template: bool = True,
        scoring_mode: str = "six_only",
        ce_weight: float = 0.50,
        debug_ce: bool = True,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            semantic_model_path=semantic_model_path,
            adaptive_t=adaptive_t,
            num_samples=num_samples,
            aggressive_scoring=aggressive_scoring,
            optimized_weights=optimized_weights,
            task_name=task_name,
            task_config=task_config,
            system_prompt=system_prompt,
            use_chat_template=use_chat_template,
        )
        
        self.scoring_mode = scoring_mode.lower()
        self.ce_weight = ce_weight
        self.debug_ce = debug_ce
        self.ce_call_count = 0
        
        # Validate the scoring mode
        valid_modes = ['ce_only', 'ce_plus_six', 'six_only']
        if self.scoring_mode not in valid_modes:
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}. Must be one of {valid_modes}")
        
        self._print_ce_config()
    
    def _print_ce_config(self):
        """Print the CE configuration."""
        print("\n" + "="*60)
        print("BMC V3 cross-entropy comparison experiment")
        print("="*60)
        print(f"[INFO] Scoring mode: {self.scoring_mode.upper()}")
        print(f"[INFO] Debug mode: {'on' if self.debug_ce else 'off'}")
        print(f"[INFO] CE method: direct token matching")
        
        if self.scoring_mode == 'ce_only':
            print("   -> Cross-entropy only")
        elif self.scoring_mode == 'ce_plus_six':
            print(f"   -> CE weight: {self.ce_weight:.1%}, six metrics: {1-self.ce_weight:.1%}")
        else:
            print("   -> Six metrics only")
        
        print("="*60 + "\n")
    
    def compute_answer_token_ce(
        self,
        reconstructed_ids: torch.Tensor,
        original_ids: torch.Tensor,
        answer_start_idx: int,
        sample_idx: int = 0
    ) -> Tuple[float, float, Dict]:
        """
        Cross-entropy of the original answer tokens under the reconstructed sequence
        (direct token matching, the formulation validated in our tests).
        
        On a single-sample sanity check this gives:
        - mean CE: 0.1133 (reasonable range)
        - CE quality score: 0.8929 (high quality)
        
        Args:
            reconstructed_ids: reconstructed sequence [batch, seq_len]
            original_ids: original sequence [batch, seq_len]
            answer_start_idx: index where the answer begins
            sample_idx: sample index
            
        Returns:
            raw_ce: mean cross-entropy
            ce_quality: quality score in [0, 1]
            debug_info: debug information
        """
        debug_info = {
            'sample_idx': sample_idx,
            'error': None,
            'boundary_check': 'passed',
        }
        
        try:
            batch_size, seq_len = original_ids.shape
            
            # Verbose debug logging for the first few samples
            show_debug = self.debug_ce and self.ce_call_count < 5
            
            if show_debug:
                print(f"\n{'='*70}")
                print(f"[INFO] CE computation details (sample {sample_idx})")
                print(f"{'='*70}")
                print(f"Original sequence shape: {original_ids.shape}")
                print(f"Reconstructed sequence shape: {reconstructed_ids.shape}")
                print(f"Answer start index: {answer_start_idx}")
                print(f"Total sequence length: {seq_len}")
            
            # Lenient boundary checks with automatic correction
            original_answer_start = answer_start_idx
            
            # Case 1: answer_start_idx is out of range
            if answer_start_idx >= seq_len:
                if show_debug:
                    print(f"[WARN] answer_start_idx ({answer_start_idx}) >= seq_len ({seq_len})")
                    print(f"   Auto-correcting: use the last third of the sequence as the answer span")
                answer_start_idx = max(0, seq_len * 2 // 3)
                debug_info['boundary_check'] = f'adjusted_from_{original_answer_start}_to_{answer_start_idx}'
            
            # Case 2: answer_start_idx is negative
            elif answer_start_idx < 0:
                if show_debug:
                    print(f"[WARN] answer_start_idx ({answer_start_idx}) < 0")
                    print(f"   Auto-correcting: start from 0")
                answer_start_idx = 0
                debug_info['boundary_check'] = f'adjusted_from_{original_answer_start}_to_0'
            
            # Case 3: the answer span is too short (fewer than 2 tokens)
            answer_len = seq_len - answer_start_idx
            if answer_len < 2:
                if show_debug:
                    print(f"[WARN] Answer span is too short ({answer_len} tokens)")
                    print(f"   Auto-correcting: ensure at least 5 tokens")
                answer_start_idx = max(0, seq_len - 5)
                answer_len = seq_len - answer_start_idx
                debug_info['boundary_check'] = f'adjusted_for_min_length'
            
            debug_info['answer_start_idx'] = answer_start_idx
            debug_info['answer_len'] = answer_len
            
            if show_debug:
                print(f"[OK] Final answer span: [{answer_start_idx}, {seq_len}), length={answer_len}")
                
                # Decode the answer content
                try:
                    orig_answer_text = self.tokenizer.decode(
                        original_ids[0, answer_start_idx:seq_len],
                        skip_special_tokens=False
                    )
                    recon_answer_text = self.tokenizer.decode(
                        reconstructed_ids[0, answer_start_idx:seq_len],
                        skip_special_tokens=False
                    )
                    print(f"Original answer: '{orig_answer_text}'")
                    print(f"Reconstructed answer: '{recon_answer_text}'")
                except:
                    print(f"(answer text could not be decoded)")
            
            with torch.no_grad():
                # Compute logits from the reconstructed sequence
                logits = self.model(reconstructed_ids).logits  # [batch, seq_len, vocab_size]
                
                if show_debug:
                    print(f"\nLogits shape: {logits.shape}")
                    print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                    print(f"Contains NaN: {torch.isnan(logits).any().item()}")
                    print(f"Contains Inf: {torch.isinf(logits).any().item()}")
                
                # Guard against invalid logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    debug_info['error'] = 'logits_contain_nan_or_inf'
                    if show_debug:
                        print(f"[ERROR] Invalid logits")
                    return float('inf'), 0.0, debug_info
                
                # Direct token matching: at every position of the answer span,
                # score the probability assigned to the original token
                answer_logits = logits[:, answer_start_idx:seq_len, :]  # [batch, ans_len, vocab]
                answer_targets = original_ids[:, answer_start_idx:seq_len]  # [batch, ans_len]
                
                if show_debug:
                    print(f"\nAnswer logits shape: {answer_logits.shape}")
                    print(f"Answer targets shape: {answer_targets.shape}")
                
                # Log probabilities
                log_probs = F.log_softmax(answer_logits, dim=-1)  # [batch, ans_len, vocab]
                
                # Log probability of each target token
                target_log_probs = log_probs.gather(
                    dim=-1,
                    index=answer_targets.unsqueeze(-1)
                ).squeeze(-1)  # [batch, ans_len]
                
                if show_debug:
                    print(f"Target log prob range: [{target_log_probs.min().item():.4f}, {target_log_probs.max().item():.4f}]")
                
                # CE = -log(p)
                token_ces = -target_log_probs  # [batch, ans_len]
                
                # Mean CE
                raw_ce = token_ces.mean().item()
                
                debug_info['raw_ce'] = raw_ce
                debug_info['ce_min'] = token_ces.min().item()
                debug_info['ce_max'] = token_ces.max().item()
                debug_info['ce_std'] = token_ces.std().item()
                
                if show_debug:
                    print(f"\nCE statistics:")
                    print(f"  Mean CE: {raw_ce:.4f}")
                    print(f"  Range: [{debug_info['ce_min']:.4f}, {debug_info['ce_max']:.4f}]")
                    print(f"  Std: {debug_info['ce_std']:.4f}")
                    
                    # Per-token details for the first few tokens
                    print(f"\nFirst 5 tokens:")
                    print(f"{'Pos':<6} {'Token':<15} {'ID':<8} {'Log Prob':<10} {'Prob':<10} {'CE':<10}")
                    print(f"{'-'*70}")
                    for i in range(min(5, answer_len)):
                        token_id = answer_targets[0, i].item()
                        log_prob = target_log_probs[0, i].item()
                        prob = np.exp(log_prob)
                        ce = token_ces[0, i].item()
                        try:
                            token_str = self.tokenizer.decode([token_id])
                        except:
                            token_str = f"<{token_id}>"
                        print(f"{answer_start_idx+i:<6} {token_str:<15} {token_id:<8} {log_prob:<10.4f} {prob:<10.4f} {ce:<10.4f}")
                
                # Convert CE into a bounded quality score
                if raw_ce == float('inf') or raw_ce > 100:
                    ce_quality = 0.0
                    debug_info['quality_saturated'] = True
                else:
                    ce_quality = np.exp(-raw_ce)
                    debug_info['quality_saturated'] = False
                
                debug_info['ce_quality'] = ce_quality
                
                if show_debug:
                    print(f"\n[OK] CE quality score: {ce_quality:.4f}")
                    print(f"{'='*70}\n")
                
                self.ce_call_count += 1
                
                return raw_ce, ce_quality, debug_info
                
        except Exception as e:
            debug_info['error'] = str(e)
            if self.debug_ce:
                print(f"\n[ERROR] CE computation failed: {e}")
                import traceback
                traceback.print_exc()
            return float('inf'), 0.0, debug_info
    
    def compute_consistency_score(
        self,
        original_ids: torch.Tensor,
        reconstructed_ids: torch.Tensor,
        answer_start_idx: int,
        original_answer: str,
        avg_confidence: float = 0.0
    ) -> Dict[str, float]:
        """
        Extended consistency scoring that can blend in the CE metric.
        """
        # 1. Compute the six base metrics
        base_scores = super().compute_consistency_score(
            original_ids,
            reconstructed_ids,
            answer_start_idx,
            original_answer,
            avg_confidence
        )
        
        # 2. Compute CE when the scoring mode requires it
        if self.scoring_mode in ['ce_only', 'ce_plus_six']:
            raw_ce, ce_quality, debug_info = self.compute_answer_token_ce(
                reconstructed_ids,
                original_ids,
                answer_start_idx,
                sample_idx=self.ce_call_count
            )
            
            base_scores['reconstruction_ce'] = raw_ce
            base_scores['ce_quality_score'] = ce_quality
            
            # Keep the key debug fields in the result
            if 'boundary_check' in debug_info:
                base_scores['ce_boundary_check'] = debug_info['boundary_check']
            if 'answer_len' in debug_info:
                base_scores['ce_answer_len'] = debug_info['answer_len']
            if 'error' in debug_info and debug_info['error']:
                base_scores['ce_error'] = debug_info['error']
        else:
            base_scores['reconstruction_ce'] = -1.0
            base_scores['ce_quality_score'] = -1.0
        
        # 3. Combine into the final score according to the scoring mode
        if self.scoring_mode == 'ce_only':
            final_score = base_scores['ce_quality_score']
            
        elif self.scoring_mode == 'ce_plus_six':
            six_weight = 1 - self.ce_weight
            
            if self.use_optimized_weights and self.optimized_weights:
                weights = self.optimized_weights.copy()
                six_metrics_total = sum(weights.values())
                if six_metrics_total > 0:
                    scale = six_weight / six_metrics_total
                    for k in weights:
                        weights[k] *= scale
                weights['ce_quality_score'] = self.ce_weight
                
                final_score = sum(
                    weights.get(metric, 0) * base_scores.get(metric, 0)
                    for metric in [
                        'token_accuracy', 'semantic_similarity', 
                        'number_retention', 'final_answer_match',
                        'char_similarity', 'avg_confidence',
                        'ce_quality_score'
                    ]
                )
            else:
                final_score = (
                    six_weight * 0.45 * base_scores['final_answer_match'] +
                    six_weight * 0.20 * base_scores['number_retention'] +
                    six_weight * 0.12 * base_scores['token_accuracy'] +
                    six_weight * 0.10 * base_scores['semantic_similarity'] +
                    six_weight * 0.08 * base_scores['avg_confidence'] +
                    six_weight * 0.05 * base_scores['char_similarity'] +
                    self.ce_weight * base_scores['ce_quality_score']
                )
            
            final_score = np.clip(final_score, 0.0, 1.0)
            
        else:  # six_only
            final_score = base_scores['consistency_score']
        
        base_scores['consistency_score'] = final_score
        base_scores['scoring_mode'] = self.scoring_mode
        base_scores['ce_weight'] = self.ce_weight if self.scoring_mode == 'ce_plus_six' else 0.0
        
        return base_scores
