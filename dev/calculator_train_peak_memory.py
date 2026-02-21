#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 XL Peak Memory Calculator for AdamW Training
Computes: total_memory = a * batch_size + b
Returns max batch_size that fits in specified GPU memory
"""

import math

# 配置参数
vocab_size = 50257
context_length_1k = 1024
context_length_16k = 16384
num_layers = 48
d_model =  1600
num_heads = 25
d_ff = 6400  # SwiGLU


def compute_transformer_memory(
    # Model hyperparameters
    vocab_size=vocab_size,
    context_length=context_length_1k,
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,  # Default: 4 * d_model
    # System constraints
    memory_limit_gb=80,
    # Precision assumptions (bytes per element)
    bytes_param=2,      # FP16 for parameters
    bytes_grad=2,       # FP16 for gradients  
    bytes_opt=4,        # FP32 for AdamW states
    bytes_act=2,        # FP16 for activations
    opt_states_per_param=2,  # AdamW: momentum + variance
    # Activation saving policy (conservative = save more)
    save_attn_scores=True,   # Save QK^T before softmax?
    save_attn_probs=True,    # Save attention weights after softmax?
    save_swiglu_both=True,   # Save both gate+up projections for SwiGLU?
):
    """
    Compute peak training memory for Transformer model with AdamW.
    
    Returns:
        a: coefficient for batch_size (bytes per sample)
        b: fixed memory overhead (bytes)  
        max_batch_size: largest integer batch_size fitting in memory_limit_gb
        details: comprehensive breakdown dict
    """
    if d_ff is None:
        d_ff = 4 * d_model
    
    S, D, H, L, V, Dff = context_length, d_model, num_heads, num_layers, vocab_size, d_ff
    
    # ========== 1. PARAMETER COUNTS (elements) ==========
    embed_params = V * D                                    # Token embeddings
    rmsnorm_params = (L + 1) * D                            # One scale vector per norm
    mha_params = 4 * D * D * L                              # Q, K, V, O projections
    ffn_params = 3 * D * Dff * L                            # SwiGLU: gate+up+down
    
    total_params = embed_params + rmsnorm_params + mha_params + ffn_params
    
    # ========== 2. FIXED MEMORY (batch_size-independent) ==========
    mem_params = total_params * bytes_param                 # Model weights
    mem_grads = total_params * bytes_grad                   # .grad buffers
    mem_opt = total_params * opt_states_per_param * bytes_opt  # Adam states
    
    fixed_memory = mem_params + mem_grads + mem_opt
    
    # ========== 3. ACTIVATIONS PER SAMPLE (elements) ==========
    act = 0
    
    # Input embeddings: (S, D)
    act += S * D
    
    # Per transformer layer:
    for _ in range(L):
        # Pre-norm input: (S, D)
        act += S * D
        
        # QKV projection outputs: 3 × (S, D)
        act += 3 * S * D
        
        # Attention mechanism:
        if save_attn_scores:
            act += H * S * S          # QK^T scores before softmax
        if save_attn_probs:
            act += H * S * S          # Attention weights after softmax
        
        # Attention output: (S, D)
        act += S * D
        
        # FFN branch:
        act += S * D                  # FFN input (post-residual)
        if save_swiglu_both:
            act += 2 * S * Dff        # SwiGLU: save gate AND up projections
        else:
            act += S * Dff            # Standard GeLU: save one projection
    
    # Final components:
    act += S * D                      # Final RMSNorm input
    act += S * V                      # Logits before cross-entropy
    
    mem_act_per_sample = act * bytes_act
    
    # ========== 4. MEMORY FORMULA & MAX BATCH SIZE ==========
    a = mem_act_per_sample  # bytes per sample
    b = fixed_memory        # fixed overhead
    
    limit_bytes = memory_limit_gb * (1024 ** 3)
    max_bs = max(0, int((limit_bytes - b) // a)) if a > 0 else 0
    
    # ========== 5. RETURN STRUCTURED RESULTS ==========
    return a, b, max_bs, {
        'formula': {
            'a_gb': a / (1024**3),
            'b_gb': b / (1024**3),
            'expression': f"{a/(1024**3):.2f}·B + {b/(1024**3):.2f} GB"
        },
        'breakdown': {
            'params_elements': total_params,
            'params_gb': total_params * bytes_param / (1024**3),
            'fixed_gb': fixed_memory / (1024**3),
            'act_per_sample_gb': mem_act_per_sample / (1024**3),
        },
        'config': {'V':V, 'S':S, 'L':L, 'D':D, 'H':H, 'Dff':Dff}
    }


def print_summary(a, b, max_bs, results, limit_gb=80):
    """Print human-readable summary"""
    f = results['formula']
    bd = results['breakdown']
    cfg = results['config']
    
    print(f"\n📊 GPT-2 XL Memory Analysis (AdamW, FP16 training)")
    print(f"   Config: L={cfg['L']}, D={cfg['D']}, S={cfg['S']}, V={cfg['V']:,}")
    print(f"\n💾 Memory Formula: {f['expression']}")
    print(f"   • Fixed overhead: {f['b_gb']:.2f} GB")
    print(f"   • Per-sample activation: {f['a_gb']:.2f} GB")
    print(f"\n🎯 Max batch_size for {limit_gb}GB GPU: {max_bs}")
    
    # Verification
    used_gb = (a * max_bs + b) / (1024**3)
    print(f"   → Memory used: {used_gb:.2f} GB / {limit_gb} GB")
    
    if max_bs < 10:
        print(f"\n⚠️  Small batch size! Consider:")
        print(f"   • Gradient accumulation")
        print(f"   • Activation checkpointing (saves ~60-80% activation memory)")
        print(f"   • Mixed precision optimization")


# ========== EXECUTION ==========
if __name__ == "__main__":
    # GPT-2 XL hyperparameters
    a, b, max_bs, res = compute_transformer_memory(
        vocab_size=50257,
        context_length=1024, 
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        memory_limit_gb=80
    )
    
    print_summary(a, b, max_bs, res)
    
    # 📦 Deliverable output
    print(f"\n📦 DELIVERABLE:")
    print(f"   Expression: {res['formula']['a_gb']:.2f} · batch_size + {res['formula']['b_gb']:.2f} GB")
    print(f"   Max batch_size: {max_bs}")