#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer FLOPs Calculator for One AdamW Training Step
Computes: Total FLOPs = Forward + Backward + Optimizer Update

Key assumptions:
- MatMul FLOPs: 2 * M * N * K for (M,N) @ (N,K)
- Backward pass ≈ 2× Forward pass FLOPs (for MatMul operations)
- AdamW: ~8 FLOPs per parameter (momentum + variance + update)
- SwiGLU FFN: 3 matrix multiplications per layer
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer model hyperparameters"""
    vocab_size: int = 50257
    context_length: int = 1024
    num_layers: int = 48
    d_model: int = 1600
    num_heads: int = 25
    d_ff: Optional[int] = None  # Default: 4 * d_model
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


@dataclass
class ComputeConfig:
    """FLOPs calculation assumptions"""
    # FLOPs per operation type
    matmul_flops_factor: float = 2.0  # 2*M*N*K for standard matmul
    
    # Backward pass multiplier (for MatMul operations)
    backward_multiplier: float = 2.0
    
    # AdamW optimizer FLOPs per parameter
    # Includes: read states, compute m/v, bias correction, weight update, write
    adamw_flops_per_param: float = 8.0
    
    # Include non-matmul operations estimates
    include_softmax: bool = True
    include_layernorm: bool = True
    include_elementwise: bool = True


class TransformerFLOPsCalculator:
    """Calculate FLOPs for Transformer training step with AdamW"""
    
    def __init__(self, model: ModelConfig, compute: ComputeConfig = None):
        self.cfg = model
        self.comp = compute or ComputeConfig()
        
        # Unpack for convenience
        self.B = None  # batch_size (symbolic)
        self.S = model.context_length
        self.D = model.d_model
        self.H = model.num_heads
        self.L = model.num_layers
        self.V = model.vocab_size
        self.Dff = model.d_ff
        self.Dk = self.D // self.H  # dimension per head
        
    def _matmul_flops(self, m: int, n: int, k: int) -> float:
        """FLOPs for matrix multiplication (M,N) @ (N,K)"""
        return self.comp.matmul_flops_factor * m * n * k
    
    def compute_forward_flops(self, batch_size: int) -> dict:
        """Compute forward pass FLOPs breakdown"""
        B, S, D, H, L, V, Dff, Dk = self.B, self.S, self.D, self.H, self.L, self.V, self.Dff, self.Dk
        B = batch_size  # Override symbolic B
        
        result = {}
        
        # ========== Embedding Layer ==========
        # Lookup: typically not counted as FLOPs, but backward has gradient scatter
        result['embedding'] = 0  # Forward is memory lookup
        
        # ========== Per Transformer Layer ==========
        layer_flops = {}
        
        # --- MHA Sublayer ---
        mha = {}
        
        # QKV projections: 3 × [B×S×D] @ [D×D] → [B×S×D]
        qkv_proj = 3 * self._matmul_flops(B*S, D, D)
        mha['qkv_projections'] = qkv_proj
        
        # QK^T for attention scores: [B×H×S×Dk] @ [B×H×Dk×S] → [B×H×S×S]
        # Reshaped as: H batches of [S×Dk] @ [Dk×S]
        qk_scores = H * self._matmul_flops(S, Dk, S)
        mha['qk_scores'] = qk_scores
        
        # Softmax: element-wise, ~4 FLOPs per element (exp, sum, divide)
        softmax_flops = 4 * B * H * S * S if self.comp.include_softmax else 0
        mha['softmax'] = softmax_flops
        
        # Attention weights × V: [B×H×S×S] @ [B×H×S×Dk] → [B×H×S×Dk]
        attn_v = H * self._matmul_flops(S, S, Dk)
        mha['attn_value'] = attn_v
        
        # Output projection: [B×S×D] @ [D×D] → [B×S×D]
        out_proj = self._matmul_flops(B*S, D, D)
        mha['output_projection'] = out_proj
        
        layer_flops['mha'] = sum(mha.values())
        result['mha_per_layer'] = mha
        
        # --- FFN Sublayer (SwiGLU) ---
        ffn = {}
        
        # Gate projection: [B×S×D] @ [D×Dff] → [B×S×Dff]
        gate_proj = self._matmul_flops(B*S, D, Dff)
        ffn['gate_projection'] = gate_proj
        
        # Up projection: [B×S×D] @ [D×Dff] → [B×S×Dff]
        up_proj = self._matmul_flops(B*S, D, Dff)
        ffn['up_projection'] = up_proj
        
        # SiLU activation: ~3 FLOPs per element (x * sigmoid(x))
        silu_flops = 3 * B * S * Dff if self.comp.include_elementwise else 0
        ffn['silu_activation'] = silu_flops
        
        # Element-wise multiply (gate * up): 1 FLOP per element
        gate_mult = 1 * B * S * Dff if self.comp.include_elementwise else 0
        ffn['gate_multiply'] = gate_mult
        
        # Down projection: [B×S×Dff] @ [Dff×D] → [B×S×D]
        down_proj = self._matmul_flops(B*S, Dff, D)
        ffn['down_projection'] = down_proj
        
        layer_flops['ffn'] = sum(ffn.values())
        result['ffn_per_layer'] = ffn
        
        # --- LayerNorm (2 per layer: pre-attention, pre-ffn) ---
        # Mean + var + normalize: ~5 FLOPs per element
        ln_flops = 2 * 5 * B * S * D if self.comp.include_layernorm else 0
        layer_flops['layernorm'] = ln_flops
        
        # Total per layer
        layer_total = sum(layer_flops.values())
        result['per_layer_total'] = layer_total
        
        # ========== All Layers ==========
        result['all_layers'] = layer_total * L
        
        # ========== Output Head ==========
        # Final LayerNorm
        final_ln = 5 * B * S * D if self.comp.include_layernorm else 0
        
        # Logits projection: [B×S×D] @ [D×V] → [B×S×V]
        logits_proj = self._matmul_flops(B*S, D, V)
        
        # Cross-entropy: ~5 FLOPs per logit (log, softmax, nll)
        ce_flops = 5 * B * S * V if self.comp.include_elementwise else 0
        
        result['output_head'] = final_ln + logits_proj + ce_flops
        
        # ========== Total Forward ==========
        result['total_forward'] = result['all_layers'] + result['output_head']
        
        return result
    
    def compute_parameter_count(self) -> dict:
        """Count model parameters (for optimizer FLOPs)"""
        S, D, H, L, V, Dff = self.S, self.D, self.H, self.L, self.V, self.Dff
        
        params = {}
        params['token_embedding'] = V * D
        params['position_embedding'] = 0  # Usually learned or rotary, negligible
        params['rmsnorm_scales'] = (L + 1) * D  # One per norm, L layers + final
        params['mha_projections'] = 4 * L * D * D  # Q, K, V, O
        params['ffn_swiglu'] = 3 * L * D * Dff  # Gate, Up, Down
        params['unembedding'] = V * D  # Output projection (tied or separate)
        
        params['total'] = sum(params.values())
        return params
    
    def compute_total_training_flops(self, batch_size: int) -> dict:
        """Compute complete FLOPs for one AdamW training step"""
        
        # 1. Forward pass
        fwd = self.compute_forward_flops(batch_size)
        
        # 2. Backward pass (~2× forward for MatMul-dominated ops)
        # Note: Some ops have different backward costs, but 2× is good estimate
        bwd_multiplier = 1 + self.comp.backward_multiplier  # 1 (fwd) + 2 (bwd) = 3
        flops_compute = fwd['total_forward'] * bwd_multiplier
        
        # 3. Optimizer step (AdamW)
        params = self.compute_parameter_count()
        flops_optimizer = params['total'] * self.comp.adamw_flops_per_param
        
        # 4. Total
        total_flops = flops_compute + flops_optimizer
        
        return {
            'forward': fwd['total_forward'],
            'backward': fwd['total_forward'] * self.comp.backward_multiplier,
            'optimizer': flops_optimizer,
            'total': total_flops,
            'breakdown': fwd,
            'params': params,
            'config': {
                'B': batch_size, 'S': self.S, 'D': self.D, 'L': self.L,
                'V': self.V, 'H': self.H, 'Dff': self.Dff
            }
        }
    
    def get_algebraic_expression(self) -> str:
        """Return symbolic algebraic expression for FLOPs"""
        # Using symbolic variables: B=batch_size, S=context_length, etc.
        
        # Forward FLOPs per layer (dominant terms only)
        # FFN (SwiGLU): 3 matmuls × 2 = 6*B*S*D*Dff
        ffn_fwd = f"6·B·S·D·Dff"
        
        # MHA: projections + attention
        # Projections: 4 × 2*B*S*D² = 8*B*S*D²
        # Attention (QK^T + PV): 2 × 2*B*S²*D = 4*B*S²*D
        mha_fwd = f"8·B·S·D² + 4·B·S²·D"
        
        # With Dff = 4D:
        # FFN: 24·B·S·D²
        # MHA: 8·B·S·D² + 4·B·S²·D
        # Total per layer: 32·B·S·D² + 4·B·S²·D = 4·B·S·D·(8D + S)
        
        expr = f"""
Forward Pass (per layer, dominant terms):
  FFN (SwiGLU):     {ffn_fwd}
  MHA:              {mha_fwd}
  
With Dff = 4·D, per layer:
  Forward_layer ≈ 4·B·S·D·(8·D + S)

Total Forward (L layers):
  Forward_total ≈ L · 4·B·S·D·(8·D + S)

Training Step (Forward + Backward + Optimizer):
  Compute (3× Forward):  ≈ 3 · L · 4·B·S·D·(8·D + S) = 12·L·B·S·D·(8·D + S)
  AdamW Update:          ≈ 8 · (12·L·D² + V·D)  [~8 FLOPs/param]
  
TOTAL FLOPs ≈ 12·L·B·S·D·(8·D + S) + 96·L·D² + 8·V·D
        """.strip()
        return expr


def format_flops(flops: float) -> str:
    """Format FLOPs with appropriate units"""
    units = [(1e18, 'EFLOPs'), (1e15, 'PFLOPs'), (1e12, 'TFLOPs'), 
             (1e9, 'GFLOPs'), (1e6, 'MFLOPs'), (1e3, 'KFLOPs'), (1, 'FLOPs')]
    for divisor, unit in units:
        if flops >= divisor:
            return f"{flops/divisor:.3f} {unit}"
    return f"{flops:.0f} FLOPs"


def print_analysis(results: dict, calculator: TransformerFLOPsCalculator):
    """Print comprehensive FLOPs analysis"""
    cfg = results['config']
    
    print(f"\n🧮 Transformer FLOPs Analysis (One AdamW Training Step)")
    print(f"   Config: L={cfg['L']}, D={cfg['D']}, S={cfg['S']}, V={cfg['V']:,}, H={cfg['H']}")
    print(f"   Batch size: {cfg['B']}, Dff={cfg['Dff']}")
    
    print(f"\n📊 FLOPs Breakdown:")
    print(f"   Forward Pass:     {format_flops(results['forward'])}")
    print(f"   Backward Pass:    {format_flops(results['backward'])}")
    print(f"   AdamW Optimizer:  {format_flops(results['optimizer'])}")
    print(f"   ─────────────────────────────")
    print(f"   TOTAL:            {format_flops(results['total'])}")
    
    print(f"\n📈 Parameter Count: {results['params']['total']:,} ({format_flops(results['params']['total'])} elements)")
    
    # Tokens processed
    tokens = cfg['B'] * cfg['S']
    flops_per_token = results['total'] / tokens if tokens > 0 else 0
    print(f"\n⚡ Efficiency: {format_flops(flops_per_token)} / token")
    
    # Compare to rule of thumb: 6 * params * tokens
    rule_of_thumb = 6 * results['params']['total'] * tokens
    ratio = results['total'] / rule_of_thumb if rule_of_thumb > 0 else 0
    print(f"\n🎯 vs Rule-of-Thumb (6·params·tokens):")
    print(f"   Expected: {format_flops(rule_of_thumb)}")
    print(f"   Computed: {format_flops(results['total'])}")
    print(f"   Ratio:    {ratio:.2f}× {'(higher due to S² attention)' if ratio > 1.1 else '(matches)'}")


def main():
    """Run analysis for GPT-2 XL with different context lengths"""
    
    # ===== CONFIGURATION =====
    model_cfg = ModelConfig(
        vocab_size=50257,
        context_length=1024,  # Will be overridden below
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400  # SwiGLU: 4 × d_model
    )
    
    compute_cfg = ComputeConfig(
        include_softmax=True,
        include_layernorm=True,
        include_elementwise=True,
        adamw_flops_per_param=8.0
    )
    
    batch_size = 1024  # Symbolic: results scale linearly with B
        
    print(f"\n" + "="*70)
    print("NUMERICAL RESULTS")
    print("="*70)
    
    # Analyze for different context lengths
    for ctx_name, ctx_len in [("1K", 1024)]:
        model_cfg.context_length = ctx_len
        calculator = TransformerFLOPsCalculator(model_cfg, compute_cfg)
        
        results = calculator.compute_total_training_flops(batch_size)
        
        print(f"\n🔹 Context Length: {ctx_name} ({ctx_len:,} tokens)")
        print(f"   Expression: a·B + b  (where B = batch_size)")
        
        # Extract coefficients for a·B + b form
        # Forward scales with B, optimizer is constant
        fwd_per_batch = results['forward'] / batch_size
        bwd_per_batch = results['backward'] / batch_size
        compute_per_batch = fwd_per_batch + bwd_per_batch
        optimizer_fixed = results['optimizer']
        
        a = compute_per_batch  # FLOPs per sample (compute part)
        b = optimizer_fixed     # Fixed optimizer FLOPs
        
        print(f"   a (per-sample compute): {format_flops(a)}")
        print(f"   b (optimizer overhead): {format_flops(b)}")
        print(f"   Total for B=1: {format_flops(a + b)}")
        
        # Show for typical batch sizes
        for test_b in [1, 2, 4, 8]:
            total_b = a * test_b + b
            print(f"   B={test_b:2d}: {format_flops(total_b)}")
        
        print_analysis(results, calculator)
    
    # ===== ALGEBRAIC EXPRESSION =====
    print(f"\n" + "="*70)
    print("📦 DELIVERABLE: Algebraic Expression")
    print("="*70)
    calculator = TransformerFLOPsCalculator(model_cfg, compute_cfg)
    print(calculator.get_algebraic_expression())
    
    print(f"\n💡 Key Insights:")
    print(f"   • Training FLOPs ≈ 3× Forward (1 fwd + 2 bwd for MatMul)")
    print(f"   • Attention has O(S²) term: dominates for long sequences")
    print(f"   • AdamW overhead is O(params), independent of batch/seq length")
    print(f"   • For S=1024: compute term dominates; for S=16K: attention S² term dominates")


if __name__ == "__main__":
    main()