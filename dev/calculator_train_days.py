#!/usr/bin/env python3
"""Verify training time calculation for GPT-2 XL on single A100"""

# ===== Configuration =====
# Model hyperparameters (GPT-2 XL + SwiGLU)
V, S, L, D, H, Dff = 50257, 1024, 48, 1600, 25, 6400
batch_size = 1024
steps = 400_000

# Hardware specs
A100_peak_tflops = 19.5  # FP32 TFLOPs/s
mfu = 0.5  # 50% model FLOPs utilization

# ===== Parameter Count (from part a/c) =====
embed_params = V * D
rmsnorm_params = (L + 1) * D
mha_params = 4 * L * D * D
ffn_params = 3 * L * D * Dff  # SwiGLU: gate+up+down
unembed_params = V * D  # Assuming untied; set to 0 if tied

total_params = embed_params + rmsnorm_params + mha_params + ffn_params + unembed_params
print(f"📦 Total parameters: {total_params/1e9:.2f}B")

# ===== FLOPs per Training Step =====
# Using Kaplan et al. rule: ~6 FLOPs per parameter per token
# This accounts for: 1× forward + 2× backward (MatMul gradient rule)
tokens_per_step = batch_size * S
flops_per_step = 6 * total_params * tokens_per_step

print(f"🔢 Tokens per step: {tokens_per_step:,}")
print(f"⚡ FLOPs per step: {flops_per_step/1e15:.3f} PFLOPs")

# ===== Total Training FLOPs =====
total_flops = flops_per_step * steps
print(f"🎯 Total training FLOPs: {total_flops/1e18:.2f} EFLOPs")

# ===== Effective Throughput =====
effective_tflops = A100_peak_tflops * mfu
print(f"🚀 Effective throughput: {effective_tflops:.2f} TFLOPs/s")

# ===== Training Time =====
seconds = total_flops / (effective_tflops * 1e12)
hours = seconds / 3600
days = hours / 24
years = days / 365.25

print(f"\n⏱️  Training Time:")
print(f"   {seconds/1e6:.1f} million seconds")
print(f"   {hours/1e3:.1f} thousand hours")
print(f"   {days:.0f} days ≈ {years:.1f} years")

# ===== Scaling Insight =====
print(f"\n💡 Distributed Training Scaling (ideal linear):")
for gpus in [8, 64, 256, 1024]:
    scaled_days = days / gpus
    print(f"   {gpus:4d} A100s: {scaled_days:6.1f} days ≈ {scaled_days/30:.1f} months")