import numpy as np

# 配置参数
vocab_size = 50257
context_length_1k = 1024
context_length_16k = 16384
num_layers = 48
d_model =  1600
num_heads = 25
d_ff = 6400  # SwiGLU

def compute_flops_transformer(s, d, d_ff, num_layers, vocab_size):
    """
    计算Transformer forward的FLOPs
    s: sequence length
    d: d_model
    d_ff: FFN intermediate dimension
    """
    # MHA每层FLOPs: 8*s*d^2 + 4*s^2*d
    mha_per_layer = 8 * s * d**2 + 4 * s**2 * d
    
    # FFN (SwiGLU) 每层FLOPs: 6*s*d*d_ff
    ffn_per_layer = 6 * s * d * d_ff
    
    # Transformer主体
    transformer_flops = num_layers * (mha_per_layer + ffn_per_layer)
    
    # LM Head: 2 * s * d * vocab_size
    lm_head_flops = 2 * s * d * vocab_size
    
    # Embedding (position): 2 * s * d (token embedding lookup通常不算FLOPs)
    pos_embed_flops = 2 * s * d
    
    total_flops = transformer_flops + lm_head_flops + pos_embed_flops
    
    return {
        'total': total_flops,
        'transformer': transformer_flops,
        'mha': num_layers * mha_per_layer,
        'ffn': num_layers * ffn_per_layer,
        'lm_head': lm_head_flops,
        'mha_ratio': (num_layers * mha_per_layer) / transformer_flops * 100,
        'ffn_ratio': (num_layers * ffn_per_layer) / transformer_flops * 100
    }

# 计算1k和16k context
flops_1k = compute_flops_transformer(context_length_1k, d_model, d_ff, num_layers, vocab_size)
flops_16k = compute_flops_transformer(context_length_16k, d_model, d_ff, num_layers, vocab_size)

print(flops_1k)
print(flops_16k)

# 比例分析
context_ratio = context_length_16k / context_length_1k
flops_ratio = flops_16k['total'] / flops_1k['total']
transformer_flops_ratio = flops_16k['transformer'] / flops_1k['transformer']

print(f"上下文长度比例: {context_ratio:.1f}x")
print(f"总FLOPs比例: {flops_ratio:.2f}x")
print(f"Transformer主体FLOPs比例: {transformer_flops_ratio:.2f}x")
print()
print("=== 1k context 详细 ===")
print(f"总FLOPs: {flops_1k['total']:.3e}")
print(f"MHA占比: {flops_1k['mha_ratio']:.1f}%")
print(f"FFN占比: {flops_1k['ffn_ratio']:.1f}%")
print(f"LM Head FLOPs: {flops_1k['lm_head']:.3e}")
print()
print("=== 16k context 详细 ===")
print(f"总FLOPs: {flops_16k['total']:.3e}")
print(f"MHA占比: {flops_16k['mha_ratio']:.1f}%")
print(f"FFN占比: {flops_16k['ffn_ratio']:.1f}%")
print(f"LM Head FLOPs: {flops_16k['lm_head']:.3e}")
print()
print("=== 各部分增长倍数 ===")
print(f"MHA FLOPs增长: {flops_16k['mha']/flops_1k['mha']:.2f}x")
print(f"FFN FLOPs增长: {flops_16k['ffn']/flops_1k['ffn']:.2f}x")
print(f"LM Head FLOPs增长: {flops_16k['lm_head']/flops_1k['lm_head']:.2f}x")
