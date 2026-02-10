import torch, torch.nn as nn
from einops import rearrange, einsum
import math

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        wt = torch.empty((in_features, out_features), device=device, dtype=dtype)
        theta = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(wt, std=theta, a=-3*theta, b=3*theta)
        self.weights = nn.Parameter(wt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(self.weights.shape)
        print(x.shape)
        # return x @ self.weights
        return einsum(x, self.weights, "... d_in,  d_in d_out -> ... d_out")
    

class Eebedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        wt = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        std = torch.sqrt(torch.tensor(2/(num_embeddings+embedding_dim))).item()
        trunc = 3 * std
        nn.init.trunc_normal_(wt, std=std, a=-trunc, b=trunc)
        self.embed_matrix = nn.Parameter(wt)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        # Note: 看见一个复杂的写法, 先将 token_ids reshape 为一维, 再 index_select, 再将结果reshape回去
        # index_select的 index 只能接收 1维 的索引
        # res = torch.index_select(self.embed_matrix, 0, token_ids)
        # 实际简写: 直接使用 高级索引 - 原理见 `test_torch.ipynb``
        return self.embed_matrix[token_ids]

class RMSNorm(nn.Module):
    
    def __init__(self, d_model:int, eps:float = 1e-5, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.eps = eps
        # g = torch.empty(d_model, device=device, dtype=dtype)
        # g = 1 # ⚠️ 灾难性错误！g 从张量变成整数 1
        # 其它方法 g_fill_ , torch.ones
        g = torch.full((d_model,), 1.0, device=device, dtype=dtype)

        self.weights = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_a = torch.sqrt(
            torch.sum(torch.square(x), dim=-1, keepdim=True) / self.d_model + self.eps
        )

        res = (x / rms_a) * self.weights

        print(x.shape)
        print(res.shape)

        return res.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def swiglu(w1_weight, w2_weight, w3_weight, x):
    w1 = einsum(w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
    w3 = einsum(w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
    w2 = einsum(w2_weight, silu(w1) * w3, "d_model d_ff, ... d_ff -> ... d_model")
    return w2

class FNN(nn.Module):

    def __init__(self, d_model:int, 
                 device: torch.device | None = None,
                 dtype: torch.device | None = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.d_ff = int((8/3) * d_model)

        self.w1 = Linear(d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, self.d_ff, device, dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2.forward(silu(self.w1.forward(x)) *self.w3.forward(x))


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        invert = theta ** (-torch.arange(0, d_k/2 , 1)/(d_k/2))
        self.register_buffer("invert", invert, persistent=False)

    def _half_rotate(self, x:torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        x3 = torch.stack((-x2, x1), dim=-1)
        return rearrange(x3, "... d j -> ... (d j)")

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        
        theta = einsum(token_positions, self.invert, "... i, j -> ... i j")
        cos = torch.cos(theta).repeat_interleave(repeats=2, dim=-1)
        sin = torch.sin(theta).repeat_interleave(repeats=2, dim=-1)

        return x * cos + self._half_rotate(x) * sin
    

def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    max_of_dim = torch.max(x, dim=dim, keepdim=True).values
    exp_res = torch.exp(x - max_of_dim)
    sum_res = torch.sum(exp_res, dim=dim, keepdim=True)
    return exp_res / sum_res

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 mask: torch.Tensor | None=None) -> torch.Tensor:
    d_k = query.shape[-1]

    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores[~mask] = - torch.inf

    atten_weights = softmax(scores, dim=-1)

    return atten_weights @ value

class MHA(nn.Module):

    def __init__(self, d_model:int, num_heads:int, max_seq_len:int = 2048,
                theta:float = 10000.0,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,
                  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.dk, self.dv =  d_model / num_heads

        self.rope = RotaryPositionalEmbedding(theta, self.dk, max_seq_len)

        

    def _create_causal_mask(self, seq_len:int, device: torch.device) -> torch.Tensor:

        masks = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return masks.unsqueeze(0).unsqueeze(0)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass