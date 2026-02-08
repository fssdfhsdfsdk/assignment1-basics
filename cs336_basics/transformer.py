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


class FNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    