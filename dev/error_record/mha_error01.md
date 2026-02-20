

```
    def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                     mask: torch.Tensor | None=None) -> torch.Tensor:
        d_k = query.shape[-1]

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
>           scores[~mask] = - torch.inf
            ^^^^^^^^^^^^^
E           IndexError: The shape of the mask [12, 12] at index 0 does not match the shape of the indexed tensor [4, 4, 12, 12] a            ^^^^^^^^^^^^^
            ^^^^^^^^^^^^^
E           IndexError: The shape of the mask [12, 12] at index 0 does not match the shape of the indexed tensor [4, 4, 12, 12] at index 0
```

The error you're encountering is due to a shape mismatch between the `mask` and the `scores` tensor in your `scaled_dot_product_attention` function. Specifically:

- `scores` has shape `[4, 4, 12, 12]` (batch_size, num_heads, seq_len, seq_len)
- `mask` has shape `[12, 12]` (seq_len, seq_len)

When you attempt `scores[~mask] = -torch.inf`, PyTorch cannot broadcast the 2D mask across the batch and head dimensions, resulting in an `IndexError`.

### ✅ Solution: Expand the Mask to Match `scores` Shape

You need to expand the mask so it aligns with the first two dimensions of `scores`. Here are two valid approaches:

#### Option 1: Use `unsqueeze` and `expand` (Efficient, No Memory Copy)
```python
if mask is not None:
    # mask: [12, 12] -> [1, 1, 12, 12] -> [4, 4, 12, 12]
    mask = mask.unsqueeze(0).unsqueeze(0).expand_as(scores)
    scores = scores.masked_fill(~mask, -torch.inf)
```

#### Option 2: Use Broadcasting Directly (Concise)
```python
if mask is not None:
    # mask: [12, 12] broadcasts to [4, 4, 12, 12]
    scores = scores.masked_fill(~mask, -torch.inf)
```

> ⚠️ Note: Direct broadcasting (Option 2) works in most cases, but if you're modifying `scores` in-place (e.g., `scores[~mask] = ...`), you must ensure the mask is expanded to match `scores` exactly (Option 1).

