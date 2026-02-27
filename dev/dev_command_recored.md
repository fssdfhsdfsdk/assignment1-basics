


```
uv run pytest 
uv run pytest tests/test_train_bpe.py
  pytest tests/test_train_bpe.py::test_train_bpe
uv run pytest tests/test_tokenizer.py
  python -m pstats profile_output.prof
    - `sort cumtime`：按**累积时间**排序（最常用，能看到包含子函数在内的总耗时）。
    - `stats 10`：列出耗时前 10 名的函数。

===
uv run pytest tests/test_model.py::test_linear
uv run pytest tests/test_model.py::test_embedding
  
pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_softmax_matches_pytorch
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
  uv run pytest tests/test_model.py::test_multihead_self_attention
  uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
  uv run pytest tests/test_model.py::test_transformer_lm
  uv run pytest tests/test_model.py::test_transformer_lm_truncated_input

===
uv run pytest -k test_cross_entropy
uv run pytest -k test_adamw
uv run pytest -k test_get_lr_cosine_schedule
uv run pytest -k test_gradient_clipping
===
uv run pytest -k test_get_batch
uv run pytest -k test_checkpointing
```


```
uv run python train.py 

```
