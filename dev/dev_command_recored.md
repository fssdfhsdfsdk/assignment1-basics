


```
uv run pytest 
uv run pytest tests/test_train_bpe.py

uv run pytest tests/test_model.py::test_linear
uv run pytest tests/test_model.py::test_embedding

pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_softmax_matches_pytorch
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
```


```
uv run python train.py 

```
