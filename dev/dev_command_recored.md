


```
uv run pytest 
uv run pytest tests/test_train_bpe.py

uv run pytest tests/test_model.py::test_linear
uv run pytest tests/test_model.py::test_embedding

pytest -k test_rmsnorm
uv run pytest -k test_swiglu
```


```
uv run python train.py 

```
