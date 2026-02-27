
# 1、模型参数-遗留问题

```
(cs336-basics) ➜  /workspace /workspace/.venv/bin/python /workspace/cs336_basics/train.py
Traceback (most recent call last):
  File "/workspace/cs336_basics/train.py", line 241, in <module>
    train(tinyStoryConfig)
  File "/workspace/cs336_basics/train.py", line 150, in train
    lm = TransformerLM(modelConfig)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cs336_basics/transformer.py", line 303, in __init__
    self.tb_layers = nn.ModuleList([TransformerBlock(config.d_model, 
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cs336_basics/transformer.py", line 283, in __init__
    self.token_positions = torch.arange(max_seq_len, device=self.device)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: arange() received an invalid combination of arguments - got (NoneType, device=NoneType), but expected one of:
 * (Number end, *, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (Number start, Number end, *, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (Number start, Number end, Number step = 1, *, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
```


# 2、numpy.load 

```
ValueError: This file contains pickled (object) data. If you trust the file you can load it unsafely using the `allow_pickle=` keyword argument or `pickle.load()`. 
```

这个错误通常发生在你尝试使用 `numpy.load()` 读取一个包含 **Python 对象**（如字典、列表或自定义类）的 `.npy` 文件时。
出于安全考虑，较新版本的 NumPy 默认禁用了加载这种包含"序列化对象"（pickle）的文件，因为加载不明来源的 pickle 文件可能会在你的电脑上执行恶意代码。