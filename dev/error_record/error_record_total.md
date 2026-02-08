



### 1、【ok】

```
    def forward(self, x: torch.Tensor) -> torch.Tensor:
>       return self.weights @ x
               ^^^^^^^^^^^^^^^^
E       RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x12 and 128x64)

cs336_basics\transformer.py:19: RuntimeError
============================================= short test summary info ============================================= 
FAILED tests/test_model.py::test_linear - RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x12 and 128x64)
```

打印：

```
tests/test_model.py::test_linear torch.Size([64, 128])
torch.Size([4, 12, 64])
```

原因：
  - 输入的x的shape是 row-vector ，而不是普通数学计算中的 column向量 （Wx）。
  - 未明确输入的shape

### 2、【ok】


```
>           ln.weights.copy_(weights.T)
E           RuntimeError: The size of tensor a (64) must match the size of tensor b (128) at non-singleton dimension 1
```

原因：ln.weights的shape与代码不符。


## RSMNorm

### 1、【ok】

[error 01](rsmnorm_error01.md)

## 2、【ok】

```
tests\conftest.py:89: AssertionError
```

[error 02](rsmnorm_error02.md)
