



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

### 2、【ok】

```
tests\conftest.py:89: AssertionError
```

[error 02](rsmnorm_error02.md)



## FNN

### 1、【ok】

===
```
tests\adapters.py:103: in run_swiglu
    return swiglu.forward(in_features)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
cs336_basics\transformer.py:99: in forward
    return self.w2.forward(silu(self.w1.forward(x)) *self.w3.forward(x))
                                ^^^^^^^^^^^^^^^^^^
cs336_basics\transformer.py:22: in forward
    return einsum(x, self.weights, "... d_in,  d_in d_out -> ... d_out")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv\Lib\site-packages\einops\einops.py:916: in einsum
    return get_backend(tensors[0]).einsum(pattern, *tensors)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv\Lib\site-packages\einops\_backends.py:288: in einsum
    return self.torch.einsum(pattern, *x)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 


E           RuntimeError: einsum(): subscript a has size 128 for operand 1 which does not broadcast with previously seen size 64

```
===

原因：

测试给定的参数（weight），要先进行转置。