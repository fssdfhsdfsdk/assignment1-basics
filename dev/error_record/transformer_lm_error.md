

【问题】

```
            raise AssertionError(f"Snapshot contains extra keys {extra_keys} for {test_name}")

        # Compare all arrays
        for key in arrays_dict:
>           np.testing.assert_allclose(
                _canonicalize_array(arrays_dict[key]),
                expected_arrays[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"Array '{key}' does not match snapshot for {test_name}",
            )
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.0001
E           Array 'array' does not match snapshot for test_transformer_lm
E           Mismatched elements: 472352 / 480000 (98.4%)
E           Max absolute difference among violations: 10.165454
E           Max relative difference among violations: 14569.58
E            ACTUAL: array([[[-5.933841,  2.806938, -4.145932, ..., -4.222383, -5.12304 ,
E                   [-5.014392,  3.368196, -2.147645, ..., -4.270598, -4.599124,...
E            DESIRED: array([[[ -2.785665,   3.587132,   0.285842, ...,  -6.533192,
E                     -6.18881 ,  -7.657421],
E                   [ -2.290899,   4.14432 ,   0.279644, ...,  -3.832705,...

tests\conftest.py:89: AssertionError
============================================= short test summary info ============================================= 
FAILED tests/test_model.py::test_transformer_lm - AssertionError:
================================================ 1 failed in 1.14s ================================================
```

【原因】 通过属性赋值，属性漏打一个字母，导致 “隐蔽bug”
```
    with torch.no_grad():
        lm.embedding.embed_matrix.dat = weights['token_embeddings.weight']
```

【耗时】30min + ，全面排查之前写的模块。耗费时间，一直漏掉 “致命的隐蔽bug”