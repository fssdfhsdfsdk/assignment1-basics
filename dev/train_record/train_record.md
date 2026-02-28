


8核16 GB： batch_size = 64
 - 内存峰值：52.5%
 - CPU利用率很低

预估训练时间：
 33033 * 15 / (24 * 3600) = 5.7 天


```
Step 1/33033 - Total: 14.20s | Forward: 3.92s |Backward: 9.99s | Optim: 0.11s |Loss:  9.25
Step 2/33033 - Total: 14.59s | Forward: 4.21s |Backward: 10.09s | Optim: 0.10s |Loss:  9.24
Step 3/33033 - Total: 14.20s | Forward: 4.05s |Backward: 9.90s | Optim: 0.10s |Loss:  9.25
Step 4/33033 - Total: 14.11s | Forward: 4.04s |Backward: 9.88s | Optim: 0.01s |Loss:  9.24
Step 5/33033 - Total: 13.88s | Forward: 4.11s |Backward: 9.50s | Optim: 0.09s |Loss:  9.25
Step 6/33033 - Total: 13.40s | Forward: 4.01s |Backward: 9.11s | Optim: 0.09s |Loss:  9.26
Step 7/33033 - Total: 13.71s | Forward: 4.01s |Backward: 9.41s | Optim: 0.01s |Loss:  9.24
Step 8/33033 - Total: 13.69s | Forward: 4.10s |Backward: 9.30s | Optim: 0.09s |Loss:  9.25
Step 9/33033 - Total: 13.51s | Forward: 4.01s |Backward: 9.30s | Optim: 0.01s |Loss:  9.25
Step 10/33033 - Total: 14.00s | Forward: 4.02s |Backward: 9.71s | Optim: 0.01s |Loss:  9.25
Step 11/33033 - Total: 14.10s | Forward: 3.94s |Backward: 9.91s | Optim: 0.01s |Loss:  9.24
Step 12/33033 - Total: 13.70s | Forward: 4.02s |Backward: 9.41s | Optim: 0.01s |Loss:  9.23
Step 13/33033 - Total: 14.11s | Forward: 4.10s |Backward: 9.79s | Optim: 0.01s |Loss:  9.24
Step 14/33033 - Total: 14.68s | Forward: 4.20s |Backward: 10.12s | Optim: 0.10s |Loss:  9.23
Step 15/33033 - Total: 14.61s | Forward: 4.13s |Backward: 10.20s | Optim: 0.09s |Loss:  9.24
Step 16/33033 - Total: 13.91s | Forward: 4.10s |Backward: 9.59s | Optim: 0.01s |Loss:  9.23
Step 17/33033 - Total: 13.79s | Forward: 4.03s |Backward: 9.50s | Optim: 0.09s |Loss:  9.23
Step 18/33033 - Total: 13.39s | Forward: 3.94s |Backward: 9.20s | Optim: 0.09s |Loss:  9.22
Step 19/33033 - Total: 13.91s | Forward: 3.95s |Backward: 9.70s | Optim: 0.01s |Loss:  9.22
Step 20/33033 - Total: 13.71s | Forward: 4.09s |Backward: 9.48s | Optim: 0.01s |Loss:  9.22
Step 21/33033 - Total: 14.08s | Forward: 4.09s |Backward: 9.79s | Optim: 0.09s |Loss:  9.21
Step 22/33033 - Total: 13.81s | Forward: 3.95s |Backward: 9.60s | Optim: 0.01s |Loss:  9.21
```


Eval:

```
Eval steps: 10, one stpe len: 16384, all token len: 5465882
New best eval loss: 9.2626
Step 5/33033 - Eval and save checkpoint: 54.25s | 
```