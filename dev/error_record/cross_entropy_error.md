

【现象】
cross_entropy 结果一直错误

【原因】

gather不是从减过最大值的logits中获取，而是直接用的输入logits

```
probility_logits = torch.gather(predict_logits, -1, torch.unsqueeze(targets, 1))
```
正确: 
```
predict_logits_minus_max = predict_logits - max_of_dim
probility_logits = torch.gather(predict_logits_minus_max, -1, torch.unsqueeze(targets, 1))
```


【耗时】较久。代码编写太少, 没有第一时间意识到公式已经错误，反而经过耗时久的对比排查 + 人工排查