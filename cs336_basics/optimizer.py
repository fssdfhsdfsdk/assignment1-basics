import math
from typing import Callable, Iterable, Optional

import torch
from jaxtyping import Float, Int

def cross_entropy(predict_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dim = -1
    max_of_dim = torch.max(predict_logits, dim=dim, keepdim=True).values
    predict_logits_minus_max = predict_logits - max_of_dim
    predict_logits_sum_res = torch.sum(predict_logits_minus_max.exp(), dim=dim, keepdim=True)
    probility_logits = torch.gather(predict_logits_minus_max, -1, torch.unsqueeze(targets, 1))

    # -log(exp(plogits)/logits_sum) = log(logits_sum) - plogits
    logp = predict_logits_sum_res.log() - probility_logits

    return torch.mean(logp)


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=10e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple of length 2, got: {betas}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 value: {beta2}")
        
        defaults = {
            "lr":lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m")
                v = state.get("v")
                if m is None:
                    m = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["m"] = m 
                if v is None:
                    v = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = v
                
                grad = p.grad
                state["m"].mul_(beta1).add_(grad, alpha=1-beta1) 
                state["v"].mul_(beta2).addcmul_(grad, grad, value=(1-beta2))
                lr_t = lr * (math.sqrt(1-beta2**t)/(1-beta1**t))
                p -= lr_t * (m/(torch.sqrt(v) + eps))
                
                if weight_decay != 0.0:
                    # p.data -= lr * weight_decay * p.data 
                    p.mul_(1.0 - lr * weight_decay)

                state["t"] = t + 1
        return loss
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * \
            (1 + math.cos(((it - warmup_iters)/(cosine_cycle_iters-warmup_iters))*math.pi)) *\
                  (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float= 1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    paras = [para for para in parameters if para.grad is not None]
    if not paras:
        return

    total_norm = 0.0
    for para in paras:
        para_norm = para.grad.detach().data.norm(2)
        total_norm += para_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_cof = max_l2_norm / (total_norm + eps)
    if clip_cof < 1:
        for para in paras:
            para.grad.detach().mul_(clip_cof)

