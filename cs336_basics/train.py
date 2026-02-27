import os
import time
import typing

import numpy as np
import numpy.typing as npt
import torch
import wandb
from jaxtyping import Int
from dataclasses import dataclass, asdict

from cs336_basics.transformer import TransformerLM, ModelConfig
from cs336_basics.optimizer import cross_entropy, AdamW, get_lr_cosine_schedule, gradient_clipping

from dotenv import load_dotenv

load_dotenv()
assert len(os.environ["WANDB_API_KEY"]) > 0

def get_batch(
    x: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    indices = np.random.randint(0, x.shape[0] - context_length, size=(batch_size,))
    inputs = np.stack([x[i : i + context_length] for i in indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in indices])
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_tensor, targets_tensor


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                    out: str|os.PathLike|typing.BinaryIO|typing.IO[bytes]):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(checkpoint, out)


def load_checkpoint(src: str|os.PathLike|typing.BinaryIO|typing.IO[bytes], model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


@dataclass(kw_only=True)
class TrainConfig:
    vocab_size:int
    d_model:int
    num_heads:int
    d_ff:int
    context_length:int
    num_layers:int
    rope_theta:float|None = 10000.0

    # optimizer
    lr:float=1e-3
    weight_decay:float=0.01
    betas:tuple[float, float]=(0.9, 0.999)
    eps_adam:float=10e-8

    # lr schedule
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    # warmup_iters: int
    # cosine_cycle_iters: int

    # grid clip
    max_l2_norm: float = 1.0 # ?
    eps_clip: float= 1e-6

    # train
    batch_size:int
    dtype: torch.dtype | None = None
    device:torch.device | None = None
    token_ids_path:str|os.PathLike|typing.BinaryIO|typing.IO[bytes] = None

    dataset_dir: str = "datasets/tiny_stories"
    train_data_path: str = "datasets/tiny_stories/train.bin"
    eval_data_path: str = "datasets/tiny_stories/eval.bin"

    # checkpoint
    save_checkpoint_per_steps:int = 10
    save_checkpoint_dir:str|os.PathLike|typing.BinaryIO|typing.IO[bytes] = "checkpoints"

    # wandb
    wandb_project:str="cs336"
    wandb_name:str="my_first_llm"
    
    # timing
    timing_interval_steps:int = 1  # 打印时间信息的间隔步数


def train(config: TrainConfig):
    modelConfig = ModelConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        context_length=config.context_length,
        num_layers=config.num_layers,
        rope_theta=config.rope_theta
    )
    lm = TransformerLM(modelConfig)

    batch_size = config.batch_size
    context_length = config.context_length
    device = config.device

    optimizer = AdamW(lm.parameters(), lr=config.lr, weight_decay=config.weight_decay, 
                      betas=config.betas, eps=config.eps_adam)

    # Load training dataset
    original_data = np.memmap(
        config.train_data_path,
        dtype=np.uint16,
        mode="r+",
    )
    token_ids = torch.from_numpy(original_data)
    # token_ids = np.load(config.train_data_path, allow_pickle=True, mmap_mode="r")


    one_step_len = config.context_length * config.batch_size
    total_steps = len(token_ids) // one_step_len

    print(f"Total train steps is: {total_steps}, one stpe len: {one_step_len}, all token len: {len(token_ids)}")

    wandb.init(project=config.wandb_project, 
               name=config.wandb_name,
               config={**asdict(config), "total_steps": total_steps})

    timing_interval = config.timing_interval_steps
    
    for step in range(total_steps):
        step_start_time = time.perf_counter()
        
        # 数据加载
        inputs, targets = get_batch(token_ids, batch_size, context_length, device)
        
        # 前向传播
        forward_start = time.perf_counter()
        logits = lm.forward(inputs)
        forward_time = time.perf_counter() - forward_start
        
        # 损失计算
        loss_start = time.perf_counter()
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss_time = time.perf_counter() - loss_start

        optimizer.zero_grad()
        
        # 反向传播
        backward_start = time.perf_counter()
        loss.backward()
        backward_time = time.perf_counter() - backward_start
        
        gradient_clipping(lm.parameters(), config.max_l2_norm, config.eps_clip)

        for params in optimizer.param_groups:
            lr = get_lr_cosine_schedule(step, config.max_learning_rate, 
                                        config.min_learning_rate,
                                        warmup_iters=total_steps // 10,
                                        cosine_cycle_iters=total_steps)
            params["lr"] = lr
        
        # 优化器步骤
        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer_time = time.perf_counter() - optimizer_start

        step_end_time = time.perf_counter()
        step_time = step_end_time - step_start_time

        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": torch.exp(loss).item(),
            "train/lr": lr,
        })
        
        if (step + 1) % timing_interval == 0:
            print(f"Step {step + 1}/{total_steps} - Total: {step_time:.2f}s | "
                  f"Forward: {forward_time:.2f}s |"
                  f"Backward: {backward_time:.2f}s | Optim: {optimizer_time:.2f}s |"
                  f"Loss: {loss.item(): .2f}")


    wandb.finish()


tinyStoryConfig = TrainConfig(
    vocab_size=10000,
    d_model=512,
    d_ff=1344,
    num_layers=4,
    num_heads=16,
    context_length=256,

    batch_size=16, #256 
    dtype=torch.float32,
    device=torch.device("cpu")
)


if __name__ == "__main__":
    train(tinyStoryConfig)