import os
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
    max_seq_len: int|None = None

    # optimizer
    lr:float=1e-3
    weight_decay:float=0.01
    betas:tuple[float, float]=(0.9, 0.999)
    eps_adam:float=10e-8

    # lr schedule
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int

    # grid clip
    max_l2_norm: float
    eps_clip: float= 1e-6

    # train
    batch_size:int
    dtype: torch.dtype | None = None
    device:torch.device | None = None
    token_ids_path:str|os.PathLike|typing.BinaryIO|typing.IO[bytes]


    # checkpoint
    save_checkpoint_per_steps:int
    save_checkpoint_dir:str|os.PathLike|typing.BinaryIO|typing.IO[bytes] = "checkpoints"

    # wandb
    wandb_project:str="cs336"
    wandb_name:str="my_first_llm"


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
    
    token_ids = np.load(config.token_ids_path, mmap_mode="r")
    one_step_len = config.context_length * config.batch_size
    total_steps = len(token_ids) // one_step_len

    print(f"Total train steps is: {total_steps}, one stpe len: {one_step_len}, all token len: {len(token_ids)}")

    wandb.init(project=config.wandb_project, 
               name=config.wandb_name,
               config={**asdict(config), "total_steps": total_steps})

    
    for step in range(total_steps):
        inputs, targets = get_batch(token_ids, batch_size, context_length, device)
        logits = lm.forward(inputs)
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(lm.parameters(), config.max_l2_norm, config.eps_clip)

        for params in optimizer.param_groups:
            lr = get_lr_cosine_schedule(step, config.max_learning_rate, 
                                        config.min_learning_rate, config.warmup_iters,
                                        config.cosine_cycle_iters)
            params["lr"] = lr
        optimizer.step()

        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": torch.exp(loss).item(),
            "train/lr": lr
        })


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