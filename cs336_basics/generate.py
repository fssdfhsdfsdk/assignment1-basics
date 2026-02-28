
import os
from cs336_basics.tokenizer_copy import Tokenizer
from cs336_basics.transformer import TransformerLM, softmax
import torch
from typing import Tuple

class Generator:
    def __init__(self, context_length:int, 
                 vocab_path:str|os.PathLike, merge_rank_path:str|os.PathLike):
        self.tokenizer = \
            Tokenizer.from_files(vocab_path, merge_rank_path, special_tokens=["<|endoftext|>"])
        self.end_token_id = self.tokenizer.encode("<|endoftext|>")[0]
        self.context_length = context_length

    def default_gen(self, prompt: str, model:TransformerLM, device: str | torch.device):
        res, token_count = self.generate_from_prompt(prompt, lm, config.device)
        print("prompt: ", prompt)
        print("Answer: ", res)
        print("Gen-length: ", len(res), ", Token Count: ", token_count)
        
    def generate_from_prompt(self, prompt: str, model:TransformerLM, 
                             device: str | torch.device,
                             tempeerature:float=0.7,
                              top_p:float=0.9, max_token_num:int=0) -> Tuple[str, int]:
        token_ids_list = self.tokenizer.encode(prompt)
        input_token_len = len(token_ids_list)
        max_generate_support = self.context_length - input_token_len
        if max_token_num > 0:
            assert max_token_num <= max_generate_support
        else:
            max_token_num = max_generate_support

        # 2D输入: 生成形状 (1, seq_len)
        token_ids = torch.tensor([token_ids_list], device=device)  #from list[int]

        with torch.no_grad():
            gen_token_count = 0
            while gen_token_count < max_token_num:
                logits = model.forward(token_ids)

                next_token_logits = logits[:, -1, :] 
                next_token = self.sample_by_top_p(next_token_logits, tempeerature, top_p)
                gen_token_count += 1
                if next_token.item() == self.end_token_id: # int类型
                    break
                # next_token 形状通常是 [1] 或 []，需要变为 [1, 1] 才能 cat 到 [1, seq]
                next_token = next_token.unsqueeze(0)
                # RuntimeError: Tensors must have same number of dimensions: got 2 and 3
                # next_token unsqueeze 1次
                token_ids = torch.cat((token_ids, next_token), dim=-1)

        gen_tokens = token_ids[0, input_token_len:]
        token_count = len(gen_tokens)
        gen_token_list = gen_tokens.cpu().tolist()
        return self.tokenizer.decode(gen_token_list), token_count
        

    def sample_by_top_p(self, logits:torch.Tensor, temperature:float=0.7, top_p:float=0.9) -> torch.Tensor:
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=False)
        
        logits /= temperature
        logits_sorted, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        probs = softmax(logits, dim=-1)
        cumsum_probs = torch.cumsum(probs, dim=-1)

        sorted_indices_to_remove = cumsum_probs > top_p
        # [Ask] 为什么要clone？是因为视图的原因？无法覆盖原有的内存？
        # 重叠内存赋值行为：未定义
        # 由于 A 和 B 内存重叠，如果在拷贝过程中，先写入的位置恰好是后续读取位置的源数据，
        # 那么读到的就是“已经被修改过的新值”，而不是“原始旧值”。
        # Desc: token向左移动一位，去掉最小概率的token. 保留最大概率的
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        logits_indices_to_rm = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices,
                                                                  src=sorted_indices_to_remove)
        logits[logits_indices_to_rm] = torch.finfo(logits.dtype).min

        probs = softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token


if __name__ == "__main__":
    from cs336_basics.train import tinyStoryConfig
    from cs336_basics.transformer import ModelConfig

    config = tinyStoryConfig
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

    generator = Generator(config.context_length, config.tokenizer_vocab_pkl_path,
                           config.tokenizer_merge_pkl_path)
    
    # 可以直接使用 未训练过的随机初始化模型
    if False:
        checkpoint = torch.load(
            "../checkpoints/checkpoint_final.pt",
            map_location=config["device"],
        )

        lm.load_state_dict(checkpoint["model"])
    lm.eval()

    prompt = "Once upon a time"
    generator.default_gen(prompt, lm, config.device)