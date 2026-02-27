import os
from multiprocessing import Process, Queue
import queue
import regex as re
import os
from typing import BinaryIO


def get_bpe_pair_stats(preTokenFreqDc:dict, mergeTokensDc:dict):
    pairs = {}
    for tok, freq in preTokenFreqDc.items():
        mTokList = mergeTokensDc[tok]
        tokpairs = zip(mTokList[:-1], mTokList[1:])
        for pair in tokpairs:
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs

def bpe_merge(mergeTokensDc:dict, merge):
    for tok, mergeToks in mergeTokensDc.items():
        mlen = len(mergeToks)
        if mlen <= 1:
            continue

        new_mergeToks = []
        i = 0
        while i < mlen:
            # Note: 不能外层判断个数, 未合并时会漏掉
            if i < mlen - 1 and mergeToks[i] == merge[0] and mergeToks[i+1] == merge[1]:
                new_mergeToks.append(mergeToks[i] + mergeToks[i+1])
                i += 2
            else:
                new_mergeToks.append(mergeToks[i])
                i += 1
        mergeTokensDc[tok] = new_mergeToks


import cProfile, pstats
from functools import wraps

def profile_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative').print_stats(5)
        stats.dump_stats("profile_output.prof")
        return result
    return wrapper

# @profile_func
def train_bpe(
        input_path: str|os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path, 'r') as f:
        content = f.read()
    
    vocabs = {}
    vocabSize = 0
    for i, tok in enumerate(special_tokens):
        vocabs[vocabSize] = bytes(tok.encode('utf-8'))
        vocabSize += 1
    
    for i in range(256):
        vocabs[vocabSize] = bytes([i])
        vocabSize += 1
    
    pats = []
    for tok in special_tokens:
        pats.append(re.escape(tok))

    docPat = re.compile("|".join(pats))
    docs = re.split(docPat, content)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    preTokenFreqDc = {}
    mergeTokensDc = {}
    for doc in docs:
        matchs = re.finditer(PAT, doc)
        for mt in matchs:
            freq = preTokenFreqDc.get(mt.group(), 0)
            if freq == 0:
                mergeTokensDc[mt.group()] = [bytes([c]) for c in mt.group().encode('utf-8')]
            preTokenFreqDc[mt.group()] = freq + 1

    merges = []

    assert(len(vocabs) == vocabSize)
    while vocabSize < vocab_size:
        pairs = get_bpe_pair_stats(preTokenFreqDc, mergeTokensDc)
        max_pair = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]
        # if True:
        #     t1 = pairs.get((b' .', b'..'), 0)
        #     if t1 > 0:
        #         print("===", t1)
        #         print(max_pair)
        #         print(pairs[max_pair])
        bpe_merge(mergeTokensDc, max_pair)
        merges.append(max_pair)
        vocabs[vocabSize] = max_pair[0] + max_pair[1]
        vocabSize += 1
        

    return vocabs, merges

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe_with_mulitprocess(
        input_path: str|os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")


        bpe_workers = []

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            for i in range(num_processes):
                Process()


if __name__ == "__main__":
    valid_data_file = r"data\TinyStoriesV2-GPT4-valid.txt"