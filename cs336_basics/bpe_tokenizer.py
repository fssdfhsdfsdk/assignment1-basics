import os
import multiprocessing
import regex as re

def get_tokenizer():
    return None


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

