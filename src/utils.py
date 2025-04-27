import torch

def simple_tokenizer(sentence):
    return sentence.lower().split()

def build_vocab(sentences):
    vocab = {}
    idx = 0
    for sent in sentences:
        for word in simple_tokenizer(sent):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def encode_sentence(sentence, vocab, max_len=10):
    tokens = simple_tokenizer(sentence)
    ids = [vocab.get(token, 0) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids[:max_len])
import torch

def compute_distance(embed1, embed2):
    return torch.norm(embed1 - embed2, dim=1).item()

