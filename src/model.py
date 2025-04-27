import torch
import torch.nn as nn

class SimpleSentenceEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def tokenize(self, sentence):
        # Improved tokenization
        tokens = sentence.lower().replace(".", "").replace(",", "").split()
        token_ids = [sum(ord(c) for c in token) % 10000 for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)

    def forward(self, sentences):
        embedded = []
        for sent in sentences:
            token_ids = self.tokenize(sent)
            embeds = self.embedding(token_ids)
            sent_embed = embeds.mean(dim=0)  # mean pooling
            embedded.append(sent_embed)
        return torch.stack(embedded)

from sentence_transformers import SentenceTransformer

def get_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

