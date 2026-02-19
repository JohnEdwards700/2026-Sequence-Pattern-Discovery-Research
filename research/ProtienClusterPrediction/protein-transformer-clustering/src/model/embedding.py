import torch
import torch.nn as nn

class ProteinEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(ProteinEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)