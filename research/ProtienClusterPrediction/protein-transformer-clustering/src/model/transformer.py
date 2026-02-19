import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=512, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.transformer_encoder = nn.TransformerEncoder(nn.ModuleList(self.encoder_layers), num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)