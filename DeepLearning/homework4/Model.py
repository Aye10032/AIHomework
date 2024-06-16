import math

import torch
from torch import nn, Tensor


class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, dropout: float, max_token: int):
        super(EmbeddingWithPosition, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, emb_dim)

        position_idx = torch.arange(0, max_token, dtype=torch.float).unsqueeze(-1)
        position_emb = position_idx * torch.exp(-torch.arange(0, emb_dim, 2) * math.log(1000.0) / emb_dim)
        position_encoding = torch.zeros(max_token, emb_dim)
        position_encoding[:, 0::2] = torch.sin(position_emb)
        position_encoding[:, 1::2] = torch.cos(position_emb)  # (max_token, emb_dim)

        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', position_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.word_embedding(x)
        x = x + self.pos_encoding[:, :x.shape[1], :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, emb_dropout: float = 0.1, max_token: int = 200):
        super(Encoder, self).__init__()

        self.emb = EmbeddingWithPosition(vocab_size, emb_dim, emb_dropout, max_token)


def main() -> None:
    net = EmbeddingWithPosition(20, 32, 0.1, 60)
    inputs = torch.zeros((16, 56), dtype=torch.long)
    outputs = net.forward(inputs)
    print(outputs.shape)


if __name__ == '__main__':
    main()
