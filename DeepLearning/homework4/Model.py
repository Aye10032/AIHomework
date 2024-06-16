import math

import torch
from torch import nn, Tensor
from Config import *


class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, dropout: float, max_token: int):
        super(EmbeddingWithPosition, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, emb_dim)

        position_idx = torch.arange(0, max_token, dtype=torch.float).unsqueeze(-1)
        position_emb = position_idx * torch.exp(-torch.arange(0, emb_dim, 2) * math.log(1000.0) / emb_dim)
        position_encoding = torch.zeros(max_token, emb_dim)
        position_encoding[:, 0::2] = torch.sin(position_emb)
        position_encoding[:, 1::2] = torch.cos(position_emb)

        # (1, max_token, emb_dim)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', position_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # (batch_size, seq_length, vocab_size)
        # -> (batch_size, seq_length, emb_dim)
        x = self.word_embedding(x)
        x = x + self.pos_encoding[:, :x.shape[1], :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, hidden_size: int, head: int):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head = head

        self.w_q = nn.Linear(emb_dim, head * hidden_size)
        self.w_k = nn.Linear(emb_dim, head * hidden_size)
        self.w_v = nn.Linear(emb_dim, head * hidden_size)

    def forward(self, x_vk: Tensor, x_q: Tensor, attn_mask: Tensor):
        # (batch_size, seq_length, emb_dim)
        # -> (batch_size, seq_length, head * hidden_size)
        q: Tensor = self.w_q(x_q)
        k: Tensor = self.w_k(x_vk)
        v: Tensor = self.w_v(x_vk)

        # -> (batch_size, head, seq_length, hidden_size)
        q = q.view((q.shape[0], q.shape[1], self.head, self.hidden_size)).transpose(1, 2)
        k = k.view((k.shape[0], k.shape[1], self.head, self.hidden_size)).transpose(1, 2)
        v = v.view((v.shape[0], v.shape[1], self.head, self.hidden_size)).transpose(1, 2)
        # -> (batch_size, head, hidden_size, seq_length)
        k = k.transpose(2, 3)

        # -> (batch_size, head, seq_length, seq_length)
        attn = torch.matmul(q, k) / math.sqrt(self.hidden_size)

        # (batch_size, seq_length, seq_length)
        # -> (batch_size, head, seq_length, seq_length)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.head, 1, 1)
        attn = attn.masked_fill(attn_mask, -torch.inf)
        attn = torch.softmax(attn, dim=-1)

        # -> (batch_size, head, seq_length, hidden_size)
        z = torch.matmul(attn, v)
        # -> (batch_size, seq_length, head, hidden_size)
        z = z.transpose(1, 2)
        # -> (batch_size, seq_length, head * hidden_size)
        z = z.reshape((z.shape[0], z.shape[1], self.head * self.hidden_size))

        return z


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, emb_dropout: float = 0.1, max_token: int = 200):
        super(Encoder, self).__init__()

        self.emb = EmbeddingWithPosition(vocab_size, emb_dim, emb_dropout, max_token)


def main() -> None:
    net = MultiHeadAttention(32, 64, 3)
    inputs = torch.zeros((16, 60, 32), dtype=torch.float)
    mask = torch.tril(torch.ones((60, 60))).bool().repeat(16, 1, 1)
    outputs = net.forward(inputs, inputs, mask)
    print(outputs.shape)


if __name__ == '__main__':
    main()
