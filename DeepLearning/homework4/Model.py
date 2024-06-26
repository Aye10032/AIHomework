import math

import torch
from torch import nn, Tensor
from Config import *


class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, dropout: float, max_token: int):
        super(EmbeddingWithPosition, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, emb_size)

        position_idx = torch.arange(0, max_token, dtype=torch.float).unsqueeze(-1)
        position_emb = position_idx * torch.exp(-torch.arange(0, emb_size, 2) * math.log(1000.0) / emb_size)
        position_encoding = torch.zeros(max_token, emb_size)
        position_encoding[:, 0::2] = torch.sin(position_emb)
        position_encoding[:, 1::2] = torch.cos(position_emb)

        # (1, max_token, emb_size)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', position_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # (batch_size, seq_length, vocab_size)
        # -> (batch_size, seq_length, emb_size)
        x = self.word_embedding(x)
        x = x + self.pos_encoding[:, :x.shape[1], :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, head: int):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head = head

        self.w_q = nn.Linear(emb_size, head * hidden_size)
        self.w_k = nn.Linear(emb_size, head * hidden_size)
        self.w_v = nn.Linear(emb_size, head * hidden_size)

        self.output = nn.Linear(head * hidden_size, emb_size)

    def forward(self, x_vk: Tensor, x_q: Tensor, attn_mask: Tensor):
        # (batch_size, seq_length, emb_size)
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

        # -> (batch_size, seq_length, emb_size)
        return self.output(z)


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, ffw_size: int):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(emb_size, ffw_size),
            nn.GELU(),
            nn.Linear(ffw_size, emb_size)
        )

    def forward(self, x: Tensor):
        return self.ffw(x)


class EncoderBlock(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, head: int, ffw_size: int, attn_dropout: float, ffw_dropout: float):
        super(EncoderBlock, self).__init__()

        self.attn_layer = MultiHeadAttention(emb_size, hidden_size, head)
        self.attn_norm = nn.LayerNorm(emb_size)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.ffw_layer = FeedForward(emb_size, ffw_size)
        self.ffw_norm = nn.LayerNorm(emb_size)
        self.ffw_drop = nn.Dropout(ffw_dropout)

    def forward(self, x: Tensor, attn_mask: Tensor):
        # (batch_size, seq_length, emb_size)
        z = self.attn_layer(x, x, attn_mask)
        output1 = self.attn_norm(x + self.attn_drop(z))

        z = self.ffw_layer(output1)
        output2 = self.ffw_norm(output1 + self.ffw_drop(z))

        return output2


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            emb_size: int,
            hidden_size: int,
            head: int,
            ffw_size: int,
            num_layers: int,
            attn_dropout: float = 0.1,
            ffw_dropout: float = 0.1,
            emb_dropout: float = 0.1,
            max_token: int = 200
    ):
        super(Encoder, self).__init__()

        self.emb = EmbeddingWithPosition(vocab_size, emb_size, emb_dropout, max_token)

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_blocks.append(EncoderBlock(emb_size, hidden_size, head, ffw_size, attn_dropout, ffw_dropout))

    def forward(self, x: Tensor):
        # (batch_size, 1, seq_length)
        # -> (batch_size, seq_length, seq_length)
        mask: Tensor = (x == PAD_IDX).unsqueeze(1).repeat(1, x.shape[1], 1)

        # (batch_size, seq_length)
        # -> (batch_size, seq_length, emb_size)
        x = self.emb(x)

        for block in self.encoder_blocks:
            x = block(x, mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, head: int, ffw_size: int, attn_dropout: float, ffw_dropout: float):
        super(DecoderBlock, self).__init__()

        self.masked_attn_layer = MultiHeadAttention(emb_size, hidden_size, head)
        self.masked_attn_norm = nn.LayerNorm(emb_size)
        self.masked_attn_drop = nn.Dropout(attn_dropout)

        self.attn_layer = MultiHeadAttention(emb_size, hidden_size, head)
        self.attn_norm = nn.LayerNorm(emb_size)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.ffw_layer = FeedForward(emb_size, ffw_size)
        self.ffw_norm = nn.LayerNorm(emb_size)
        self.ffw_drop = nn.Dropout(ffw_dropout)

    def forward(self, x: Tensor, encoder_output: Tensor, mask1: Tensor, mask2: Tensor):
        z = self.masked_attn_layer(x, x, mask1)
        output1 = self.masked_attn_norm(x + self.masked_attn_norm(z))

        z = self.attn_layer(encoder_output, output1, mask2)
        output2 = self.attn_norm(output1 + self.attn_drop(z))

        z = self.ffw_layer(output2)
        output = self.ffw_norm(output2 + self.ffw_drop(z))

        return output


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            emb_size: int,
            hidden_size: int,
            head: int,
            ffw_size: int,
            num_layers: int,
            attn_dropout: float = 0.1,
            ffw_dropout: float = 0.1,
            emb_dropout: float = 0.1,
            max_token: int = 200
    ):
        super(Decoder, self).__init__()

        self.emb = EmbeddingWithPosition(vocab_size, emb_size, emb_dropout, max_token)

        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_blocks.append(DecoderBlock(emb_size, hidden_size, head, ffw_size, attn_dropout, ffw_dropout))

    def forward(self, x: Tensor, encoder_in: Tensor, encoder_out: Tensor):
        # (batch_size, seq_len, seq_len)
        mask1: Tensor = (x == PAD_IDX).unsqueeze(1).repeat(1, x.shape[1], 1)
        mask1_fill = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().unsqueeze(0).repeat(x.shape[0], 1, 1).to(
            mask1.device)
        mask1 = mask1 | mask1_fill

        # (batch_size, target_len, src_len)
        mask2: Tensor = (encoder_in == PAD_IDX).unsqueeze(1).repeat(1, x.shape[1], 1)
        x = self.emb(x)

        for block in self.decoder_blocks:
            x = block(x, encoder_out, mask1, mask2)

        return x


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            target_vocab_size: int,
            emb_size: int,
            hidden_size: int,
            head: int,
            ffw_size: int,
            num_layers: int,
            attn_dropout: float = 0.1,
            ffw_dropout: float = 0.1,
            emb_dropout: float = 0.1,
            max_token: int = 200
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size, emb_size, hidden_size, head, ffw_size, num_layers, attn_dropout, ffw_dropout, emb_dropout, max_token
        )
        self.decoder = Decoder(
            target_vocab_size, emb_size, hidden_size, head, ffw_size, num_layers, attn_dropout, ffw_dropout, emb_dropout, max_token
        )
        self.liner = nn.Linear(emb_size, target_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: Tensor, target: Tensor):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(target, src, encoder_output)
        output = self.liner(decoder_output)

        return self.softmax(output)

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, x: Tensor, encoder_in: Tensor, encoder_out: Tensor):
        decoder_out = self.decoder(x, encoder_in, encoder_out)
        decoder_out = self.liner(decoder_out)
        return self.softmax(decoder_out)
