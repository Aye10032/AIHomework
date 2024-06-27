import math

import torch
from torch import nn, Tensor
from Config import *


class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, dropout: float, max_token: int):
        super(EmbeddingWithPosition, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, emb_size)

        position_idx = torch.arange(0, max_token, dtype=torch.float).unsqueeze(-1)
        position_emb = position_idx * torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size)
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

    def forward(self, x_kv: Tensor, x_q: Tensor, attn_mask: Tensor):
        # (batch_size, seq_length, emb_size)
        # -> (batch_size, seq_length, head * hidden_size)
        q: Tensor = self.w_q(x_q)
        k: Tensor = self.w_k(x_kv)
        v: Tensor = self.w_v(x_kv)

        # -> (batch_size, seq_length, head, hidden_size)
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

    def forward(self, encoder_in: Tensor, attn_mask: Tensor):
        # (batch_size, seq_length, emb_size)
        attn_out = self.attn_layer(encoder_in, encoder_in, attn_mask)
        ffw_in = self.attn_norm(encoder_in + self.attn_drop(attn_out))

        ffw_out = self.ffw_layer(ffw_in)
        output = self.ffw_norm(ffw_in + self.ffw_drop(ffw_out))

        return output


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            emb_size: int,
            hidden_size: int,
            head: int,
            ffw_size: int,
            num_layers: int,
            attn_dropout: float = 0.,
            ffw_dropout: float = 0.,
            emb_dropout: float = 0.,
            max_token: int = 200
    ):
        super(Encoder, self).__init__()

        self.emb = EmbeddingWithPosition(vocab_size, emb_size, emb_dropout, max_token)

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_blocks.append(EncoderBlock(emb_size, hidden_size, head, ffw_size, attn_dropout, ffw_dropout))

    def forward(self, encoder_in: Tensor):
        # (batch_size, 1, seq_length)
        # -> (batch_size, seq_length, seq_length)
        mask: Tensor = encoder_in.eq(PAD_IDX).unsqueeze(1).repeat(1, encoder_in.shape[1], 1)

        # (batch_size, seq_length)
        # -> (batch_size, seq_length, emb_size)
        encoder_out = self.emb(encoder_in)

        for block in self.encoder_blocks:
            encoder_out = block(encoder_out, mask)

        return encoder_out


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

    def forward(self, decoder_in: Tensor, encoder_output: Tensor, decoder_mask: Tensor, cross_mask: Tensor):
        mask_attn_out = self.masked_attn_layer(decoder_in, decoder_in, decoder_mask)
        mask_attn_out = self.masked_attn_norm(decoder_in + self.masked_attn_norm(mask_attn_out))

        attn_out = self.attn_layer(encoder_output, mask_attn_out, cross_mask)
        attn_out = self.attn_norm(mask_attn_out + self.attn_drop(attn_out))

        ffw_out = self.ffw_layer(attn_out)
        output = self.ffw_norm(attn_out + self.ffw_drop(ffw_out))

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

    def forward(self, decoder_in: Tensor, encoder_in: Tensor, encoder_out: Tensor):
        # (batch_size, seq_len, seq_len)
        decoder_mask: Tensor = decoder_in.eq(PAD_IDX).unsqueeze(1).repeat(1, decoder_in.shape[1], 1)
        decoder_mask_fill = torch.triu(torch.ones(decoder_in.shape[1], decoder_in.shape[1]), diagonal=1)
        decoder_mask_fill = decoder_mask_fill.bool().unsqueeze(0).repeat(decoder_in.shape[0], 1, 1).to(decoder_mask.device)
        decoder_mask = decoder_mask | decoder_mask_fill

        # (batch_size, target_len, src_len)
        cross_mask: Tensor = encoder_in.eq(PAD_IDX).unsqueeze(1).repeat(1, decoder_in.shape[1], 1)
        decoder_in = self.emb(decoder_in)

        for block in self.decoder_blocks:
            decoder_in = block(decoder_in, encoder_out, decoder_mask, cross_mask)

        return decoder_in


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

    def forward(self, encoder_in: Tensor, decoder_in: Tensor):
        encoder_out = self.encoder(encoder_in)
        decoder_out = self.decoder(decoder_in, encoder_in, encoder_out)
        output = self.liner(decoder_out)

        return self.softmax(output)

    def encode(self, encoder_in: Tensor):
        return self.encoder(encoder_in)

    def decode(self, decoder_in: Tensor, encoder_in: Tensor, encoder_out: Tensor):
        decoder_out = self.decoder(decoder_in, encoder_in, encoder_out)
        decoder_out = self.liner(decoder_out)
        return self.softmax(decoder_out)
