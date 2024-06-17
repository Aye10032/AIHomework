from dataclasses import dataclass

EPOCH = 5
LR = 1e-3

UNK_IDX = 0
SOB_IDX = 1
EOB_IDX = 2
PAD_IDX = 3


@dataclass
class ModelConfig:
    src_vocab_size: int
    target_vocab_size: int
    emb_size: int
    hidden_size: int
    head: int
    ffw_size: int
    num_layers: int
    attn_dropout: float
    ffw_dropout: float
    emb_dropout: float
    max_token: int

    @classmethod
    def default_config(cls):
        return cls(53713, 35029, 512, 64, 8, 2048, 6, 0.1, 0.1, 0.1, 200)
