import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.transforms import (
    Compose,
    Resize,
    RandomResizedCrop,
    CenterCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize
)
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trans_train = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

trans_valid = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_set = CIFAR100(root='./cifar100', train=True, download=True, transform=trans_train)
test_set = CIFAR100(root='./cifar100', train=False, download=False, transform=trans_valid)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super(Attention, self).__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super(FeedForward, self).__init__()

        self.layer = nn.Sequential(
            nn.LayerNorm(dim, hidden_dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        return self.layer(x)


class Transformer(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            dim_head: int,
            mlp_dim: int,
            dropout: float = 0.
    ):
        super(Transformer, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ])
            )

    def forward(self, x: Tensor):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x) + x
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    def __init__(
            self, *,
            image_size: tuple[int, int] | int,
            patch_size: tuple[int, int] | int,
            num_classes: int,
            dim: int,
            depth: int,
            heads: int,
            mlp_dim: int,
            pool: str = 'cls',
            channels: int = 3,
            dim_head: int = 64,
            dropout: float = 0.,
            emb_dropout: float = 0
    ):
        super(ViT, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Error image size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'err pool type'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transform = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img: Tensor):
        x: Tensor = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transform(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


net = ViT(
    image_size=256,
    patch_size=32,
    num_classes=100,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512
).to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


def train(epoch: int):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in loop:
        inputs: Tensor
        targets: Tensor

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=f'{round(100. * correct / total, 3)} ({correct}/{total})')


def main() -> None:
    for i in range(50):
        train(i)


if __name__ == '__main__':
    main()
