import os

import numpy as np
import torch
from accelerate import Accelerator
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn as nn
import seaborn as sns
from evaluate import CombinedEvaluations, EvaluationModule
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.
    ):
        """
        初始化注意力机制模块。

        :param dim: 输入特征维度
        :param heads: 注意力头的数量，默认为8
        :param dim_head: 每个注意力头的维度，默认为64
        :param dropout: Dropout比例，默认为0.
        """
        super(Attention, self).__init__()

        # 计算内部维度和是否需要投影输出的标志
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # 注意力头数
        self.scale = dim_head ** -0.5  # 缩放因子

        self.norm = nn.LayerNorm(dim)  # 层归一化

        self.attend = nn.Softmax(dim=-1)  # 注意力分布计算
        self.dropout = nn.Dropout(dropout)  # Dropout层

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 投影层，用于生成Q、K、V

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 投影回原始维度或保持不变

    def forward(self, x):
        """
        前向传播。

        :param x: 输入特征
        :return: 经过注意力机制处理后的特征
        """
        x = self.norm(x)  # 层归一化

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 将输出分为Q、K、V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # 重新排列Q、K、V的形状

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算Q和K的点积
        attn = self.attend(dots)  # 计算注意力分布
        attn = self.dropout(attn)  # 应用dropout

        out = torch.matmul(attn, v)  # 根据注意力分布加权V
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新排列输出形状
        return self.to_out(out)  # 投影回原始维度或保持不变


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
            layers: int,
            heads: int,
            hidden_size: int,
            mlp_size: int,
            dropout: float = 0.,
    ):
        """
        初始化Transformer模型。

        :param dim: 输入数据的维度。
        :param layers: 隐藏层的数量。
        :param heads: 注意力机制的头数。
        :param hidden_size: 注意力机制中每个头的维度。
        :param mlp_size: 多层感知器(MLP)的维度。
        :param dropout: Dropout比例，默认为0。
        """
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for index in range(layers):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=hidden_size, dropout=dropout),
                    FeedForward(dim, mlp_size, dropout=dropout)
                ])
            )

    def forward(self, x: Tensor):
        """
        前向传播过程。

        :param x: 输入的张量。
        :return: 处理后的张量。
        """

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x) + x
        return self.norm(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    def __init__(
            self, *,
            image_size: tuple[int, int] | int,
            patch_size: tuple[int, int] | int,
            num_classes: int,
            dim: int,
            layers: int,
            heads: int,
            hidden_size: int,
            mlp_size: int,
            pool: str = 'cls',
            channels: int = 3,
            dropout: float = 0.,
            emb_dropout: float = 0.
    ):
        """
        初始化视觉Transformer模型。

        :param image_size: 输入图像的尺寸，可以是单个整数（正方形图像）或包含两个整数的元组（长宽）。
        :param patch_size: 将图像切割成的小块的尺寸，可以是单个整数（正方形小块）或包含两个整数的元组（长宽）。
        :param num_classes: 分类任务的目标类别数。
        :param dim: Transformer编码器的维度。
        :param layers: Transformer编码器的层数。
        :param heads: Transformer编码器中多头注意力的头数。
        :param hidden_size: Transformer编码器中FeedForward层的隐藏尺寸。
        :param mlp_size: Transformer编码器中多层感知器（MLP）的尺寸。
        :param pool: 对编码器输出进行池化的方法，'cls'表示使用CLS标记，'mean'表示使用平均值。
        :param channels: 图像的通道数，默认为3（RGB图像）。
        :param dropout: Transformer编码器中应用的dropout比例。
        :param emb_dropout: 嵌入层中应用的dropout比例。
        """
        super(ViT, self).__init__()

        # 计算图像切割后的小块数量和每个小块的维度
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 确保图像尺寸能被小块尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Error image size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # 确保池化方法的有效性
        assert pool in {'cls', 'mean'}, 'err pool type'

        # 定义从图像到patch嵌入的转换过程
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 初始化位置嵌入和分类token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化Transformer编码器
        self.transform = Transformer(dim, layers, heads, hidden_size, mlp_size, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # 定义最终的线性分类器
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img: Tensor):
        """
        前向传播过程。

        :param img: 输入的图像张量。
        :return: 经过模型处理后得到的类别概率分布。
        """
        x: Tensor = self.to_patch_embedding(img)  # 将图像转换为patch嵌入
        b, n, _ = x.shape

        # 添加分类token并添加位置嵌入
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 通过Transformer编码器
        x = self.transform(x)

        # 根据配置进行池化
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ChannelSelection(nn.Module):
    def __init__(self, num_channels):
        super(ChannelSelection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        output = input_tensor.mul(self.indexes)
        return output


def sparse_selection(net: nn.Module):
    s = 1e-4
    for m in net.modules():
        if isinstance(m, ChannelSelection):
            m.indexes.grad.data.add_(s * torch.sign(m.indexes.data))


def train(
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        schedule: torch.optim.lr_scheduler.ReduceLROnPlateau,
        accelerator: Accelerator,
        epoch: int,
        train_loader: DataLoader,
        writer: SummaryWriter,
        metric: EvaluationModule
):
    """
    训练给定的网络模型。

    :param net: 要训练的网络，继承自nn.Module。
    :param optimizer: 用于优化网络参数的优化器，来自torch.optim。
    :param schedule: 调度器，更新学习率
    :param accelerator: 分布式计算对象
    :param epoch: 当前训练的轮次。
    :param train_loader: 训练数据的加载器，来自torch.utils.data.DataLoader。
    :param writer: 用于记录训练过程数据的SummaryWriter，常用于TensorBoard。
    :param metric: 模型评估pipline
    :return: 无返回值。
    """
    # 初始化损失函数
    criterion = nn.CrossEntropyLoss()
    # 将网络设置为训练模式
    net.train()
    # 初始化训练过程的损失和正确率等统计
    train_loss = []

    # 开始训练循环
    for batch_idx, (inputs, targets) in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Epoch {epoch}({accelerator.device})',
            disable=not accelerator.is_local_main_process
    ):
        inputs: Tensor
        targets: Tensor

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        accelerator.backward(loss)
        sparse_selection(net)
        optimizer.step()

        train_loss.append(loss.item())
        _, predicted = outputs.max(1)

        global_predictions, global_targets = accelerator.gather_for_metrics((predicted, targets))
        metric.add_batch(predictions=global_predictions, references=global_targets)

    global_train_loss = accelerator.gather_for_metrics(train_loss)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        result = metric.compute()

        schedule.step(np.mean(global_train_loss), epoch)
        writer.add_scalar('lr', schedule.get_last_lr()[0], epoch)

        writer.add_scalar('Train/loss', np.mean(global_train_loss), epoch)
        writer.add_scalar('Train/acc', 100. * result['accuracy'], epoch)

        writer.flush()


def test(
        net: nn.Module,
        accelerator: Accelerator,
        epoch: int,
        test_loader: DataLoader,
        writer: SummaryWriter,
        metric: CombinedEvaluations,
        best_acc: float,
        model_name: str = '',
        _labels: list = None,
) -> int:
    """
    测试给定网络的性能。

    :param net: 要测试的网络，是一个nn.Module的子类。
    :param accelerator: 分布式计算对象
    :param epoch: 当前的训练轮次，用于记录测试结果。
    :param test_loader: 测试数据集的数据加载器。
    :param writer: 用于写入TensorBoard日志的SummaryWriter对象。
    :param metric: 模型评估pipline
    :param best_acc: 全局最优ACC，用于保存模型
    :param model_name: 保存模型的名称
    :param _labels: 类别的标签，用于可视化
    :return: 更新后的全局最优ACC
    """

    criterion = nn.CrossEntropyLoss()

    net.eval()  # 将网络设置为评估模式
    test_loss = []
    with torch.no_grad():  # 禁止计算梯度
        for batch_idx, (inputs, targets) in tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f'Test {epoch}({accelerator.device})',
                disable=not accelerator.is_local_main_process
        ):
            inputs: Tensor
            targets: Tensor

            outputs = net(inputs)  # 网络前向传播

            loss = criterion(outputs, targets)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)

            global_predictions, global_targets = accelerator.gather_for_metrics((predicted, targets))
            metric.add_batch(predictions=global_predictions, references=global_targets)

        global_test_loss = accelerator.gather_for_metrics(test_loss)

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            result = metric.compute()

            writer.add_scalar('Test/loss', np.mean(global_test_loss), epoch)
            new_acc = 100. * result['accuracy']
            writer.add_scalar('Test/acc', new_acc, epoch)

            if best_acc < new_acc:
                if _labels is not None:
                    fig, ax = plt.subplots()
                    sns.heatmap(
                        np.array(result['confusion_matrix']),
                        annot=True,
                        cmap='Blues',
                        fmt='d',
                        xticklabels=_labels,
                        yticklabels=_labels,
                        ax=ax
                    )
                    ax.set_xlabel('Predicted label')
                    ax.set_ylabel('True label')
                    writer.add_figure('', fig, epoch)
                    plt.close()

                best_acc = new_acc
                torch.save(net.state_dict(), f'models/{model_name}.pth')

            writer.flush()
            return best_acc
