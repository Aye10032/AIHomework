import os

import torch
import torch.distributed as dist
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
            device: list[torch.device],
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

        self.device = device
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim).to(device[-1])

        self.change_layer = layers // 2

        for index in range(layers):
            if index <= self.change_layer:
                self.layers.append(
                    nn.ModuleList([
                        Attention(dim, heads=heads, dim_head=hidden_size, dropout=dropout),
                        FeedForward(dim, mlp_size, dropout=dropout)
                    ]).to(device[0])
                )
            else:
                self.layers.append(
                    nn.ModuleList([
                        Attention(dim, heads=heads, dim_head=hidden_size, dropout=dropout),
                        FeedForward(dim, mlp_size, dropout=dropout)
                    ]).to(device[-1])
                )

    def forward(self, x: Tensor):
        """
        前向传播过程。

        :param x: 输入的张量。
        :return: 处理后的张量。
        """

        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x)
            x = ff(x) + x
            if index == self.change_layer:
                x = x.to(self.device[-1])
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
            device: list[torch.device] = None,
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

        if device is None:
            self.device = [torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')]
        else:
            self.device = device

        # 定义从图像到patch嵌入的转换过程
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        ).to(device[0])

        # 初始化位置嵌入和分类token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim).to(device[0]))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim).to(device[0]))
        self.dropout = nn.Dropout(emb_dropout).to(device[0])

        # 初始化Transformer编码器
        self.transform = Transformer(dim, layers, heads, hidden_size, mlp_size, device, dropout)

        self.pool = pool
        self.to_latent = nn.Identity().to(self.device[-1])

        # 定义最终的线性分类器
        self.mlp_head = nn.Linear(dim, num_classes).to(self.device[-1])

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
        schedule: torch.optim.lr_scheduler.CosineAnnealingLR,
        device: list[torch.device],
        epoch: int,
        train_loader: DataLoader,
        writer: SummaryWriter,
        ddp: bool = False
):
    """
    训练给定的网络模型。

    :param net: 要训练的网络，继承自nn.Module。
    :param optimizer: 用于优化网络参数的优化器，来自torch.optim。
    :param schedule: 学习率调度器，用于动态调整学习率，此处为CosineAnnealingLR。
    :param device: 指定训练时使用的设备列表。
    :param epoch: 当前训练的轮次。
    :param train_loader: 训练数据的加载器，来自torch.utils.data.DataLoader。
    :param writer: 用于记录训练过程数据的SummaryWriter，常用于TensorBoard。
    :param ddp: 是否使用分布式数据并行训练，默认为False。
    :return: 无返回值。
    """
    # 初始化损失函数
    criterion = nn.CrossEntropyLoss()
    # 将网络设置为训练模式
    net.train()
    # 初始化训练过程的损失和正确率等统计
    train_loss = 0
    correct = 0
    total = 0
    # 初始化用于TensorBoard嵌入展示的张量
    # output_embed = torch.empty((0, 10))
    # target_embeds = torch.empty(0)

    # 如果使用分布式数据并行，更新训练数据采样器的epoch
    if ddp:
        train_loader.sampler.set_epoch(epoch)

        # 设置进度条，用于显示训练进度和性能指标
        if dist.get_rank() == 0:
            loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
        else:
            loop = enumerate(train_loader)
    else:
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')

    # 开始训练循环
    for batch_idx, (inputs, targets) in loop:
        inputs: Tensor
        targets: Tensor

        # 数据移至指定设备
        inputs, targets = inputs.to(device[0]), targets.to(device[-1])
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, targets)
        # 反向传播
        loss.backward()
        # 执行特定的稀疏选择操作
        sparse_selection(net)
        # 更新网络参数
        optimizer.step()
        # 更新学习率
        if schedule is not None:
            schedule.step()

        # 更新训练过程的统计量
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 如果是主进程，则更新进度条显示
        if ddp:
            if dist.get_rank() == 0:
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total)
        else:
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total)

        # 前几个批次，收集输出和目标以用于TensorBoard的嵌入展示
        # if batch_idx <= 3:
        #     output_embed = torch.cat((output_embed, outputs.clone().cpu()), 0)
        #     target_embeds = torch.cat((target_embeds, targets.data.clone().cpu()), 0)

    # 记录当前学习率
    if schedule is not None:
        writer.add_scalar('lr', schedule.get_last_lr()[0], epoch)

    # 如果是主进程，记录训练细节到TensorBoard
    if ddp:
        if dist.get_rank() != 0:
            return

    # 每隔一定轮次，记录嵌入到TensorBoard
    # if epoch % 9 == 0:
    #     writer.add_embedding(
    #         output_embed,
    #         metadata=target_embeds,
    #         global_step=epoch + 1,
    #         tag='cifar10'
    #     )

    # 记录训练损失和准确率
    writer.add_scalar('Train/loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('Train/acc', 100. * correct / total, epoch)

    # 记录每层参数的直方图
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy(), epoch)

    # 刷新writer，确保数据写入
    writer.flush()


def test(
        net: nn.Module,
        device: list[torch.device],
        epoch: int,
        test_loader: DataLoader,
        writer: SummaryWriter,
        ddp: bool = False
):
    """
    测试给定网络的性能。

    :param net: 要测试的网络，是一个nn.Module的子类。
    :param device: 用于测试的设备列表，第一个设备用于加载数据，最后一个设备用于计算。
    :param epoch: 当前的训练轮次，用于记录测试结果。
    :param test_loader: 测试数据集的数据加载器。
    :param writer: 用于写入TensorBoard日志的SummaryWriter对象。
    :param ddp: 是否使用分布式数据并行训练。默认为False。
    :return: 无返回值。
    """
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    net.eval()  # 将网络设置为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总预测数
    with torch.no_grad():  # 禁止计算梯度
        # 根据是否使用DDP和当前进程排名，选择不同的进度条更新方式
        if ddp:
            if dist.get_rank() == 0:
                loop = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test {epoch}')
            else:
                loop = enumerate(test_loader)
        else:
            loop = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test {epoch}')

        for batch_idx, (inputs, targets) in loop:
            inputs: Tensor
            targets: Tensor

            # 数据移动到指定设备
            inputs, targets = inputs.to(device[0]), targets.to(device[-1])
            outputs = net(inputs)  # 网络前向传播
            loss = criterion(outputs, targets)  # 计算损失

            test_loss += loss.item()  # 累加测试损失
            _, predicted = outputs.max(1)  # 获取预测类别
            total += targets.size(0)  # 累加总样本数
            correct += predicted.eq(targets).sum().item()  # 累加正确预测数

            # 如果是主进程或未使用DDP，更新进度条显示
            if ddp:
                if dist.get_rank() == 0:
                    loop.set_postfix(loss=test_loss / (batch_idx + 1), acc=100. * correct / total)
            else:
                loop.set_postfix(loss=test_loss / (batch_idx + 1), acc=100. * correct / total)

        # 如果是主进程或未使用DDP，记录测试损失和准确率到TensorBoard
        if ddp and dist.get_rank() != 0:
            return

        writer.add_scalar('Test/loss', test_loss / (batch_idx + 1), epoch)
        writer.add_scalar('Test/acc', 100. * correct / total, epoch)
        writer.flush()  # 刷新TensorBoard日志
