import math

from torch import nn
import torch.nn.functional as F
import torch
from mmengine.model import BaseModule  #mmcv
from mmcv.cnn import build_activation_layer
__all__ =['MLCA','MSCASpatialAttention','MSCAAttention']










class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x=x * att_all
        return x








# Source from https://github.com/haoshao-nku/medical_seg
# 提出的注意力的实现模块
class MSCAAttention(BaseModule):
    """Multi-Scale Convolutional Attention(MSCA)模块.
    多尺度特征提取：通过多个卷积核大小和填充的卷积操作，以提取不同尺度的特征信息。
                这些卷积操作包括一个具有较大卷积核的初始卷积 (self.conv0) 和多个后续的卷积操作（self.conv0_1，self.conv0_2，self.conv1_1，self.conv1_2，self.conv2_1，self.conv2_2），每个都针对不同的核大小和填充。
    通道混合：在提取多尺度特征之后，通过对这些特征进行通道混合来整合不同尺度的信息。通道混合操作由最后一个卷积层 self.conv3 完成。
    卷积注意力：最后，通过将通道混合后的特征与输入特征进行逐元素乘法，实现了一种卷积注意力机制。这意味着模块通过对不同通道的特征赋予不同的权重来选择性地强调或抑制输入特征。
    """
    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        """

        :param channels: 通道数.
        :param kernel_sizes: 注意力核大小. 默认: [5, [1, 7], [1, 11], [1, 21]].
        :param paddings: 注意力模块中相应填充值的个数.
            默认: [2, [0, 3], [0, 5], [0, 10]].
        """
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        # 多尺度特征提取
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # 通道融合（也是通过1x1卷积）
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


# 原论文模型中带有封装MSCAAttention，可用于参考作者怎么使用这个注意力模块
class MSCASpatialAttention(BaseModule):
    """
    Spatial Attention Module in Multi-Scale Convolutional Attention Module，多尺度卷积注意力模块中的空间注意模块
    先过1x1卷积，gelu激活后过注意力，再过一次1x1卷积，最后和跳跃连接
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        """

        :param in_channels: 通道数.
        :param attention_kernel_sizes (list): 注意力核大小. 默认: [5, [1, 7], [1, 11], [1, 21]].
        :param attention_kernel_paddings (list): 注意力模块中相应填充值的个数.
        :param act_cfg (list): 注意力模块中相应填充值的个数.
        """
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 跳跃连接
        shorcut = x.clone()
        # 先过1x1卷积
        x = self.proj_1(x)
        # 激活
        x = self.activation(x)
        # 过MSCAAttention
        x = self.spatial_gating_unit(x)
        # 1x1卷积
        x = self.proj_2(x)
        # 残差融合
        x = x + shorcut
        return x

