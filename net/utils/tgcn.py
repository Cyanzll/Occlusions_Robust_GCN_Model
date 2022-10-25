# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        # 3
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            # 3
            in_channels,
            # 64 * 3 (k * c, kc)
            out_channels * kernel_size,
            # (1, 1)
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        # 断言：保证A的第一个维度（邻接矩阵个数）与卷积核大小是一致的
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        # 经过 1 * 1 卷积后，改变的只有通道数量（c -> k * c），相当于升维
        n, kc, t, v = x.size()
        # 形状变为 n(N*M), k, c, t, v
        # 到这一步。形状变成 (512, 3, 64, 150, 18)，A的形状是(3, 18, 18)
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)

        # 这一步运算不涉及时间维度运算，可以理解为对节点信息按照邻接矩阵进行聚合
        # 每个节点信息具有64个维度，可以认为是一个 18（节点数量） * 64 的节点特征矩阵分别与3个邻接矩阵相乘
        # 运算后得到的张量是 n, c, t, v （w为节点数量v，用w只是为了避免重复）
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
