import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from einops import rearrange
from rep_check.utils.graph import MediaPipeGraph


class TemporalGraphConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        t_kernel_size: Optional[int] = 1,
        t_stride: Optional[int] = 1,
        t_padding: Optional[int] = 0,
        t_dilation: Optional[int] = 1,
        bias: Optional[bool] = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            stride=(t_stride, 1),
            padding=(t_padding, 0),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x: Tensor, A: Tensor):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = rearrange(x, 'b (k c) t v -> b k c t v', k=self.kernel_size)
        x = torch.einsum('bkctv, kvw -> bctw', (x, A))
        return x, A
    

class STGCNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
        residual: Optional[bool] = True
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = TemporalGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[1]
        )
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, A: Tensor):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.activation(self.tcn(x) + res)
        return x, A
    

class STGCN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        graph: MediaPipeGraph,
        edge_weights: Optional[bool] = True,
        dropout: Optional[float] = 0.0
    ):
        super().__init__()
        # Adjacency matrix
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # GCN layers
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.input_norm = nn.BatchNorm1d(in_channels * A.size(1))
        self.blocks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False),
            STGCNBlock(64, 64, kernel_size, 1, dropout),
            STGCNBlock(64, 128, kernel_size, 2, dropout),
            STGCNBlock(128, 256, kernel_size, 2, dropout),
        ))
        # Initialize edge weights
        if edge_weights:
            self.edge_weights = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in range(len(self.blocks))
            ])
        else:
            self.edge_weights = [1.0] * len(self.blocks)
        # Final layer
        self.fcn = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: Tensor):
        """
        Args:
            x: graph sequence (batch_size, in_channels, timesteps, joints)
        """
        # Input normalization
        in_channels = x.shape[1]
        x = rearrange(x, 'b c t v -> b (v c) t')
        x = self.input_norm(x)
        x = rearrange(x, 'b (v c) t -> b c t v', c=in_channels)
        # Forward
        for block, weight in zip(self.blocks, self.edge_weights):
            x, _ = block(x, self.A * weight)
        # Global pooling
        x = F.avg_pool2d(x, x.shape[2:])
        # Output head
        x = self.fcn(x).flatten(1)
        return x