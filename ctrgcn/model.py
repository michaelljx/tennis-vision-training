"""
model.py — CTR-GCN: Channel-wise Topology Refinement Graph Convolutional Network.

Reference: Chen et al., ICCV 2021, "Channel-wise Topology Refinement Graph Convolution
for Skeleton-Based Action Recognition"

Input:  (N, C=3, T=45, V=17, M=1)  — batch, channels, frames, joints, persons
Output: (N, num_class=5)  or  (N, 256) from extract_features()

Architecture:
  DataBatchNorm
  → 10 CTRGCNBlocks (3 stages: 64→128→256 channels)
  → Global Average Pool (T, V)
  → FC(256 → num_class)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from ctrgcn.graph import Graph


class TemporalConv(nn.Module):
    """
    Multi-scale temporal convolution — 3 parallel branches whose outputs are summed
    with learned blend weights (no channel split → no divisibility constraint).

      Branch 1: Conv2d kernel=(3,1)
      Branch 2: Conv2d kernel=(5,1)
      Branch 3: MaxPool2d(3,1) + Conv2d kernel=(1,1)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0),
                      stride=(stride, 1), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1), padding=(2, 0),
                      stride=(stride, 1), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Learned per-branch blend weights (softmax-normalized)
        self.alpha = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, T, V)
        a = F.softmax(self.alpha, dim=0)
        return a[0] * self.branch1(x) + a[1] * self.branch2(x) + a[2] * self.branch3(x)


class SpatialConv(nn.Module):
    """
    Channel-wise Topology Refinement GCN (core of CTR-GCN).

    Combines:
      A_base:  fixed normalized adjacency (3, V, V)
      B:       learnable global residual (3, V, V)
      C_dyn:   data-dependent per-sample V×V attention (computed from input)

    Total: A_eff[k] = A_base[k] + B[k] + C_dyn[sample, k]
    """

    def __init__(self, in_ch: int, out_ch: int, A: np.ndarray, adaptive: bool = True):
        super().__init__()
        K, V, _ = A.shape
        self.K = K
        self.V = V
        self.adaptive = adaptive

        self.register_buffer("A_base", torch.tensor(A.copy(), dtype=torch.float32))  # (K, V, V)

        if adaptive:
            self.B = nn.Parameter(torch.zeros(K, V, V))
            # Project input to K*V features for pairwise attention
            self.theta = nn.Conv2d(in_ch, K * V, kernel_size=(1, 1), bias=False)
            self.phi   = nn.Conv2d(in_ch, K * V, kernel_size=(1, 1), bias=False)

        # One conv per partition
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False)
            for _ in range(K)
        ])
        self.bn = nn.BatchNorm2d(out_ch)

    def _compute_C_dyn(self, x: Tensor) -> Tensor:
        """Compute data-dependent (N, K, V, V) attention from input x (N, C, T, V)."""
        N, C, T, V = x.shape
        K = self.K

        # Average over time → (N, C, V), then compute joint-level features
        x_avg = x.mean(dim=2, keepdim=True)  # (N, C, 1, V)

        # (N, K*V, 1, V) → squeeze T dim → (N, K*V, V)
        theta = self.theta(x_avg).squeeze(2)   # (N, K*V, V)
        phi   = self.phi(x_avg).squeeze(2)     # (N, K*V, V)

        # Reshape to (N*K, V, V) for batched matmul
        theta = theta.view(N * K, V, V)        # each row = feature for one joint
        phi   = phi.view(N * K, V, V)

        # V×V similarity: theta @ phi^T, softmax-normalized
        C_dyn = torch.softmax(theta.bmm(phi.transpose(1, 2)), dim=-1)  # (N*K, V, V)
        return C_dyn.view(N, K, V, V)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, T, V)
        N, C, T, V = x.shape

        if self.adaptive:
            C_dyn = self._compute_C_dyn(x)  # (N, K, V, V)

        out = 0
        for k in range(self.K):
            A_k = self.A_base[k]  # (V, V)
            if self.adaptive:
                # A_k: (N, V, V) via broadcast + per-sample C
                A_k = A_k + self.B[k] + C_dyn[:, k]  # (N, V, V)
                # Aggregate: sum over source joints v → target joint w
                xk = torch.einsum("nctv,nvw->nctw", x, A_k)
            else:
                xk = torch.einsum("nctv,vw->nctw", x, A_k)
            out = out + self.convs[k](xk)

        return self.bn(out)


class CTRGCNBlock(nn.Module):
    """Single CTR-GCN block: SpatialConv → ReLU → TemporalConv + residual → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        A: np.ndarray,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
    ):
        super().__init__()
        self.gcn = SpatialConv(in_ch, out_ch, A, adaptive=adaptive)
        self.tcn = TemporalConv(out_ch, out_ch, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda _: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        res = self.residual(x)
        x   = self.relu(self.gcn(x))
        x   = self.tcn(x)
        return self.relu(x + res)


class CTRGCN(nn.Module):
    """
    CTR-GCN for skeleton-based stroke classification.

    Input:  (N, C=3, T=45, V=17, M=1)
    Output: (N, num_class)  via forward()
            (N, 256)        via extract_features()
    """

    def __init__(
        self,
        num_class: int = 5,
        num_point: int = 17,
        num_person: int = 1,
        in_channels: int = 3,
        base_channels: int = 64,
        adaptive: bool = True,
    ):
        super().__init__()
        self.num_person  = num_person
        self.in_channels = in_channels
        self.num_point   = num_point

        graph = Graph(num_node=num_point)
        A = graph.get_adjacency()  # (3, V, V)

        # Input normalization over all joints × channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c = base_channels  # 64

        # 10 blocks: Stage 1 (64ch) → Stage 2 (128ch) → Stage 3 (256ch)
        # fmt: off
        self.blocks = nn.ModuleList([
            CTRGCNBlock(in_channels, c,     A, stride=1, residual=False, adaptive=adaptive),
            CTRGCNBlock(c,           c,     A, stride=1, adaptive=adaptive),
            CTRGCNBlock(c,           c,     A, stride=1, adaptive=adaptive),
            CTRGCNBlock(c,           c,     A, stride=1, adaptive=adaptive),
            CTRGCNBlock(c,           c * 2, A, stride=2, adaptive=adaptive),
            CTRGCNBlock(c * 2,       c * 2, A, stride=1, adaptive=adaptive),
            CTRGCNBlock(c * 2,       c * 2, A, stride=1, adaptive=adaptive),
            CTRGCNBlock(c * 2,       c * 4, A, stride=2, adaptive=adaptive),
            CTRGCNBlock(c * 4,       c * 4, A, stride=1, adaptive=adaptive),
            CTRGCNBlock(c * 4,       c * 4, A, stride=1, adaptive=adaptive),
        ])
        # fmt: on

        feat_dim = c * 4  # 256
        self.fc = nn.Linear(feat_dim, num_class)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _preprocess(self, x: Tensor) -> Tensor:
        """(N, C, T, V, M) → (N*M, C, T, V) after data batch norm."""
        N, C, T, V, M = x.shape

        # Permute to (N, M, V, C, T), flatten to (N*M, V*C, T) for BN
        x = x.permute(0, 4, 3, 1, 2).contiguous()   # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        return x

    def extract_features(self, x: Tensor) -> Tensor:
        """Return global average-pooled features (N, 256) without classification head."""
        N, C, T, V, M = x.shape
        x = self._preprocess(x)

        for block in self.blocks:
            x = block(x)

        # Global average pool over T and V → (N*M, feat_dim)
        x = x.mean(dim=(2, 3))
        x = x.view(N, M, -1).mean(dim=1)  # (N, feat_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass → logits (N, num_class)."""
        return self.fc(self.extract_features(x))
