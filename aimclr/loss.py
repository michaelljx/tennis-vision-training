"""
loss.py — InfoNCE contrastive loss (SimCLR / NT-Xent style).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """
    NT-Xent loss for SimCLR-style contrastive learning.

    Given two sets of embeddings z1 and z2 (both shape N×D, L2-normalized),
    positive pairs are (z1[i], z2[i]).  All other 2N-2 pairs are negatives.

    Loss value starts near log(2N) (random baseline) and decreases toward 0
    as representations become more discriminative.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Args:
            z1: (N, D) L2-normalized embeddings from view 1
            z2: (N, D) L2-normalized embeddings from view 2
        Returns:
            scalar loss
        """
        N = z1.shape[0]
        device = z1.device

        # Concatenate all embeddings: (2N, D)
        z = torch.cat([z1, z2], dim=0)

        # (2N, 2N) cosine similarity matrix (z already normalized)
        sim = z @ z.T / self.temperature

        # Mask out self-similarity on diagonal
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Positive pair indices:
        #   for z1[i] (row i), positive is z2[i] (row i+N)
        #   for z2[i] (row i+N), positive is z1[i] (row i)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N,     device=device),
        ])  # shape (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss
