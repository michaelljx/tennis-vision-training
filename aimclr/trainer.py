"""
trainer.py — AimCLR self-supervised pretraining loop.

Wraps CTR-GCN backbone + projection head + InfoNCE loss into a
SimCLR-style contrastive training workflow.

After training, only the backbone state_dict is saved (projector is discarded).
The backbone can then be loaded in Phase 5 supervised fine-tuning.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader


class AimCLRTrainer:
    """
    AimCLR / SimCLR-style contrastive pretrainer.

    Args:
        backbone:      CTR-GCN model (or any model with extract_features method)
        feature_dim:   backbone output dimension (default 256)
        proj_dim:      projection head output dimension (default 128)
        temperature:   InfoNCE temperature (default 0.1)
        lr:            Adam learning rate (default 1e-3)
        weight_decay:  Adam weight decay (default 1e-4)
        device:        "cuda" or "cpu"
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda",
    ):
        self.backbone = backbone.to(device)
        self.device   = device

        # Projection head (discarded after pretraining)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, proj_dim),
        ).to(device)

        from aimclr.loss import InfoNCELoss
        self.criterion = InfoNCELoss(temperature=temperature)

        params = list(backbone.parameters()) + list(self.projector.parameters())
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one epoch. Returns average InfoNCE loss."""
        self.backbone.train()
        self.projector.train()

        total_loss = 0.0
        n_batches  = 0

        for view1, view2 in loader:
            # view1, view2: (N, C, T, V)
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)

            # Add person dimension M=1 for CTR-GCN
            v1 = view1.unsqueeze(-1)  # (N, C, T, V, 1)
            v2 = view2.unsqueeze(-1)

            # Encode both views
            z1 = self.projector(self.backbone.extract_features(v1))  # (N, proj_dim)
            z2 = self.projector(self.backbone.extract_features(v2))

            # L2 normalize
            z1 = nn.functional.normalize(z1, dim=1)
            z2 = nn.functional.normalize(z2, dim=1)

            loss = self.criterion(z1, z2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(1, n_batches)

    def train(
        self,
        loader: DataLoader,
        epochs: int = 100,
        save_path: str | None = None,
        print_every: int = 10,
    ) -> list[float]:
        """
        Full training loop.

        Args:
            loader:      DataLoader yielding (view1, view2) pairs
            epochs:      number of training epochs
            save_path:   if given, saves backbone checkpoint here after training
            print_every: log interval in epochs

        Returns:
            list of per-epoch average losses
        """
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        losses = []
        print(f"Starting AimCLR pretraining: {epochs} epochs, {len(loader)} batches/epoch")
        print(f"Random baseline loss: log(2*{loader.batch_size}) ≈ {__import__('math').log(2*loader.batch_size):.3f}")

        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(loader)
            scheduler.step()
            losses.append(avg_loss)

            if epoch % print_every == 0 or epoch == 1:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  lr={lr:.2e}")

        if save_path:
            self.save_backbone(save_path)

        return losses

    def save_backbone(self, path: str) -> None:
        """Save only backbone weights (projector is discarded — not needed for Phase 5)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.backbone.state_dict(), path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Saved backbone: {path}  ({size_mb:.1f} MB)")
