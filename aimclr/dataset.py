"""
dataset.py — PyTorch Dataset for AimCLR self-supervised pretraining.

Loads unlabeled.npz (shape M×45×17×3) and returns two independently augmented
views of the same sample for contrastive learning.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from aimclr.augmentations import compose_augmentation


class UnlabeledSkeletonDataset(Dataset):
    """
    Dataset for AimCLR pretraining.

    Each __getitem__ returns (view1, view2): two augmented views of the same
    45-frame skeleton window.

    Each view has shape (C=3, T=45, V=17).
    The CTR-GCN backbone expects (N, C, T, V, M=1), so callers should add the M dim:
        view1 = view1.unsqueeze(-1)  # (C, T, V, 1)
    """

    def __init__(
        self,
        npz_path: str,
        augment_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        data = np.load(npz_path)
        self.X = data["X"]  # (M, T, V, C)
        self.augment_fn = augment_fn if augment_fn is not None else compose_augmentation

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = torch.from_numpy(self.X[idx])   # (T, V, C)
        x = x.permute(2, 0, 1).float()      # (C, T, V)

        view1 = self.augment_fn(x.clone())  # (C, T, V)
        view2 = self.augment_fn(x.clone())  # (C, T, V)
        return view1, view2


class LabeledSkeletonDataset(Dataset):
    """
    Dataset for supervised training (Phase 5).
    Loads train.npz or val.npz and optionally applies augmentation.

    Each __getitem__ returns (x, label):
        x:     (C=3, T=45, V=17) tensor
        label: int64 scalar (0=forehand, 1=backhand, 2=serve, 3=volley, 4=invalid)
    """

    def __init__(
        self,
        npz_path: str,
        augment_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]  # (N, T, V, C)
        self.y = data["y"]  # (N,)
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = torch.from_numpy(self.X[idx]).float()  # (T, V, C)
        x = x.permute(2, 0, 1)                     # (C, T, V)
        if self.augment_fn is not None:
            x = self.augment_fn(x)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, label
