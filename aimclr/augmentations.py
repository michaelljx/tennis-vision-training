"""
augmentations.py — Skeleton-specific data augmentations for AimCLR contrastive learning.

All functions operate on tensors of shape (C=3, T=45, V=17):
  C[0] = x_model, C[1] = y_model, C[2] = confidence
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor


# Left/right joint swap pairs for mirror_horizontal (0-indexed in JOINT_NAMES)
# ["nose","left_eye","right_eye","left_ear","right_ear",
#  "left_shoulder","right_shoulder","left_elbow","right_elbow",
#  "left_wrist","right_wrist","left_hip","right_hip",
#  "left_knee","right_knee","left_ankle","right_ankle"]
SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def random_rotation_2d(x: Tensor, max_angle: float = 15.0) -> Tensor:
    """
    Rotate x,y coords around the center (0.5, 0.5) by a random angle in [-max_angle, max_angle] degrees.
    Channel 2 (confidence) is unchanged.
    """
    angle = (torch.rand(1).item() * 2 - 1) * max_angle  # degrees in [-max, max]
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    out = x.clone()
    cx = out[0] - 0.5  # center
    cy = out[1] - 0.5
    out[0] = cos_t * cx - sin_t * cy + 0.5
    out[1] = sin_t * cx + cos_t * cy + 0.5
    return out


def random_shear(x: Tensor, magnitude: float = 0.1) -> Tensor:
    """Apply random shear transform to x,y coords."""
    sx = (torch.rand(1).item() * 2 - 1) * magnitude
    sy = (torch.rand(1).item() * 2 - 1) * magnitude

    out = x.clone()
    ox, oy = out[0].clone(), out[1].clone()
    out[0] = ox + sx * oy
    out[1] = sy * ox + oy
    return out


def temporal_crop_resize(x: Tensor, min_ratio: float = 0.7) -> Tensor:
    """
    Randomly crop a contiguous temporal sub-sequence and resize back to T frames
    via linear interpolation.  Shape (C, T, V) → (C, T, V).
    """
    C, T, V = x.shape
    ratio = min_ratio + torch.rand(1).item() * (1.0 - min_ratio)
    crop_len = max(2, int(T * ratio))
    start = torch.randint(0, T - crop_len + 1, (1,)).item()

    # Crop: (C, crop_len, V)
    cropped = x[:, start : start + crop_len, :]

    # Resize back to T: treat as (1, C*V, crop_len) for 1D interpolation
    reshaped = cropped.reshape(1, C * V, crop_len)
    resized  = F.interpolate(reshaped, size=T, mode="linear", align_corners=False)
    return resized.reshape(C, T, V)


def joint_dropout(x: Tensor, drop_prob: float = 0.1) -> Tensor:
    """
    Randomly zero out entire joints (all time steps, all channels).
    Mimics occlusion.  Nose (index 0) is never dropped.
    """
    out = x.clone()
    V = x.shape[2]
    for v in range(1, V):  # skip nose
        if torch.rand(1).item() < drop_prob:
            out[:, :, v] = 0.0
    return out


def gaussian_noise(x: Tensor, std: float = 0.01) -> Tensor:
    """Add Gaussian noise to x,y coords.  Confidence channel unchanged."""
    out = x.clone()
    noise = torch.randn_like(out[:2]) * std
    out[:2] = out[:2] + noise
    return out


def bone_length_perturbation(x: Tensor, std: float = 0.05) -> Tensor:
    """
    Scale each joint's displacement from joint 0 (nose) by a per-joint random factor.
    Simulates body shape variation.
    """
    out = x.clone()
    V = x.shape[2]
    for v in range(1, V):
        scale = 1.0 + torch.randn(1).item() * std
        out[:2, :, v] = out[:2, :, 0] + scale * (out[:2, :, v] - out[:2, :, 0])
    return out


def mirror_horizontal(x: Tensor) -> Tensor:
    """
    Flip x-coord (channel 0) around 0.5, then swap left/right joint indices.
    """
    out = x.clone()
    out[0] = 1.0 - out[0]  # flip x
    for i, j in SWAP_PAIRS:
        out[:, :, i], out[:, :, j] = out[:, :, j].clone(), out[:, :, i].clone()
    return out


def compose_augmentation(x: Tensor, strong: bool = True) -> Tensor:
    """
    Apply a random composition of augmentations to a (C, T, V) skeleton tensor.
    Each augmentation is applied independently with its own probability.
    """
    if torch.rand(1).item() < 0.5:
        x = random_rotation_2d(x, max_angle=15.0)
    if torch.rand(1).item() < 0.5:
        x = temporal_crop_resize(x, min_ratio=0.7)
    if torch.rand(1).item() < 0.5:
        x = gaussian_noise(x, std=0.01)
    if torch.rand(1).item() < 0.5:
        x = joint_dropout(x, drop_prob=0.1)
    if torch.rand(1).item() < 0.5:
        x = mirror_horizontal(x)
    if torch.rand(1).item() < 0.3:
        x = random_shear(x, magnitude=0.1)
    if strong and torch.rand(1).item() < 0.3:
        x = bone_length_perturbation(x, std=0.05)
    return x
