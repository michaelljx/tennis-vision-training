#!/usr/bin/env python3
"""
export_coreml.py — Export CTR-GCN to CoreML INT8 for iOS deployment.

Flow: PyTorch weights → TorchScript (trace) → CoreML FP32 → INT8 weight quantization

Usage:
    python tools/export_coreml.py
    python tools/export_coreml.py --model models/ctrgcn_best.pth --out models/ctrgcn_stroke_int8.mlpackage

Requirements:
    pip install coremltools>=7.0

Input:  models/ctrgcn_best.pth  (trained on Kaggle — download from Output tab)
Output: models/ctrgcn_stroke_int8.mlpackage
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# Ensure ctrgcn module is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ctrgcn.model import CTRGCN

CLASS_NAMES = ['forehand', 'backhand', 'serve', 'invalid']
N_CLASSES = 4


# ── Reproduce training-time wrapper ──────────────────────────────────────────
class CTRGCNWithDropout(nn.Module):
    """
    Wrapper used during training (see kaggle_train.ipynb Cell 12).
    Must match exactly so the state dict loads correctly.

    State dict layout:
      base.*     — CTRGCN backbone (fc replaced with Identity)
      fc.*       — actual Linear(256, 4) classifier
    """
    def __init__(self, base_model: CTRGCN, dropout: float = 0.5):
        super().__init__()
        self.base = base_model
        self.drop = nn.Dropout(dropout)
        self.fc   = base_model.fc           # save the Linear(256, 4)
        base_model.fc = nn.Identity()       # replace backbone fc with Identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base(x)    # (N, 256)
        feat = self.drop(feat)
        return self.fc(feat)   # (N, 4)


def build_model() -> CTRGCNWithDropout:
    base = CTRGCN(num_class=N_CLASSES, base_channels=32, adaptive=True)
    model = CTRGCNWithDropout(base, dropout=0.5)
    return model


def load_model(pth_path: str) -> CTRGCNWithDropout:
    model = build_model()
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded: {pth_path}  ({n_params:,} params)')
    return model


def verify_inference(model: CTRGCNWithDropout) -> None:
    """Smoke-test: run one forward pass, check output shape and logit range."""
    dummy = torch.zeros(1, 3, 45, 17, 1)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, N_CLASSES), f'Unexpected output shape: {out.shape}'
    probs = torch.softmax(out, dim=1)
    print(f'Smoke test  logits: {out[0].tolist()}')
    print(f'            probs:  {[f"{p:.3f}" for p in probs[0].tolist()]}')
    print(f'            argmax: {CLASS_NAMES[out.argmax(1).item()]}')


def trace_model(model: CTRGCNWithDropout) -> torch.jit.ScriptModule:
    """TorchScript trace — required for coremltools conversion."""
    dummy = torch.zeros(1, 3, 45, 17, 1)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    print('TorchScript trace: OK')
    return traced


def convert_to_coreml(traced, out_path: str) -> None:
    """Convert TorchScript → CoreML FP32 → INT8 weight quantization → save."""
    try:
        import coremltools as ct
        import coremltools.optimize.coreml as cto
    except ImportError:
        print('\nERROR: coremltools not installed.')
        print('Run: pip install coremltools>=7.0')
        sys.exit(1)

    print(f'\ncoremltools version: {ct.__version__}')

    # ── FP32 CoreML conversion ────────────────────────────────────────────
    print('Converting TorchScript → CoreML FP32 ...')
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name='skeleton',
            shape=(1, 3, 45, 17, 1),
            dtype=np.float32,
        )],
        classifier_config=ct.ClassifierConfig(CLASS_NAMES),
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to='mlprogram',
    )

    # Metadata
    mlmodel.short_description = 'Tennis stroke classifier (CTR-GCN 4-class)'
    mlmodel.input_description['skeleton'] = (
        'Skeleton window: (1, C=3, T=45, V=17, M=1) float32. '
        'C channels: x (normalized 0-1), y (normalized 0-1), confidence (0-1). '
        'Joints in COCO-17 order.'
    )
    mlmodel.output_description['classLabel'] = 'Predicted stroke class'
    try:
        mlmodel.output_description['classProbability'] = 'Per-class softmax probabilities'
    except AttributeError:
        pass  # output name varies by coremltools version

    # Model version info
    import json
    meta_path = os.path.join(REPO_ROOT, 'models', 'ctrgcn_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        val_acc = meta.get('val_acc', 'unknown')
        mlmodel.version = f"1.0-acc{val_acc}"
    else:
        mlmodel.version = '1.0'

    # Save FP32 first (for debugging if INT8 fails)
    fp32_path = out_path.replace('_int8', '_fp32')
    mlmodel.save(fp32_path)
    fp32_size = _dir_size_mb(fp32_path)
    print(f'Saved FP32: {fp32_path}  ({fp32_size:.1f} MB)')

    # ── INT8 weight quantization ──────────────────────────────────────────
    print('\nApplying INT8 weight quantization ...')
    op_config = cto.OpLinearQuantizerConfig(
        mode='linear_symmetric',
        dtype='int8',
        granularity='per_channel',
    )
    config = cto.OptimizationConfig(global_config=op_config)
    mlmodel_int8 = cto.linear_quantize_weights(mlmodel, config=config)

    mlmodel_int8.save(out_path)
    int8_size = _dir_size_mb(out_path)
    print(f'Saved INT8: {out_path}  ({int8_size:.1f} MB)')
    print(f'Size reduction: {fp32_size:.1f} → {int8_size:.1f} MB  ({(1-int8_size/fp32_size)*100:.0f}% smaller)')


def _dir_size_mb(path: str) -> float:
    """Recursively sum directory size in MB."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def verify_coreml(mlpackage_path: str) -> None:
    """Load saved model and verify input/output spec."""
    try:
        import coremltools as ct
    except ImportError:
        return

    m = ct.models.MLModel(mlpackage_path)
    print(f'\n=== CoreML Model Spec ===')
    print(f'Input:  {m.input_description._fd_spec}')
    print(f'Output: {m.output_description._fd_spec}')

    # Run prediction
    dummy = np.zeros((1, 3, 45, 17, 1), dtype=np.float32)
    pred = m.predict({'skeleton': dummy})
    print(f'\nTest prediction:')
    print(f'  classLabel:       {pred["classLabel"]}')
    # coremltools 9.x names the probability output "classLabel_probs"
    probs = pred.get('classLabel_probs') or pred.get('classProbability') or {}
    for cls in CLASS_NAMES:
        print(f'  {cls:<12} {probs.get(cls, 0):.4f}')
    # Print actual output keys for iOS implementation reference
    prob_key = next((k for k in pred if k != 'classLabel'), None)
    print(f'\niOS output key for probabilities: "{prob_key}"')


def main():
    parser = argparse.ArgumentParser(description='Export CTR-GCN to CoreML INT8')
    parser.add_argument('--model', default='models/ctrgcn_best.pth',
                        help='Path to trained .pth file (default: models/ctrgcn_best.pth)')
    parser.add_argument('--out', default='models/ctrgcn_stroke_int8.mlpackage',
                        help='Output .mlpackage path')
    parser.add_argument('--no-int8', action='store_true',
                        help='Skip INT8 quantization (export FP32 only)')
    args = parser.parse_args()

    # Resolve paths relative to repo root
    pth_path = os.path.join(REPO_ROOT, args.model) if not os.path.isabs(args.model) else args.model
    out_path = os.path.join(REPO_ROOT, args.out)   if not os.path.isabs(args.out)   else args.out

    if not os.path.exists(pth_path):
        print(f'\nERROR: Model file not found: {pth_path}')
        print()
        print('Download ctrgcn_best.pth from Kaggle:')
        print('  1. Open your Kaggle notebook')
        print('  2. Go to Output tab → ctrgcn_best.pth → Download')
        print(f'  3. Place it at: {pth_path}')
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print('=' * 55)
    print('CTR-GCN → CoreML INT8 Export')
    print(f'Classes: {CLASS_NAMES}')
    print('=' * 55)

    # 1. Load weights
    model = load_model(pth_path)

    # 2. Verify inference
    print('\n--- Inference smoke test ---')
    verify_inference(model)

    # 3. TorchScript trace
    print('\n--- TorchScript trace ---')
    traced = trace_model(model)

    # 4. CoreML conversion + INT8
    if args.no_int8:
        print('\n--no-int8 flag: skipping INT8, exporting FP32 only')
        out_path = out_path.replace('_int8', '_fp32')

    print('\n--- CoreML conversion ---')
    convert_to_coreml(traced, out_path)

    # 5. Verify the saved model
    print('\n--- Verify saved model ---')
    verify_coreml(out_path)

    print(f'\n✓ Export complete: {out_path}')
    print()
    print('Next: copy to iOS app')
    print(f'  cp -r {out_path} <ios-app>/ios/ACE/CV/Models/ctrgcn_stroke_int8.mlpackage')


if __name__ == '__main__':
    main()
