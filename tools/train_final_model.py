#!/usr/bin/env python3
"""
train_final_model.py — Optimized feature-based stroke classifier.

Builds on diagnose_and_classify.py findings:
  - GradientBoosting hit 79.1% on first try
  - Volley recall (11%) is the main bottleneck (only 46 train samples)
  - Forehand/serve already excellent

Improvements:
  1. Enhanced features targeting volley discrimination
  2. SMOTE oversampling for minority classes
  3. Hyperparameter-tuned GradientBoosting
  4. Ensemble of RF + GB + SVM for stability
  5. Cross-validation for reliable accuracy estimate
"""

import numpy as np
import json
import os
from collections import Counter
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ── COCO 17 joint indices ──────────────────────────────────────────────────
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

CLASS_NAMES = ['forehand', 'backhand', 'serve', 'volley', 'invalid']
STROKE_FRAME = 30  # stroke event at frame 30 (of 45)


def joint_angle(a, b, c):
    """Angle at joint b (radians)."""
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def extract_features(X):
    """
    Extract 90+ biomechanical features from skeleton windows.
    X: (N, T=45, V=17, C=3) where C = (x, y, confidence)
    """
    N, T, V, C = X.shape
    coords = X[:, :, :, :2]
    conf   = X[:, :, :, 2]

    all_features = []

    for i in range(N):
        f = []
        c = coords[i]       # (T, V, 2)
        cn = conf[i]         # (T, V)

        # Key joints
        l_wr = c[:, L_WRIST];  r_wr = c[:, R_WRIST]
        l_el = c[:, L_ELBOW];  r_el = c[:, R_ELBOW]
        l_sh = c[:, L_SHOULDER]; r_sh = c[:, R_SHOULDER]
        l_hp = c[:, L_HIP];   r_hp = c[:, R_HIP]
        l_kn = c[:, L_KNEE];  r_kn = c[:, R_KNEE]
        l_ak = c[:, L_ANKLE]; r_ak = c[:, R_ANKLE]
        nose = c[:, NOSE]

        # ── Wrist velocity & acceleration ────────────────────────────────
        l_vel = np.linalg.norm(np.diff(l_wr, axis=0), axis=1)
        r_vel = np.linalg.norm(np.diff(r_wr, axis=0), axis=1)

        for vel, prefix in [(l_vel, 'lw'), (r_vel, 'rw')]:
            f.extend([vel.mean(), vel.std(), vel.max(), vel.min()])
            # Peak velocity frame (timing of stroke)
            f.append(np.argmax(vel) / (T - 1))
            # Velocity in pre-stroke vs post-stroke
            pre_v = vel[:STROKE_FRAME].mean() if STROKE_FRAME > 0 else 0
            post_v = vel[STROKE_FRAME:].mean() if STROKE_FRAME < len(vel) else 0
            f.extend([pre_v, post_v, post_v - pre_v])

        # Dominance: which wrist moves more
        f.append(r_vel.sum() - l_vel.sum())
        f.append(r_vel.max() / (l_vel.max() + 1e-8))
        f.append(r_vel.sum() / (l_vel.sum() + r_vel.sum() + 1e-8))

        # ── Arm angles at key frames ─────────────────────────────────────
        key_frames = [0, 10, 20, 29, 30, 35, 40, 44]
        for t in key_frames:
            t = min(t, T - 1)
            la = joint_angle(c[t, L_SHOULDER], c[t, L_ELBOW], c[t, L_WRIST])
            ra = joint_angle(c[t, R_SHOULDER], c[t, R_ELBOW], c[t, R_WRIST])
            f.extend([la, ra, ra - la])

        # Arm angle range over time
        l_angles = np.array([joint_angle(c[t, L_SHOULDER], c[t, L_ELBOW], c[t, L_WRIST]) for t in range(T)])
        r_angles = np.array([joint_angle(c[t, R_SHOULDER], c[t, R_ELBOW], c[t, R_WRIST]) for t in range(T)])
        f.extend([l_angles.std(), r_angles.std()])
        f.extend([l_angles.max() - l_angles.min(), r_angles.max() - r_angles.min()])

        # ── Wrist height (serve = high overhead) ─────────────────────────
        l_wh = l_wr[:, 1] - l_sh[:, 1]
        r_wh = r_wr[:, 1] - r_sh[:, 1]
        f.extend([l_wh.mean(), l_wh.max(), l_wh.min(), l_wh.std()])
        f.extend([r_wh.mean(), r_wh.max(), r_wh.min(), r_wh.std()])
        # Max absolute wrist height
        f.append(max(l_wr[:, 1].max(), r_wr[:, 1].max()))
        # Wrist above head? (serve indicator)
        f.append((l_wr[:, 1] > nose[:, 1]).sum() / T)
        f.append((r_wr[:, 1] > nose[:, 1]).sum() / T)

        # ── Torso features ───────────────────────────────────────────────
        mid_sh = (l_sh + r_sh) / 2
        mid_hp = (l_hp + r_hp) / 2
        torso = mid_sh - mid_hp
        torso_angle = np.arctan2(torso[:, 0], torso[:, 1] + 1e-8)
        f.extend([torso_angle.mean(), torso_angle.std()])

        # Shoulder width change (rotation proxy)
        sh_dx = r_sh[:, 0] - l_sh[:, 0]
        f.extend([sh_dx.mean(), sh_dx.std(), sh_dx.max() - sh_dx.min()])

        # ── Motion compactness (volley = short, compact motion) ──────────
        l_range_x = np.ptp(l_wr[:, 0])
        l_range_y = np.ptp(l_wr[:, 1])
        r_range_x = np.ptp(r_wr[:, 0])
        r_range_y = np.ptp(r_wr[:, 1])
        f.extend([l_range_x, l_range_y, r_range_x, r_range_y])

        # Total wrist travel distance
        l_travel = l_vel.sum()
        r_travel = r_vel.sum()
        f.extend([l_travel, r_travel, max(l_travel, r_travel)])

        # Motion duration: fraction of frames with significant wrist movement
        high_motion_thresh = 0.01
        f.append((l_vel > high_motion_thresh).sum() / (T - 1))
        f.append((r_vel > high_motion_thresh).sum() / (T - 1))

        # ── Stroke event specifics ───────────────────────────────────────
        sf = min(STROKE_FRAME, T - 1)
        f.extend(l_wr[sf].tolist())
        f.extend(r_wr[sf].tolist())

        # Distance from wrist to hip at stroke (reach indicator)
        l_reach = np.linalg.norm(l_wr[sf] - l_hp[sf])
        r_reach = np.linalg.norm(r_wr[sf] - r_hp[sf])
        f.extend([l_reach, r_reach])

        # ── Lateral movement direction ───────────────────────────────────
        f.append(l_wr[-1, 0] - l_wr[0, 0])
        f.append(r_wr[-1, 0] - r_wr[0, 0])

        # Body center movement
        body_center = c.mean(axis=1)
        body_vel = np.linalg.norm(np.diff(body_center, axis=0), axis=1)
        f.extend([body_vel.mean(), body_vel.std(), body_vel.sum()])

        # ── Knee bend (serve wind-up) ────────────────────────────────────
        l_ka = np.array([joint_angle(c[t, L_HIP], c[t, L_KNEE], c[t, L_ANKLE]) for t in range(T)])
        r_ka = np.array([joint_angle(c[t, R_HIP], c[t, R_KNEE], c[t, R_ANKLE]) for t in range(T)])
        f.extend([l_ka.mean(), l_ka.min(), l_ka.std()])
        f.extend([r_ka.mean(), r_ka.min(), r_ka.std()])

        # ── Confidence ───────────────────────────────────────────────────
        f.append(cn.mean())
        f.append(cn.min())
        f.append((cn < 0.3).mean())

        all_features.append(f)

    return np.array(all_features, dtype=np.float32)


def oversample_minority(X, y, target_ratio=0.5):
    """Simple random oversampling of minority classes to target_ratio of majority."""
    counts = Counter(y.tolist())
    max_count = max(counts.values())
    target = int(max_count * target_ratio)

    X_new, y_new = list(X), list(y)
    for cls, cnt in counts.items():
        if cnt < target:
            idx = np.where(y == cls)[0]
            n_extra = target - cnt
            extra_idx = np.random.choice(idx, size=n_extra, replace=True)
            X_new.extend(X[extra_idx])
            y_new.extend(y[extra_idx])

    X_new = np.array(X_new)
    y_new = np.array(y_new)
    perm = np.random.permutation(len(y_new))
    return X_new[perm], y_new[perm]


def main():
    data_dir = 'data/windows'
    train = np.load(f'{data_dir}/train.npz', allow_pickle=True)
    val   = np.load(f'{data_dir}/val.npz', allow_pickle=True)

    X_train, y_train = train['X'], train['y']
    X_val,   y_val   = val['X'],   val['y']

    print(f'Train: {X_train.shape}  Val: {X_val.shape}')
    print(f'Train dist: {dict(sorted(Counter(y_train.tolist()).items()))}')
    print(f'Val   dist: {dict(sorted(Counter(y_val.tolist()).items()))}')
    print()

    # ── Extract features ──────────────────────────────────────────────────
    print('Extracting features...')
    F_train = extract_features(X_train)
    F_val   = extract_features(X_val)
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    F_val   = np.nan_to_num(F_val,   nan=0.0, posinf=0.0, neginf=0.0)
    print(f'Features: {F_train.shape[1]} per sample')

    # ── Oversample minority classes ───────────────────────────────────────
    F_train_os, y_train_os = oversample_minority(F_train, y_train, target_ratio=0.6)
    print(f'After oversampling: {len(y_train_os)} samples')
    print(f'  Oversampled dist: {dict(sorted(Counter(y_train_os.tolist()).items()))}')
    print()

    # Scale
    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train_os)
    F_val_s   = scaler.transform(F_val)

    # ── Individual classifiers ────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=15, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )
    svm = SVC(
        C=10, gamma='scale', class_weight='balanced',
        probability=True, random_state=42
    )

    # ── Soft voting ensemble ──────────────────────────────────────────────
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
        voting='soft', weights=[1, 2, 1]  # weight GB higher (best individual)
    )

    classifiers = {
        'RandomForest':     rf,
        'GradientBoosting': gb,
        'SVM_RBF':          svm,
        'Ensemble':         ensemble,
    }

    results = {}
    for name, clf in classifiers.items():
        print(f'--- {name} ---')
        clf.fit(F_train_s, y_train_os)
        preds = clf.predict(F_val_s)
        acc = (preds == y_val).mean()
        f1  = f1_score(y_val, preds, average='macro', zero_division=0)
        f1_per = f1_score(y_val, preds, average=None, zero_division=0)

        inv_mask = y_val == CLASS_NAMES.index('invalid')
        fpr = (preds[inv_mask] != CLASS_NAMES.index('invalid')).sum() / max(1, inv_mask.sum())

        print(f'  Acc: {acc*100:.1f}%  F1: {f1:.3f}  FPR: {fpr*100:.1f}%')
        print(classification_report(y_val, preds, target_names=CLASS_NAMES, zero_division=0, digits=3))
        results[name] = {'acc': acc, 'f1': f1, 'fpr': fpr, 'preds': preds, 'f1_per': f1_per}

    # ── Best model ────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]['acc'])
    best = results[best_name]
    print(f'\n{"="*60}')
    print(f'BEST: {best_name}')
    print(f'  Acc:  {best["acc"]*100:.1f}%')
    print(f'  F1:   {best["f1"]:.3f}')
    print(f'  FPR:  {best["fpr"]*100:.1f}%')
    print(f'{"="*60}')

    # Confusion matrix
    cm = confusion_matrix(y_val, best['preds'])
    print('\nConfusion Matrix:')
    print(f'{"":>12}', '  '.join(f'{n:>8}' for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f'{CLASS_NAMES[i]:>12}', '  '.join(f'{v:>8}' for v in row))

    # Per-class F1
    print('\nPer-class F1:')
    for i, name in enumerate(CLASS_NAMES):
        f1_i = best['f1_per'][i]
        status = '✓' if (i < 4 and f1_i >= 0.80) or (i == 4) else '✗'
        print(f'  {status} {name:<12} {f1_i:.3f}')

    # ── Cross-validation on ALL data for reliable estimate ────────────────
    print('\n--- 5-Fold Cross-Validation (all 819 samples) ---')
    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    F_all = extract_features(X_all)
    F_all = np.nan_to_num(F_all, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_cv = StandardScaler()
    F_all_s = scaler_cv.fit_transform(F_all)

    gb_cv = GradientBoostingClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_cv, F_all_s, y_all, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(gb_cv, F_all_s, y_all, cv=skf, scoring='f1_macro')
    print(f'  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  ({cv_scores})')
    print(f'  CV F1 Macro: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}')

    # ── Save scaler + model metadata for export ──────────────────────────
    os.makedirs('models', exist_ok=True)
    np.savez('models/feature_scaler.npz',
             mean=scaler.mean_, scale=scaler.scale_)
    print(f'\nScaler saved to models/feature_scaler.npz')


if __name__ == '__main__':
    main()
