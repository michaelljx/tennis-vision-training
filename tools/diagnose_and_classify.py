#!/usr/bin/env python3
"""
diagnose_and_classify.py — Data quality diagnosis + feature-based classification.

Extracts biomechanical features from skeleton windows and trains
classical ML classifiers. Runs locally (no GPU needed, <30 seconds).

This serves two purposes:
1. Diagnose data quality — if classical ML can't classify well, the data has issues
2. Potentially provide a better model for small datasets than end-to-end deep learning
"""

import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
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

CLASS_NAMES = ['forehand', 'backhand', 'serve', 'invalid']


def joint_angle(a, b, c):
    """Angle at joint b given points a, b, c (in radians)."""
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def extract_features(X):
    """
    Extract biomechanical features from skeleton windows.
    X: (N, T=45, V=17, C=3) where C = (x, y, confidence)
    Returns: (N, F) feature matrix
    """
    N, T, V, C = X.shape
    coords = X[:, :, :, :2]   # (N, T, V, 2) — x, y only
    conf   = X[:, :, :, 2]    # (N, T, V) — confidence

    features = []

    for i in range(N):
        f = []
        c = coords[i]          # (T, V, 2)
        cn = conf[i]           # (T, V)

        # ── 1. Wrist trajectories (most important for stroke type) ──────
        l_wrist = c[:, L_WRIST]   # (T, 2)
        r_wrist = c[:, R_WRIST]   # (T, 2)
        l_elbow = c[:, L_ELBOW]
        r_elbow = c[:, R_ELBOW]
        l_shoulder = c[:, L_SHOULDER]
        r_shoulder = c[:, R_SHOULDER]

        # Wrist velocity (frame-to-frame displacement)
        l_wrist_vel = np.linalg.norm(np.diff(l_wrist, axis=0), axis=1)  # (T-1,)
        r_wrist_vel = np.linalg.norm(np.diff(r_wrist, axis=0), axis=1)

        # Wrist speed stats
        f.extend([l_wrist_vel.mean(), l_wrist_vel.std(), l_wrist_vel.max()])
        f.extend([r_wrist_vel.mean(), r_wrist_vel.std(), r_wrist_vel.max()])

        # Which wrist moves more? (forehand vs backhand discriminator)
        f.append(r_wrist_vel.sum() - l_wrist_vel.sum())  # positive = right hand dominant
        f.append(r_wrist_vel.max() / (l_wrist_vel.max() + 1e-8))

        # ── 2. Arm angles ────────────────────────────────────────────────
        # Left arm angle (shoulder-elbow-wrist) at key frames
        for t_idx in [0, 15, 29, 30, 35, 44]:  # start, mid-approach, pre-stroke, stroke, follow-through, end
            t = min(t_idx, T - 1)
            la = joint_angle(c[t, L_SHOULDER], c[t, L_ELBOW], c[t, L_WRIST])
            ra = joint_angle(c[t, R_SHOULDER], c[t, R_ELBOW], c[t, R_WRIST])
            f.extend([la, ra])

        # ── 3. Wrist height relative to shoulder (serve = high) ──────────
        l_wrist_h = l_wrist[:, 1] - l_shoulder[:, 1]  # y-axis relative
        r_wrist_h = r_wrist[:, 1] - r_shoulder[:, 1]

        f.extend([l_wrist_h.mean(), l_wrist_h.max(), l_wrist_h.min()])
        f.extend([r_wrist_h.mean(), r_wrist_h.max(), r_wrist_h.min()])

        # Max wrist height (serve discriminator)
        f.append(max(l_wrist[:, 1].max(), r_wrist[:, 1].max()))

        # ── 4. Body lean and torso rotation ──────────────────────────────
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip = (c[:, L_HIP] + c[:, R_HIP]) / 2

        # Torso lean angle (from vertical)
        torso_vec = mid_shoulder - mid_hip  # (T, 2)
        torso_angle = np.arctan2(torso_vec[:, 0], torso_vec[:, 1] + 1e-8)
        f.extend([torso_angle.mean(), torso_angle.std(), torso_angle.max(), torso_angle.min()])

        # Shoulder rotation (left-right shoulder distance in x)
        shoulder_dx = r_shoulder[:, 0] - l_shoulder[:, 0]
        f.extend([shoulder_dx.mean(), shoulder_dx.std()])

        # ── 5. Movement range (total displacement) ───────────────────────
        # How much does each wrist travel total
        l_travel = l_wrist_vel.sum()
        r_travel = r_wrist_vel.sum()
        f.extend([l_travel, r_travel])

        # Total body movement (center of mass proxy)
        body_center = c.mean(axis=1)  # (T, 2) — average of all joints
        body_vel = np.linalg.norm(np.diff(body_center, axis=0), axis=1)
        f.extend([body_vel.mean(), body_vel.std(), body_vel.sum()])

        # ── 6. Stroke event features (around frame 30) ──────────────────
        stroke_frame = min(30, T - 1)
        pre = max(0, stroke_frame - 5)
        post = min(T - 1, stroke_frame + 5)

        # Wrist acceleration around stroke
        l_accel = np.diff(l_wrist_vel[pre:post]) if post - pre > 1 else np.array([0])
        r_accel = np.diff(r_wrist_vel[pre:post]) if post - pre > 1 else np.array([0])
        f.extend([l_accel.max() if len(l_accel) > 0 else 0,
                  r_accel.max() if len(r_accel) > 0 else 0])

        # Wrist position at stroke moment
        f.extend(l_wrist[stroke_frame].tolist())
        f.extend(r_wrist[stroke_frame].tolist())

        # ── 7. Lateral movement (forehand=one side, backhand=other) ──────
        # Wrist x-position trajectory: direction of motion
        l_x_dir = l_wrist[-1, 0] - l_wrist[0, 0]
        r_x_dir = r_wrist[-1, 0] - r_wrist[0, 0]
        f.extend([l_x_dir, r_x_dir])

        # ── 8. Compactness (volley = compact, short motion) ──────────────
        l_wrist_range_x = l_wrist[:, 0].max() - l_wrist[:, 0].min()
        l_wrist_range_y = l_wrist[:, 1].max() - l_wrist[:, 1].min()
        r_wrist_range_x = r_wrist[:, 0].max() - r_wrist[:, 0].min()
        r_wrist_range_y = r_wrist[:, 1].max() - r_wrist[:, 1].min()
        f.extend([l_wrist_range_x, l_wrist_range_y, r_wrist_range_x, r_wrist_range_y])

        # Total motion range (small = volley)
        f.append(max(l_travel, r_travel))

        # ── 9. Confidence statistics ─────────────────────────────────────
        f.append(cn.mean())
        f.append(cn.min())
        f.append((cn < 0.3).mean())  # fraction of low-confidence keypoints

        # ── 10. Knee bend (serve wind-up) ────────────────────────────────
        l_knee_angle = joint_angle(c[:, L_HIP], c[:, L_KNEE], c[:, L_ANKLE])
        r_knee_angle = joint_angle(c[:, R_HIP], c[:, R_KNEE], c[:, R_ANKLE])
        f.extend([l_knee_angle.mean(), l_knee_angle.min()])
        f.extend([r_knee_angle.mean(), r_knee_angle.min()])

        features.append(f)

    return np.array(features, dtype=np.float32)


def main():
    import sys

    data_dir = 'data/windows'
    train = np.load(f'{data_dir}/train.npz', allow_pickle=True)
    val   = np.load(f'{data_dir}/val.npz', allow_pickle=True)

    X_train, y_train = train['X'], train['y']
    X_val,   y_val   = val['X'],   val['y']

    print(f'Train: {X_train.shape}  Val: {X_val.shape}')
    print(f'Train dist: {dict(Counter(y_train.tolist()))}')
    print(f'Val   dist: {dict(Counter(y_val.tolist()))}')
    print()

    # ── Data quality check ────────────────────────────────────────────────
    print('=== Data Quality Check ===')
    conf_train = X_train[:, :, :, 2]
    conf_val   = X_val[:, :, :, 2]
    print(f'Train confidence: mean={conf_train.mean():.3f}, min={conf_train.min():.3f}, '
          f'low(<0.3): {(conf_train < 0.3).mean()*100:.1f}%')
    print(f'Val   confidence: mean={conf_val.mean():.3f}, min={conf_val.min():.3f}, '
          f'low(<0.3): {(conf_val < 0.3).mean()*100:.1f}%')

    # Check coordinate ranges
    for name, X in [('Train', X_train), ('Val', X_val)]:
        x_coords = X[:, :, :, 0]
        y_coords = X[:, :, :, 1]
        print(f'{name} x range: [{x_coords.min():.3f}, {x_coords.max():.3f}]  '
              f'y range: [{y_coords.min():.3f}, {y_coords.max():.3f}]')
    print()

    # ── Extract features ──────────────────────────────────────────────────
    print('Extracting features...')
    F_train = extract_features(X_train)
    F_val   = extract_features(X_val)
    print(f'Features: {F_train.shape[1]} per sample')

    # Handle NaN/Inf
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    F_val   = np.nan_to_num(F_val,   nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)
    F_val_s   = scaler.transform(F_val)
    print()

    # ── Train classifiers ─────────────────────────────────────────────────
    classifiers = {
        'RandomForest':       RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                                      class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting':   GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                          subsample=0.8, random_state=42),
        'SVM_RBF':            SVC(C=10, gamma='scale', class_weight='balanced', random_state=42),
    }

    best_acc = 0
    best_name = ''

    for name, clf in classifiers.items():
        print(f'--- {name} ---')
        clf.fit(F_train_s, y_train)
        preds = clf.predict(F_val_s)
        acc = (preds == y_val).mean()
        f1  = f1_score(y_val, preds, average='macro', zero_division=0)

        print(f'  Val Acc: {acc:.4f} ({acc*100:.1f}%)  Macro F1: {f1:.4f}')
        print(classification_report(y_val, preds, target_names=CLASS_NAMES, zero_division=0, digits=3))

        # FPR
        inv_idx = CLASS_NAMES.index('invalid')
        inv_mask = y_val == inv_idx
        fpr = (preds[inv_mask] != inv_idx).sum() / max(1, inv_mask.sum())
        print(f'  FPR (invalid→stroke): {fpr:.3f} ({fpr*100:.1f}%)')

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_preds = preds

        print()

    print(f'=== Best: {best_name} — {best_acc*100:.1f}% ===')
    print()

    # Confusion matrix for best
    cm = confusion_matrix(y_val, best_preds)
    print('Confusion Matrix:')
    print(f'{"":>12}', '  '.join(f'{n:>8}' for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f'{CLASS_NAMES[i]:>12}', '  '.join(f'{v:>8}' for v in row))

    # Feature importance (from RF)
    if 'RandomForest' in classifiers:
        rf = classifiers['RandomForest']
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print(f'\nTop 15 feature importances (RandomForest):')
        for rank, idx in enumerate(top_idx):
            print(f'  {rank+1:2d}. feature[{idx:3d}] importance={importances[idx]:.4f}')


if __name__ == '__main__':
    main()
