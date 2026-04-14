#!/usr/bin/env python3
"""
train_v2_twostage.py — Two-stage classifier with volley-specific features.

Strategy:
  Stage 1: Binary classifier (stroke vs invalid) — optimize FPR
  Stage 2: 4-class stroke classifier (FH/BH/SV/VL) — only on predicted strokes

Plus:
  - Volley-targeted features (motion duration, acceleration profile, elbow range)
  - Temporal segmentation (split window into thirds for finer features)
  - SMOTE oversampling via manual implementation
  - Per-class probability threshold optimization
"""

import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ── COCO joints ──────────────────────────────────────────────────────────────
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
SF = 30  # stroke frame


def joint_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def extract_features_v2(X):
    """Extract 120+ features with temporal segmentation and volley-specific features."""
    N, T, V, C = X.shape
    coords = X[:, :, :, :2]
    conf   = X[:, :, :, 2]

    # Temporal segments: first third, middle third, last third
    t1 = T // 3          # 0..14
    t2 = 2 * T // 3      # 15..29
    # t3: 30..44

    all_feats = []

    for i in range(N):
        f = []
        c = coords[i]
        cn = conf[i]

        l_wr = c[:, L_WRIST];  r_wr = c[:, R_WRIST]
        l_el = c[:, L_ELBOW];  r_el = c[:, R_ELBOW]
        l_sh = c[:, L_SHOULDER]; r_sh = c[:, R_SHOULDER]
        l_hp = c[:, L_HIP];   r_hp = c[:, R_HIP]
        l_kn = c[:, L_KNEE];  r_kn = c[:, R_KNEE]
        l_ak = c[:, L_ANKLE]; r_ak = c[:, R_ANKLE]
        nose = c[:, NOSE]

        # ── 1. Wrist velocity per segment ────────────────────────────────
        l_vel = np.linalg.norm(np.diff(l_wr, axis=0), axis=1)
        r_vel = np.linalg.norm(np.diff(r_wr, axis=0), axis=1)

        for vel in [l_vel, r_vel]:
            # Global stats
            f.extend([vel.mean(), vel.std(), vel.max(), vel.min()])
            # Peak timing
            f.append(np.argmax(vel) / (T - 1))
            # Per-segment stats (TEMPORAL GRANULARITY)
            for s, e in [(0, t1), (t1, t2), (t2, T-1)]:
                seg = vel[s:e]
                if len(seg) > 0:
                    f.extend([seg.mean(), seg.max()])
                else:
                    f.extend([0, 0])

        # Dominance
        f.append(r_vel.sum() - l_vel.sum())
        f.append(r_vel.max() / (l_vel.max() + 1e-8))
        f.append(r_vel.sum() / (l_vel.sum() + r_vel.sum() + 1e-8))

        # ── 2. Arm angles at key frames ──────────────────────────────────
        key_frames = [0, 10, 20, 29, 30, 35, 40, 44]
        for t in key_frames:
            t = min(t, T - 1)
            la = joint_angle(c[t, L_SHOULDER], c[t, L_ELBOW], c[t, L_WRIST])
            ra = joint_angle(c[t, R_SHOULDER], c[t, R_ELBOW], c[t, R_WRIST])
            f.extend([la, ra, ra - la])

        # Arm angle dynamics
        l_angles = np.array([joint_angle(c[t, L_SHOULDER], c[t, L_ELBOW], c[t, L_WRIST]) for t in range(T)])
        r_angles = np.array([joint_angle(c[t, R_SHOULDER], c[t, R_ELBOW], c[t, R_WRIST]) for t in range(T)])
        f.extend([l_angles.std(), r_angles.std()])
        f.extend([l_angles.max() - l_angles.min(), r_angles.max() - r_angles.min()])
        # Angle velocity (how fast arm extends/flexes)
        l_angle_vel = np.abs(np.diff(l_angles))
        r_angle_vel = np.abs(np.diff(r_angles))
        f.extend([l_angle_vel.mean(), l_angle_vel.max()])
        f.extend([r_angle_vel.mean(), r_angle_vel.max()])

        # ── 3. Wrist height relative features ────────────────────────────
        l_wh = l_wr[:, 1] - l_sh[:, 1]
        r_wh = r_wr[:, 1] - r_sh[:, 1]
        f.extend([l_wh.mean(), l_wh.max(), l_wh.min(), l_wh.std()])
        f.extend([r_wh.mean(), r_wh.max(), r_wh.min(), r_wh.std()])
        f.append(max(l_wr[:, 1].max(), r_wr[:, 1].max()))
        # Wrist above nose (serve indicator)
        f.append((l_wr[:, 1] > nose[:, 1]).sum() / T)
        f.append((r_wr[:, 1] > nose[:, 1]).sum() / T)

        # ── 4. Torso & rotation ──────────────────────────────────────────
        mid_sh = (l_sh + r_sh) / 2
        mid_hp = (l_hp + r_hp) / 2
        torso = mid_sh - mid_hp
        torso_angle = np.arctan2(torso[:, 0], torso[:, 1] + 1e-8)
        f.extend([torso_angle.mean(), torso_angle.std()])

        sh_dx = r_sh[:, 0] - l_sh[:, 0]
        f.extend([sh_dx.mean(), sh_dx.std(), sh_dx.max() - sh_dx.min()])

        # ── 5. Motion compactness ────────────────────────────────────────
        l_range_x = np.ptp(l_wr[:, 0])
        l_range_y = np.ptp(l_wr[:, 1])
        r_range_x = np.ptp(r_wr[:, 0])
        r_range_y = np.ptp(r_wr[:, 1])
        f.extend([l_range_x, l_range_y, r_range_x, r_range_y])

        l_travel = l_vel.sum()
        r_travel = r_vel.sum()
        f.extend([l_travel, r_travel, max(l_travel, r_travel)])

        # ── 6. VOLLEY-SPECIFIC FEATURES ──────────────────────────────────

        # 6a. Motion compactness ratio (volley = compact)
        dom_vel = r_vel if r_vel.sum() > l_vel.sum() else l_vel
        dom_wr = r_wr if r_vel.sum() > l_vel.sum() else l_wr
        dom_range = max(np.ptp(dom_wr[:, 0]), np.ptp(dom_wr[:, 1]))
        f.append(dom_range)  # smaller for volley

        # 6b. Velocity spike ratio: peak vs mean (volley = sharp spike)
        spike_ratio = dom_vel.max() / (dom_vel.mean() + 1e-8)
        f.append(spike_ratio)

        # 6c. Motion concentration: what % of total motion is in a 5-frame window
        if len(dom_vel) >= 5:
            conv = np.convolve(dom_vel, np.ones(5)/5, mode='valid')
            best_window = conv.max() * 5
            motion_concentration = best_window / (dom_vel.sum() + 1e-8)
        else:
            motion_concentration = 1.0
        f.append(motion_concentration)  # higher for volley (concentrated action)

        # 6d. Duration of significant motion (volley = short burst)
        motion_thresh = dom_vel.mean() * 0.5
        active_frames = (dom_vel > motion_thresh).sum()
        f.append(active_frames / (T - 1))  # lower for volley

        # 6e. Elbow angle at stroke moment (volley = more flexed)
        sf = min(SF, T - 1)
        l_elbow_at_stroke = joint_angle(c[sf, L_SHOULDER], c[sf, L_ELBOW], c[sf, L_WRIST])
        r_elbow_at_stroke = joint_angle(c[sf, R_SHOULDER], c[sf, R_ELBOW], c[sf, R_WRIST])
        f.extend([l_elbow_at_stroke, r_elbow_at_stroke])

        # 6f. Follow-through length (volley = minimal follow-through)
        peak_frame = np.argmax(dom_vel)
        follow_through = dom_vel[peak_frame:].sum() / (dom_vel.sum() + 1e-8)
        f.append(follow_through)

        # 6g. Wrist distance from body center (volley = close to body)
        body_center = c.mean(axis=1)  # (T, 2)
        l_dist_body = np.linalg.norm(l_wr - body_center, axis=1)
        r_dist_body = np.linalg.norm(r_wr - body_center, axis=1)
        f.extend([l_dist_body.mean(), r_dist_body.mean()])
        f.extend([l_dist_body.max(), r_dist_body.max()])

        # 6h. Two-hand involvement (volley often two-handed)
        wrist_dist = np.linalg.norm(l_wr - r_wr, axis=1)
        f.extend([wrist_dist.mean(), wrist_dist.min(), wrist_dist.std()])

        # ── 7. Stroke event ──────────────────────────────────────────────
        f.extend(l_wr[sf].tolist())
        f.extend(r_wr[sf].tolist())
        l_reach = np.linalg.norm(l_wr[sf] - l_hp[sf])
        r_reach = np.linalg.norm(r_wr[sf] - r_hp[sf])
        f.extend([l_reach, r_reach])

        # ── 8. Lateral movement ──────────────────────────────────────────
        f.append(l_wr[-1, 0] - l_wr[0, 0])
        f.append(r_wr[-1, 0] - r_wr[0, 0])

        body_vel = np.linalg.norm(np.diff(body_center, axis=0), axis=1)
        f.extend([body_vel.mean(), body_vel.std(), body_vel.sum()])

        # ── 9. Knee bend ─────────────────────────────────────────────────
        l_ka = np.array([joint_angle(c[t, L_HIP], c[t, L_KNEE], c[t, L_ANKLE]) for t in range(T)])
        r_ka = np.array([joint_angle(c[t, R_HIP], c[t, R_KNEE], c[t, R_ANKLE]) for t in range(T)])
        f.extend([l_ka.mean(), l_ka.min(), l_ka.std()])
        f.extend([r_ka.mean(), r_ka.min(), r_ka.std()])

        # ── 10. Confidence ───────────────────────────────────────────────
        f.append(cn.mean())
        f.append(cn.min())
        f.append((cn < 0.3).mean())

        # ── 11. Elbow velocity (forearm action) ─────────────────────────
        l_el_vel = np.linalg.norm(np.diff(l_el, axis=0), axis=1)
        r_el_vel = np.linalg.norm(np.diff(r_el, axis=0), axis=1)
        f.extend([l_el_vel.mean(), l_el_vel.max()])
        f.extend([r_el_vel.mean(), r_el_vel.max()])

        # Wrist-to-elbow velocity ratio (how much is wrist "flicking")
        f.append(l_vel.max() / (l_el_vel.max() + 1e-8))
        f.append(r_vel.max() / (r_el_vel.max() + 1e-8))

        all_feats.append(f)

    return np.array(all_feats, dtype=np.float32)


def smote_oversample(X, y, target_count, k=5):
    """SMOTE oversampling for a specific class."""
    from sklearn.neighbors import NearestNeighbors

    new_samples = []
    nn = NearestNeighbors(n_neighbors=min(k, len(X) - 1), algorithm='auto')
    nn.fit(X)

    n_needed = target_count - len(X)
    for _ in range(n_needed):
        idx = np.random.randint(len(X))
        neighbors = nn.kneighbors([X[idx]], return_distance=False)[0]
        nn_idx = neighbors[np.random.randint(len(neighbors))]
        lam = np.random.random()
        new_samples.append(X[idx] + lam * (X[nn_idx] - X[idx]))

    if new_samples:
        return np.vstack([X, np.array(new_samples)])
    return X


def optimize_thresholds(probs, y_true, n_classes=5):
    """Find per-class probability thresholds that maximize F1."""
    best_thresholds = np.zeros(n_classes)

    for cls in range(n_classes):
        best_f1 = 0
        best_t = 0.0
        for t in np.arange(0.1, 0.6, 0.02):
            # Predict: if prob > threshold, assign class; else use argmax
            preds = np.argmax(probs, axis=1)
            # Boost this class: lower threshold = more likely to predict
            adjusted = probs.copy()
            adjusted[:, cls] += t
            preds_adj = np.argmax(adjusted, axis=1)
            f1 = f1_score(y_true, preds_adj, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[cls] = best_t

    return best_thresholds


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

    # ── Extract V2 features ───────────────────────────────────────────────
    print('Extracting V2 features...')
    F_train = extract_features_v2(X_train)
    F_val   = extract_features_v2(X_val)
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    F_val   = np.nan_to_num(F_val,   nan=0.0, posinf=0.0, neginf=0.0)
    print(f'Features: {F_train.shape[1]} per sample')

    # ── SMOTE oversampling ────────────────────────────────────────────────
    counts = Counter(y_train.tolist())
    max_count = max(counts.values())
    target = int(max_count * 0.7)  # at least 70% of majority

    F_smote, y_smote = [], []
    for cls in range(5):
        mask = y_train == cls
        F_cls = F_train[mask]
        if len(F_cls) < target:
            F_cls_up = smote_oversample(F_cls, y_train[mask], target)
        else:
            F_cls_up = F_cls
        F_smote.append(F_cls_up)
        y_smote.extend([cls] * len(F_cls_up))

    F_train_sm = np.vstack(F_smote)
    y_train_sm = np.array(y_smote)
    perm = np.random.permutation(len(y_train_sm))
    F_train_sm = F_train_sm[perm]
    y_train_sm = y_train_sm[perm]
    print(f'After SMOTE: {len(y_train_sm)} samples')
    print(f'  Dist: {dict(sorted(Counter(y_train_sm.tolist()).items()))}')
    print()

    # Scale
    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train_sm)
    F_val_s   = scaler.transform(F_val)

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH A: Direct 5-class (baseline with new features)
    # ═══════════════════════════════════════════════════════════════════════
    print('=' * 60)
    print('APPROACH A: Direct 5-class with V2 features + SMOTE')
    print('=' * 60)

    gb = GradientBoostingClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.02,
        subsample=0.8, min_samples_leaf=2, random_state=42
    )
    gb.fit(F_train_s, y_train_sm)
    preds_a = gb.predict(F_val_s)
    acc_a = (preds_a == y_val).mean()
    f1_a = f1_score(y_val, preds_a, average='macro', zero_division=0)
    inv_mask = y_val == 4
    fpr_a = (preds_a[inv_mask] != 4).sum() / max(1, inv_mask.sum())
    print(f'  Acc: {acc_a*100:.1f}%  F1: {f1_a:.3f}  FPR: {fpr_a*100:.1f}%')
    print(classification_report(y_val, preds_a, target_names=CLASS_NAMES, zero_division=0, digits=3))

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH B: Two-Stage Classification
    # ═══════════════════════════════════════════════════════════════════════
    print('=' * 60)
    print('APPROACH B: Two-Stage (stroke/invalid → stroke type)')
    print('=' * 60)

    # Stage 1: Binary stroke vs invalid
    y_binary_train = (y_train_sm < 4).astype(int)  # 1=stroke, 0=invalid
    y_binary_val = (y_val < 4).astype(int)

    stage1 = GradientBoostingClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=2, random_state=42
    )
    stage1.fit(F_train_s, y_binary_train)
    stage1_preds = stage1.predict(F_val_s)
    stage1_probs = stage1.predict_proba(F_val_s)

    # Tune Stage 1 threshold to reduce FPR
    best_fpr = 1.0
    best_t1 = 0.5
    best_acc1 = 0
    for t in np.arange(0.3, 0.7, 0.01):
        preds_t = (stage1_probs[:, 1] > t).astype(int)
        fpr_t = (preds_t[~y_binary_val.astype(bool)]).mean()
        acc_t = (preds_t == y_binary_val).mean()
        # We want acc > 85% AND low FPR
        if acc_t >= 0.84 and fpr_t < best_fpr:
            best_fpr = fpr_t
            best_t1 = t
            best_acc1 = acc_t

    stage1_final = (stage1_probs[:, 1] > best_t1).astype(int)
    s1_acc = (stage1_final == y_binary_val).mean()
    s1_fpr = stage1_final[~y_binary_val.astype(bool)].mean()
    print(f'\n  Stage 1 (stroke/invalid): threshold={best_t1:.2f}')
    print(f'  Binary Acc: {s1_acc*100:.1f}%  FPR: {s1_fpr*100:.1f}%')

    # Stage 2: 4-class stroke type (trained only on strokes)
    stroke_train_mask = y_train_sm < 4
    stroke_val_mask = stage1_final == 1  # predicted as stroke

    stage2 = GradientBoostingClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.02,
        subsample=0.8, min_samples_leaf=2, random_state=42
    )
    stage2.fit(F_train_s[stroke_train_mask], y_train_sm[stroke_train_mask])

    # Combine predictions
    preds_b = np.full(len(y_val), 4)  # default: invalid
    if stroke_val_mask.any():
        preds_b[stroke_val_mask] = stage2.predict(F_val_s[stroke_val_mask])

    acc_b = (preds_b == y_val).mean()
    f1_b = f1_score(y_val, preds_b, average='macro', zero_division=0)
    fpr_b = (preds_b[inv_mask] != 4).sum() / max(1, inv_mask.sum())
    print(f'\n  Two-Stage Combined:')
    print(f'  Acc: {acc_b*100:.1f}%  F1: {f1_b:.3f}  FPR: {fpr_b*100:.1f}%')
    print(classification_report(y_val, preds_b, target_names=CLASS_NAMES, zero_division=0, digits=3))

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH C: Threshold-optimized probabilities
    # ═══════════════════════════════════════════════════════════════════════
    print('=' * 60)
    print('APPROACH C: GB + per-class threshold optimization')
    print('=' * 60)

    gb_prob = GradientBoostingClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.02,
        subsample=0.8, min_samples_leaf=2, random_state=42
    )
    gb_prob.fit(F_train_s, y_train_sm)
    probs_c = gb_prob.predict_proba(F_val_s)

    # Grid search over class-specific bias terms
    best_f1_c = 0
    best_bias = np.zeros(5)
    for trial in range(5000):
        bias = np.random.uniform(-0.15, 0.15, size=5)
        adjusted = probs_c + bias
        preds_trial = np.argmax(adjusted, axis=1)
        f1_trial = f1_score(y_val, preds_trial, average='macro', zero_division=0)
        if f1_trial > best_f1_c:
            best_f1_c = f1_trial
            best_bias = bias.copy()

    preds_c = np.argmax(probs_c + best_bias, axis=1)
    acc_c = (preds_c == y_val).mean()
    fpr_c = (preds_c[inv_mask] != 4).sum() / max(1, inv_mask.sum())
    print(f'  Best bias: {best_bias.round(3)}')
    print(f'  Acc: {acc_c*100:.1f}%  F1: {best_f1_c:.3f}  FPR: {fpr_c*100:.1f}%')
    print(classification_report(y_val, preds_c, target_names=CLASS_NAMES, zero_division=0, digits=3))

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH D: RandomForest + GradientBoosting Stacking
    # ═══════════════════════════════════════════════════════════════════════
    print('=' * 60)
    print('APPROACH D: Stacking (RF+GB base → GB meta)')
    print('=' * 60)

    from sklearn.model_selection import cross_val_predict

    rf_base = RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    gb_base = GradientBoostingClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )

    # Get cross-validated predictions for stacking
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_base.fit(F_train_s, y_train_sm)
    gb_base.fit(F_train_s, y_train_sm)

    rf_probs_train = cross_val_predict(rf_base, F_train_s, y_train_sm, cv=skf, method='predict_proba')
    gb_probs_train = cross_val_predict(gb_base, F_train_s, y_train_sm, cv=skf, method='predict_proba')

    # Stack: combine base model probabilities as meta-features
    meta_train = np.hstack([rf_probs_train, gb_probs_train, F_train_s[:, :20]])  # top 20 original features too

    rf_probs_val = rf_base.predict_proba(F_val_s)
    gb_probs_val = gb_base.predict_proba(F_val_s)
    meta_val = np.hstack([rf_probs_val, gb_probs_val, F_val_s[:, :20]])

    meta_clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        random_state=42
    )
    meta_clf.fit(meta_train, y_train_sm)
    preds_d = meta_clf.predict(meta_val)
    acc_d = (preds_d == y_val).mean()
    f1_d = f1_score(y_val, preds_d, average='macro', zero_division=0)
    fpr_d = (preds_d[inv_mask] != 4).sum() / max(1, inv_mask.sum())
    print(f'  Acc: {acc_d*100:.1f}%  F1: {f1_d:.3f}  FPR: {fpr_d*100:.1f}%')
    print(classification_report(y_val, preds_d, target_names=CLASS_NAMES, zero_division=0, digits=3))

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    approaches = {
        'A (Direct 5-class)':       {'acc': acc_a, 'f1': f1_a, 'fpr': fpr_a},
        'B (Two-Stage)':            {'acc': acc_b, 'f1': f1_b, 'fpr': fpr_b},
        'C (Threshold-optimized)':  {'acc': acc_c, 'f1': best_f1_c, 'fpr': fpr_c},
        'D (Stacking)':             {'acc': acc_d, 'f1': f1_d, 'fpr': fpr_d},
    }
    for name, r in approaches.items():
        go = '✓' if r['acc'] >= 0.85 and r['fpr'] <= 0.15 else '✗'
        print(f'  {go} {name}: Acc={r["acc"]*100:.1f}%  F1={r["f1"]:.3f}  FPR={r["fpr"]*100:.1f}%')

    # Best by F1
    best_name = max(approaches, key=lambda k: approaches[k]['f1'])
    best = approaches[best_name]
    print(f'\n  BEST by F1: {best_name}')
    print(f'  Acc={best["acc"]*100:.1f}%  F1={best["f1"]:.3f}  FPR={best["fpr"]*100:.1f}%')

    # Feature importance from direct GB
    importances = gb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    print(f'\nTop 20 feature importances:')
    for rank, idx in enumerate(top_idx):
        print(f'  {rank+1:2d}. feature[{idx:3d}] imp={importances[idx]:.4f}')


if __name__ == '__main__':
    main()
