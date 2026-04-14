#!/usr/bin/env python3
"""
train_v3_optimized.py — Optimize the 81.3% GB baseline.

Strategy:
  1. Use original 92 features (proven better than 120)
  2. Feature selection: remove noisy features via recursive elimination
  3. Aggressive hyperparameter grid search on GB
  4. Try different oversampling ratios
  5. Test different val splits to check robustness
  6. Combine: GB-optimized + per-class bias tuning (approach C from v2)
"""

import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
import sys

# Reuse the proven feature extractor from train_final_model
sys.path.insert(0, 'tools')
from train_final_model import extract_features, CLASS_NAMES


def oversample(F, y, target_ratio):
    counts = Counter(y.tolist())
    max_count = max(counts.values())
    target = int(max_count * target_ratio)
    F_new, y_new = list(F), list(y)
    for cls, cnt in counts.items():
        if cnt < target:
            idx = np.where(y == cls)[0]
            extra = np.random.choice(idx, size=target - cnt, replace=True)
            F_new.extend(F[extra])
            y_new.extend(y[extra])
    F_out = np.array(F_new)
    y_out = np.array(y_new)
    perm = np.random.permutation(len(y_out))
    return F_out[perm], y_out[perm]


def evaluate(clf, F_train, y_train, F_val, y_val, name='', verbose=True):
    clf.fit(F_train, y_train)
    preds = clf.predict(F_val)
    acc = (preds == y_val).mean()
    f1 = f1_score(y_val, preds, average='macro', zero_division=0)
    inv_mask = y_val == 4
    fpr = (preds[inv_mask] != 4).sum() / max(1, inv_mask.sum())
    if verbose:
        print(f'  {name}: Acc={acc*100:.1f}%  F1={f1:.3f}  FPR={fpr*100:.1f}%')
    return acc, f1, fpr, preds


def main():
    np.random.seed(42)
    data_dir = 'data/windows'
    train = np.load(f'{data_dir}/train.npz', allow_pickle=True)
    val   = np.load(f'{data_dir}/val.npz', allow_pickle=True)
    X_train, y_train = train['X'], train['y']
    X_val,   y_val   = val['X'],   val['y']

    print(f'Train: {X_train.shape}  Val: {X_val.shape}')
    print(f'Train dist: {dict(sorted(Counter(y_train.tolist()).items()))}')
    print(f'Val   dist: {dict(sorted(Counter(y_val.tolist()).items()))}')
    print()

    # ── Extract original 92 features ──────────────────────────────────────
    print('Extracting features...')
    F_train_raw = extract_features(X_train)
    F_val_raw   = extract_features(X_val)
    F_train_raw = np.nan_to_num(F_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    F_val_raw   = np.nan_to_num(F_val_raw,   nan=0.0, posinf=0.0, neginf=0.0)
    print(f'Features: {F_train_raw.shape[1]}')
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Hyperparameter grid search
    # ═══════════════════════════════════════════════════════════════════════
    print('=' * 60)
    print('EXP 1: GB Hyperparameter Grid Search')
    print('=' * 60)

    # Oversample at ratio=0.6 (proven best from v1)
    F_train_os, y_train_os = oversample(F_train_raw, y_train, 0.6)
    scaler = StandardScaler()
    F_tr = scaler.fit_transform(F_train_os)
    F_va = scaler.transform(F_val_raw)

    configs = [
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'min_samples_leaf': 3},
        {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.02, 'subsample': 0.8, 'min_samples_leaf': 2},
        {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.85, 'min_samples_leaf': 3},
        {'n_estimators': 1200, 'max_depth': 3, 'learning_rate': 0.015, 'subsample': 0.8, 'min_samples_leaf': 2},
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.75, 'min_samples_leaf': 3},
        {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.02, 'subsample': 0.8, 'min_samples_leaf': 4},
        {'n_estimators': 1500, 'max_depth': 3, 'learning_rate': 0.01, 'subsample': 0.8, 'min_samples_leaf': 2},
        {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'min_samples_leaf': 3},
    ]

    best_acc = 0
    best_cfg = None
    for i, cfg in enumerate(configs):
        gb = GradientBoostingClassifier(random_state=42, **cfg)
        acc, f1, fpr, _ = evaluate(gb, F_tr, y_train_os, F_va, y_val, f'Config {i}')
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
    print(f'\n  Best: Acc={best_acc*100:.1f}% — {best_cfg}')

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Oversampling ratio sweep
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('EXP 2: Oversampling Ratio Sweep')
    print('=' * 60)

    best_acc2 = 0
    best_ratio = 0
    for ratio in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        F_os, y_os = oversample(F_train_raw, y_train, ratio)
        sc = StandardScaler()
        F_t = sc.fit_transform(F_os)
        F_v = sc.transform(F_val_raw)
        gb = GradientBoostingClassifier(random_state=42, **best_cfg)
        acc, f1, fpr, _ = evaluate(gb, F_t, y_os, F_v, y_val, f'ratio={ratio:.1f}')
        if acc > best_acc2:
            best_acc2 = acc
            best_ratio = ratio
    print(f'\n  Best ratio: {best_ratio} → Acc={best_acc2*100:.1f}%')

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Feature selection
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('EXP 3: Feature Selection (Mutual Info)')
    print('=' * 60)

    F_os, y_os = oversample(F_train_raw, y_train, best_ratio)
    sc = StandardScaler()
    F_t = sc.fit_transform(F_os)
    F_v = sc.transform(F_val_raw)

    for k in [30, 40, 50, 60, 70, 80, 92]:
        if k > F_train_raw.shape[1]:
            continue
        if k == F_train_raw.shape[1]:
            F_t_sel, F_v_sel = F_t, F_v
            label = 'all'
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
            F_t_sel = selector.fit_transform(F_t, y_os)
            F_v_sel = selector.transform(F_v)
            label = f'k={k}'

        gb = GradientBoostingClassifier(random_state=42, **best_cfg)
        acc, f1, fpr, _ = evaluate(gb, F_t_sel, y_os, F_v_sel, y_val, label)

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: Per-class bias on best config
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('EXP 4: Best Config + Per-class Bias Optimization')
    print('=' * 60)

    F_os, y_os = oversample(F_train_raw, y_train, best_ratio)
    sc = StandardScaler()
    F_t = sc.fit_transform(F_os)
    F_v = sc.transform(F_val_raw)

    gb = GradientBoostingClassifier(random_state=42, **best_cfg)
    gb.fit(F_t, y_os)
    probs = gb.predict_proba(F_v)

    # Argmax baseline
    preds_base = np.argmax(probs, axis=1)
    acc_base = (preds_base == y_val).mean()
    f1_base = f1_score(y_val, preds_base, average='macro', zero_division=0)
    print(f'  Baseline (argmax): Acc={acc_base*100:.1f}%  F1={f1_base:.3f}')

    # Exhaustive bias search
    best_f1_bias = f1_base
    best_bias = np.zeros(5)
    best_acc_bias = acc_base

    for _ in range(20000):
        bias = np.random.uniform(-0.2, 0.2, size=5)
        adjusted = probs + bias
        preds_adj = np.argmax(adjusted, axis=1)
        f1_adj = f1_score(y_val, preds_adj, average='macro', zero_division=0)
        acc_adj = (preds_adj == y_val).mean()
        # Optimize for a combination of F1 and accuracy
        score = 0.5 * f1_adj + 0.5 * acc_adj
        if score > 0.5 * best_f1_bias + 0.5 * best_acc_bias:
            best_f1_bias = f1_adj
            best_acc_bias = acc_adj
            best_bias = bias.copy()

    preds_biased = np.argmax(probs + best_bias, axis=1)
    inv_mask = y_val == 4
    fpr_biased = (preds_biased[inv_mask] != 4).sum() / max(1, inv_mask.sum())
    print(f'  Bias-optimized: Acc={best_acc_bias*100:.1f}%  F1={best_f1_bias:.3f}  FPR={fpr_biased*100:.1f}%')
    print(f'  Bias: {best_bias.round(3)}')
    print(classification_report(y_val, preds_biased, target_names=CLASS_NAMES, zero_division=0, digits=3))

    cm = confusion_matrix(y_val, preds_biased)
    print('Confusion Matrix:')
    print(f'{"":>12}', '  '.join(f'{n:>8}' for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f'{CLASS_NAMES[i]:>12}', '  '.join(f'{v:>8}' for v in row))

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 5: Cross-validation stability check
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('EXP 5: 5-Fold CV on All Data')
    print('=' * 60)

    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    F_all = extract_features(X_all)
    F_all = np.nan_to_num(F_all, nan=0.0, posinf=0.0, neginf=0.0)

    sc_all = StandardScaler()
    F_all_s = sc_all.fit_transform(F_all)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(F_all_s, y_all)):
        F_tr_fold = F_all_s[train_idx]
        y_tr_fold = y_all[train_idx]
        F_va_fold = F_all_s[val_idx]
        y_va_fold = y_all[val_idx]

        # Oversample training fold
        counts = Counter(y_tr_fold.tolist())
        max_c = max(counts.values())
        tgt = int(max_c * best_ratio)
        F_os_fold, y_os_fold = [], []
        for cls in range(5):
            mask = y_tr_fold == cls
            F_cls = F_tr_fold[mask]
            if len(F_cls) < tgt:
                extra_idx = np.random.choice(len(F_cls), size=tgt - len(F_cls), replace=True)
                F_cls = np.vstack([F_cls, F_cls[extra_idx]])
            F_os_fold.append(F_cls)
            y_os_fold.extend([cls] * len(F_cls))
        F_os_fold = np.vstack(F_os_fold)
        y_os_fold = np.array(y_os_fold)

        gb_fold = GradientBoostingClassifier(random_state=42, **best_cfg)
        gb_fold.fit(F_os_fold, y_os_fold)
        preds_fold = gb_fold.predict(F_va_fold)
        acc_fold = (preds_fold == y_va_fold).mean()
        f1_fold = f1_score(y_va_fold, preds_fold, average='macro', zero_division=0)
        cv_accs.append(acc_fold)
        cv_f1s.append(f1_fold)
        print(f'  Fold {fold+1}: Acc={acc_fold*100:.1f}%  F1={f1_fold:.3f}')

    print(f'\n  CV Acc: {np.mean(cv_accs)*100:.1f}% ± {np.std(cv_accs)*100:.1f}%')
    print(f'  CV F1:  {np.mean(cv_f1s):.3f} ± {np.std(cv_f1s):.3f}')

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 6: RandomForest (sometimes better than GB for small data)
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('EXP 6: Optimized RandomForest')
    print('=' * 60)

    F_os, y_os = oversample(F_train_raw, y_train, best_ratio)
    sc = StandardScaler()
    F_t = sc.fit_transform(F_os)
    F_v = sc.transform(F_val_raw)

    rf_configs = [
        {'n_estimators': 2000, 'max_depth': 10, 'min_samples_leaf': 2, 'class_weight': 'balanced'},
        {'n_estimators': 2000, 'max_depth': 12, 'min_samples_leaf': 1, 'class_weight': 'balanced'},
        {'n_estimators': 2000, 'max_depth': 15, 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample'},
        {'n_estimators': 3000, 'max_depth': None, 'min_samples_leaf': 3, 'class_weight': 'balanced'},
    ]

    for i, cfg in enumerate(rf_configs):
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, **cfg)
        acc, f1, fpr, _ = evaluate(rf, F_t, y_os, F_v, y_val, f'RF Config {i}')

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('FINAL: Best achievable with feature-based approach')
    print(f'  Best GB config: {best_cfg}')
    print(f'  Best oversample ratio: {best_ratio}')
    print(f'  Val Acc: {best_acc_bias*100:.1f}%  F1: {best_f1_bias:.3f}')
    print(f'  CV Acc:  {np.mean(cv_accs)*100:.1f}% ± {np.std(cv_accs)*100:.1f}%')
    print('=' * 60)


if __name__ == '__main__':
    main()
