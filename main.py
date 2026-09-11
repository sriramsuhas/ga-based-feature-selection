from preprocessing import extract_and_split_dataset
from feature_extraction import extract_features
from mlmodel_training import train_rf_balanced, evaluate_rf_balanced_once
from ga_feature_selection import run_ga, apply_feature_mask

import numpy as np
import os
import sys
import random
import torch
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def set_global_seed(seed=42, force_cpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

def eval_svm(npzfile, mask=None):
    d = np.load(npzfile)
    X, y = d["X"], d["y"]
    if mask is not None:
        X = X[:, mask.astype(bool)]
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    return cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

def main(dataset_input, dataset_name, groups_file=None, force_cpu=False):
    print("\n[Main] Starting pipeline...")
    set_global_seed(42, force_cpu=force_cpu)
    os.makedirs('results', exist_ok=True)

    if isinstance(dataset_input, str):
        paths = extract_and_split_dataset(dataset_input, 'data_processed_' + dataset_name)
    elif isinstance(dataset_input, list) and len(dataset_input) == 3:
        paths = {'train': dataset_input[0], 'validate': dataset_input[1], 'test': dataset_input[2]}
    elif isinstance(dataset_input, list) and len(dataset_input) == 2:
        trainval_folder, test_folder = dataset_input
        trainval_paths = extract_and_split_dataset(trainval_folder, 'data_processed_' + dataset_name)
        paths = {'train': trainval_paths['train'], 'validate': trainval_paths['validate'], 'test': test_folder}
    else:
        raise ValueError("Unsupported dataset input configuration.")

    train_feat_file = extract_features(paths['train'], f'results/{dataset_name}_features_train.npz')
    val_feat_file   = extract_features(paths['validate'], f'results/{dataset_name}_features_val.npz')
    test_feat_file  = extract_features(paths['test'], f'results/{dataset_name}_features_test.npz')

    group_info = None
    if groups_file and os.path.isfile(groups_file):
        g = np.load(groups_file)
        group_info = {'train': g.get('groups_train'), 'val': g.get('groups_val'), 'test': g.get('groups_test')}

    baseline_rf_train = train_rf_balanced(train_feat_file, groups=group_info['train'] if group_info else None)
    baseline_rf_val = evaluate_rf_balanced_once(train_feat_file, val_feat_file, groups_train=group_info['train'] if group_info else None)
    baseline_rf_test = evaluate_rf_balanced_once(train_feat_file, test_feat_file, groups_train=group_info['train'] if group_info else None)
    baseline_svm_train = eval_svm(train_feat_file)
    baseline_svm_val = eval_svm(val_feat_file)
    baseline_svm_test = eval_svm(test_feat_file)

    print(f"[Main] Baseline RF balanced_acc train: {baseline_rf_train:.4f} val: {baseline_rf_val:.4f} test: {baseline_rf_test:.4f}")
    print(f"[Main] Baseline SVM acc train: {baseline_svm_train:.4f} val: {baseline_svm_val:.4f} test: {baseline_svm_test:.4f}")

    d_train = np.load(train_feat_file)
    X_train, y_train = d_train['X'], d_train['y']
    groups_train = group_info['train'] if group_info else None

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_N = 200
    top_idx = np.argsort(importances)[::-1][:top_N]
    X_filtered = X_train[:, top_idx]
    print("[Main] Feature space reduced for GA. Running simple GA on top 200 features only.")

    best_mask_small = run_ga(X_filtered, y_train, pop_size=40, n_gens=25, mutation_rate=0.02, crossover_rate=0.8)
    best_mask_full = np.zeros(X_train.shape[1], dtype=np.int8)
    best_mask_full[top_idx] = best_mask_small

    np.savez(f'results/{dataset_name}_ga_mask.npz', mask=best_mask_full)
    for split, feat_file in zip(['train', 'val', 'test'], [train_feat_file, val_feat_file, test_feat_file]):
        out = f'results/{dataset_name}_features_{split}_selected.npz'
        apply_feature_mask(feat_file, out, best_mask_full)
        print(f"[Main] Saved selected features for {split}: {out}")

    selected_rf_train = train_rf_balanced(f'results/{dataset_name}_features_train_selected.npz', groups=groups_train)
    selected_rf_val = evaluate_rf_balanced_once(
        npz_train=f'results/{dataset_name}_features_train_selected.npz',
        npz_eval=f'results/{dataset_name}_features_val_selected.npz',
        groups_train=groups_train
    )
    selected_rf_test = evaluate_rf_balanced_once(
        npz_train=f'results/{dataset_name}_features_train_selected.npz',
        npz_eval=f'results/{dataset_name}_features_test_selected.npz',
        groups_train=groups_train
    )
    selected_svm_train = eval_svm(f'results/{dataset_name}_features_train_selected.npz')
    selected_svm_val = eval_svm(f'results/{dataset_name}_features_val_selected.npz')
    selected_svm_test = eval_svm(f'results/{dataset_name}_features_test_selected.npz')

    print("[Main] SELECTED RF balanced_acc train: {:.4f} val: {:.4f} test: {:.4f}".format(selected_rf_train, selected_rf_val, selected_rf_test))
    print("[Main] SELECTED SVM acc train: {:.4f} val: {:.4f} test: {:.4f}".format(selected_svm_train, selected_svm_val, selected_svm_test))
    print("\n[Main] FINAL RESULTS TABLE (RF = balanced_acc, SVM = accuracy)")
    print("{:<20} {:<10} {:<25} {:<25} {:<25}".format('Model', 'Features', 'Train', 'Val', 'Test'))
    print("{:<20} {:<10} {:<25} {:<25} {:<25}".format(
        'Baseline RF', X_train.shape[1],
        f"RF={baseline_rf_train:.4f}  SVM={baseline_svm_train:.4f}",
        f"RF={baseline_rf_val:.4f}   SVM={baseline_svm_val:.4f}",
        f"RF={baseline_rf_test:.4f}   SVM={baseline_svm_test:.4f}"
    ))
    print("{:<20} {:<10} {:<25} {:<25} {:<25}".format(
        'Hybrid RF+GA', int(best_mask_full.sum()),
        f"RF={selected_rf_train:.4f}  SVM={selected_svm_train:.4f}",
        f"RF={selected_rf_val:.4f}   SVM={selected_svm_val:.4f}",
        f"RF={selected_rf_test:.4f}   SVM={selected_svm_test:.4f}"
    ))
    print("[Main] All steps completed successfully.")

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) in [1, 2, 3]:
        dataset_name = input("Enter a dataset name for output files: ").strip()
        groups_npz = os.environ.get("GROUPS_FILE", None)
        force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
        main(args if len(args) > 1 else args[0], dataset_name, groups_file=groups_npz, force_cpu=force_cpu)
    else:
        print("Usage: python main.py <dataset_folder> | python main.py <train> <val> <test> | python main.py <trainval> <test>")
