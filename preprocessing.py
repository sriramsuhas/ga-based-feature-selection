import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

def get_all_class_folders(root_folder):
    class_folders = []
    # walk deterministically: sort directory names
    for dirpath, dirnames, filenames in os.walk(root_folder):
        image_files = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        if image_files:
            class_folders.append(dirpath)
    # sort for deterministic order
    class_folders = sorted(class_folders)
    return class_folders

def extract_and_split_dataset(dataset_folder, out_dir):
    print(f"[Preprocessing] Using dataset folder: {dataset_folder}")

    class_folders = get_all_class_folders(dataset_folder)

    X = []
    y = []
    print("[Preprocessing] Scanning classes and images...")

    for class_folder in class_folders:
        label = os.path.basename(class_folder)
        img_list = [
            os.path.join(class_folder, img)
            for img in sorted(os.listdir(class_folder))
            if os.path.isfile(os.path.join(class_folder, img)) and img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
        ]
        if len(img_list) < 3:
            print(f"[Preprocessing] Warning: Skipping class '{label}' (not enough images to split).")
            continue
        for img in img_list:
            X.append(img)
            y.append(label)

    X, y = np.array(X), np.array(y)
    if len(X) < 3:
        raise RuntimeError("[Preprocessing] Not enough images for splitting.")

    print("[Preprocessing] Performing train/val/test split...")
    # Use fixed random_state for reproducibility. stratify to keep class ratios.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    # split X_temp into val/test (use test_size such that val:test = 1:2 since earlier code expected ~30% then val/test distribution)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.666, stratify=y_temp, random_state=42)

    def copy_images(filepaths, labels, split):
        split_folder = os.path.join(out_dir, split)
        os.makedirs(split_folder, exist_ok=True)
        for filepath, label in zip(filepaths, labels):
            label_folder = os.path.join(split_folder, label)
            os.makedirs(label_folder, exist_ok=True)
            shutil.copy(filepath, os.path.join(label_folder, os.path.basename(filepath)))

    print(f"[Preprocessing] Train set: {len(X_train)} images.")
    print(f"[Preprocessing] Validate set: {len(X_val)} images.")
    print(f"[Preprocessing] Test set: {len(X_test)} images.")
    copy_images(X_train, y_train, "train")
    copy_images(X_val, y_val, "validate")
    copy_images(X_test, y_test, "test")
    print("[Preprocessing] Data splitting completed!")

    return {
        'train': os.path.join(out_dir, "train"),
        'validate': os.path.join(out_dir, "validate"),
        'test': os.path.join(out_dir, "test")
    }
