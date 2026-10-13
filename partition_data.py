"""
partition_data.py — Preprocess and Partition WISDM Data for Multi-Computer FL
=============================================================================
Splits the 51 subjects from analysis_output/features_phone_accel.csv into:
  1. client_1_data.pt: Data for Computer 1 (local client)
  2. client_2_data.pt: Data for Computer 2 (remote client)
  3. holdout_test.pt:  Holdout data for Server evaluation (10 unseen subjects)
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset

OUTPUT_DIR = "analysis_output"
CSV_PATH = os.path.join(OUTPUT_DIR, "features_phone_accel.csv")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def partition_and_save():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Ensure data has been preprocessed.")

    print(f"Reading features from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    meta_cols = ['activity', 'activity_name', 'activity_group', 'subject', 'device', 'sensor']
    feat_cols = [c for c in df.columns if c not in meta_cols]

    group_mapping = {
        'Sedentary': 0,
        'Eating': 1,
        'Hand-Oriented': 2,
        'Locomotion': 3,
        'Sports': 4
    }
    df['target'] = df['activity_group'].map(group_mapping)

    X = df[feat_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    subjects = df['subject'].values

    # Z-score normalization
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-7
    X = (X - mean) / std

    unique_subjects = sorted(np.unique(subjects))
    print(f"Total samples: {len(df)} across {len(unique_subjects)} subjects.")

    shuffled_subjects = np.random.permutation(unique_subjects)
    # Split: 10 holdout subjects, remaining 41 subjects split between Computer 1 and Computer 2
    holdout_subjects = shuffled_subjects[:10]
    train_subjects = shuffled_subjects[10:]

    mid = len(train_subjects) // 2
    c1_subjects = train_subjects[:mid]
    c2_subjects = train_subjects[mid:]

    print(f"Client 1 Subjects ({len(c1_subjects)}): {c1_subjects.tolist()}")
    print(f"Client 2 Subjects ({len(c2_subjects)}): {c2_subjects.tolist()}")
    print(f"Holdout Subjects  ({len(holdout_subjects)}): {holdout_subjects.tolist()}")

    # Extract tensors
    c1_mask = np.isin(subjects, c1_subjects)
    c2_mask = np.isin(subjects, c2_subjects)
    holdout_mask = np.isin(subjects, holdout_subjects)

    c1_data = {
        'X': torch.tensor(X[c1_mask]),
        'y': torch.tensor(y[c1_mask]),
        'subjects': c1_subjects.tolist(),
        'num_features': X.shape[1],
        'num_classes': len(group_mapping)
    }

    c2_data = {
        'X': torch.tensor(X[c2_mask]),
        'y': torch.tensor(y[c2_mask]),
        'subjects': c2_subjects.tolist(),
        'num_features': X.shape[1],
        'num_classes': len(group_mapping)
    }

    holdout_data = {
        'X': torch.tensor(X[holdout_mask]),
        'y': torch.tensor(y[holdout_mask]),
        'subjects': holdout_subjects.tolist(),
        'num_features': X.shape[1],
        'num_classes': len(group_mapping)
    }

    p1 = os.path.join(OUTPUT_DIR, "client_1_data.pt")
    p2 = os.path.join(OUTPUT_DIR, "client_2_data.pt")
    ph = os.path.join(OUTPUT_DIR, "holdout_test.pt")

    torch.save(c1_data, p1)
    torch.save(c2_data, p2)
    torch.save(holdout_data, ph)

    print(f"Saved: {p1} ({len(c1_data['X'])} samples)")
    print(f"Saved: {p2} ({len(c2_data['X'])} samples)")
    print(f"Saved: {ph} ({len(holdout_data['X'])} samples)")

if __name__ == "__main__":
    partition_and_save()
