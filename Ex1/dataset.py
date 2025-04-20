# dataset.py
import os
import random
import torch
from pathlib import Path
from collections import Counter
from constants import ALLELE_LABEL_MAP, AA_TO_IDX
from config import DATA_DIR, TRAIN_RATIO


def print_dist(data, name):
    """
    Print distribution of labels in a dataset
    """
    counter = Counter([label for _, _, label in data])
    total = sum(counter.values())
    print(f"\n{name} set distribution:")
    for lbl in sorted(counter):
        allele = [a for a, l in ALLELE_LABEL_MAP.items() if l == lbl][0]
        print(f"  {allele:6s} (label {lbl}): {counter[lbl]} samples ({100 * counter[lbl]/total:.2f}%)")


def load_and_split_data():
    """
    Load peptide data from files and split into train/test sets
    """
    # Locate all positive allele files
    allele_files = [f for f in DATA_DIR.glob("*.txt") if "neg" not in f.name]
    
    # Store train/test samples
    pos_train, pos_test = [], []
    
    # Process each allele file separately
    for file in allele_files:
        allele = file.stem.replace("_pos", "")
        label = ALLELE_LABEL_MAP[allele]
        with open(file) as f:
            peptides = [line.strip() for line in f if line.strip()]
            random.shuffle(peptides)
            split_idx = int(len(peptides) * TRAIN_RATIO)
            pos_train += [(pep, allele, label) for pep in peptides[:split_idx]]
            pos_test  += [(pep, allele, label) for pep in peptides[split_idx:]]
    
    # Process negatives
    with open(DATA_DIR / "negs.txt") as f:
        neg_peptides = [line.strip() for line in f if line.strip()]
        random.shuffle(neg_peptides)
        split_idx = int(len(neg_peptides) * TRAIN_RATIO)
        neg_train = [(pep, "NEG", 0) for pep in neg_peptides[:split_idx]]
        neg_test  = [(pep, "NEG", 0) for pep in neg_peptides[split_idx:]]
    
    # Combine and shuffle
    train_data = pos_train + neg_train
    test_data = pos_test + neg_test
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f"Train set size: {len(train_data)} ({((len(train_data) / (len(train_data) + len(test_data))) * 100):.2f}%))")
    print(f"Test set size: {len(test_data)} ({((len(test_data) / (len(train_data) + len(test_data))) * 100):.2f}%)")
    print_dist(train_data, "Train")
    print_dist(test_data, "Test")
    
    return train_data, test_data


def peptide_to_indices(peptide):
    """
    Convert peptide (string of amino acids) to list of indices
    """
    return [AA_TO_IDX[aa] for aa in peptide]


def prepare_data(train_data, test_data):
    """
    Convert data to PyTorch tensors
    """
    # Split features (X) and labels (y) for train and test
    X_train = [peptide_to_indices(p) for p, _, _ in train_data]
    y_train = [label for _, _, label in train_data]
    
    X_test = [peptide_to_indices(p) for p, _, _ in test_data]
    y_test = [label for _, _, label in test_data]
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
    # Check a sample
    print("\nExample input (peptide indices):", X_train[0])
    print("Corresponding label (allele class):", y_train[0])
    
    return X_train, y_train, X_test, y_test


def save_dataset(X_train, y_train, X_test, y_test):
    """
    Save processed datasets to files
    """
    torch.save({
        'X_train': X_train,
        'y_train': y_train
    }, 'saved_train_dataset.pt')
    
    torch.save({
        'X_test': X_test,
        'y_test': y_test
    }, 'saved_test_dataset.pt')


def load_dataset():
    """
    Load processed datasets from files
    """
    train_data = torch.load('saved_train_dataset.pt')
    test_data = torch.load('saved_test_dataset.pt')
    
    return (
        train_data['X_train'],
        train_data['y_train'],
        test_data['X_test'],
        test_data['y_test']
    )
