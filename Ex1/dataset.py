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
    Load peptide data from files and split into train/test sets.
    Returns 6 datasets, one for each positive allele, with binary classification (1 for positive, 0 for negative).
    Each dataset has independently shuffled negatives.
    """
    # Locate all positive allele files
    allele_files = [f for f in DATA_DIR.glob("*.txt") if "neg" not in f.name]
    
    # Dictionary to store data for each allele
    allele_data = {}
    
    # Process each allele file separately
    for file in allele_files:
        allele = file.stem.replace("_pos", "")
        with open(file) as f:
            peptides = [line.strip() for line in f if line.strip()]
            random.shuffle(peptides)
            split_idx = int(len(peptides) * TRAIN_RATIO)
            # Store as positive examples (label 1) for this allele
            train_pos = [(pep, allele, 1) for pep in peptides[:split_idx]]
            test_pos = [(pep, allele, 1) for pep in peptides[split_idx:]]
            allele_data[allele] = {"train_pos": train_pos, "test_pos": test_pos}
    
    # Read all negative peptides once
    with open(DATA_DIR / "negs.txt") as f:
        neg_peptides = [line.strip() for line in f if line.strip()]
    
    # Create separate datasets for each allele
    datasets = {}
    for allele in allele_data:
        # Get a fresh copy of negatives and shuffle them independently for each allele
        neg_peptides_copy = neg_peptides.copy()
        random.shuffle(neg_peptides_copy)
        split_idx = int(len(neg_peptides_copy) * TRAIN_RATIO)
        
        # Create negative samples for this specific allele
        neg_train = [(pep, "NEG", 0) for pep in neg_peptides_copy[:split_idx]]
        neg_test = [(pep, "NEG", 0) for pep in neg_peptides_copy[split_idx:]]
        
        # Combine positives for this specific allele with the independently shuffled negatives
        train_data = allele_data[allele]["train_pos"] + neg_train
        test_data = allele_data[allele]["test_pos"] + neg_test
        
        # Shuffle the combined data
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        # Store the final datasets
        datasets[allele] = (train_data, test_data)
        
        print(f"\n--- {allele} Dataset ---")
        print(f"Train set size: {len(train_data)} ({((len(train_data) / (len(train_data) + len(test_data))) * 100):.2f}%))")
        print(f"Test set size: {len(test_data)} ({((len(test_data) / (len(train_data) + len(test_data))) * 100):.2f}%)")
        print_dist(train_data, f"{allele} Train")
        print_dist(test_data, f"{allele} Test")
    
    return datasets


def peptide_to_indices(peptide):
    """
    Convert peptide (string of amino acids) to list of indices
    """
    return [AA_TO_IDX[aa] for aa in peptide]


def prepare_data(datasets):
    """
    Convert data to PyTorch tensors for all 6 allele datasets
    
    Parameters:
    - datasets: Dictionary containing datasets for all alleles
      Format: {allele_name: (train_data, test_data), ...}
    
    Returns:
    - Dictionary of processed datasets for all alleles
      Format: {allele_name: (X_train, y_train, X_test, y_test), ...}
    """
    processed_datasets = {}
    
    for allele, (train_data, test_data) in datasets.items():
        print(f"\n--- Processing {allele} Dataset ---")
        
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
        
        print(f"{allele} X_train shape:", X_train.shape)
        print(f"{allele} y_train shape:", y_train.shape)
        print(f"{allele} X_test shape:", X_test.shape)
        print(f"{allele} y_test shape:", y_test.shape)
        
        # Check a sample
        print(f"\n{allele} example input (peptide indices):", X_train[0])
        print(f"{allele} corresponding label (binary):", y_train[0])
        
        # Store processed data for this allele
        processed_datasets[allele] = (X_train, y_train, X_test, y_test)
    return processed_datasets


def save_dataset(processed_datasets):
    """
    Save processed datasets to files
    
    Parameters:
    - processed_datasets: Dictionary of datasets by allele
      Format: {allele_name: (X_train, y_train, X_test, y_test), ...}
    """
    # Save each allele's dataset to a separate file
    for allele, (X_train, y_train, X_test, y_test) in processed_datasets.items():
        torch.save({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }, f'saved_{allele}_dataset.pt')
    
    print("All datasets saved successfully!")


def load_dataset(allele=None):
    """
    Load processed datasets from files
    
    Parameters:
    - allele: If provided, load only this allele's dataset
    
    Returns:
    - If allele specified: Tuple (X_train, y_train, X_test, y_test) for that allele
    - If no allele specified: Dictionary of datasets by allele
      Format: {allele_name: (X_train, y_train, X_test, y_test), ...}
    """
    if allele is not None:
        try:
            data = torch.load(f'saved_{allele}_dataset.pt')
            return (
                data['X_train'],
                data['y_train'],
                data['X_test'],
                data['y_test']
            )
        except FileNotFoundError:
            raise ValueError(f"No saved dataset found for allele {allele}")
    
    # Load all datasets
    from constants import ALLELE_LABEL_MAP
    alleles = [a for a in ALLELE_LABEL_MAP if a != "NEG"]
    
    all_datasets = {}
    for allele in alleles:
        try:
            data = torch.load(f'saved_{allele}_dataset.pt')
            all_datasets[allele] = (
                data['X_train'],
                data['y_train'],
                data['X_test'],
                data['y_test']
            )
        except FileNotFoundError:
            print(f"Warning: No saved dataset found for allele {allele}")
    
    if not all_datasets:
        raise FileNotFoundError("No saved datasets found")
        
    return all_datasets
