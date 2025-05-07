# ================================
#          dataset.py
# ================================
import random
import torch
from collections import Counter
from constants import AA_TO_IDX
from config import TRAIN_RATIO, ALLELE_POSITIVE_FILES, NEGATIVE_FILE

def load_and_split_data():
    """
    Load peptide data from files and split into train/test sets.
    Returns 6 datasets, one for each positive allele, with binary classification (1 for positive, 0 for negative).
    Each dataset has independently shuffled negatives.
    """
    
    # Dictionary to store data for each allele
    allele_data = {}
    
    # Process each allele file separately
    for file in ALLELE_POSITIVE_FILES:
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
    with open(NEGATIVE_FILE) as f:
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
    summarize_datasets(datasets)
    return datasets

import pandas as pd
from collections import Counter
import torch

# Add this function at the end of dataset.py
def summarize_datasets(datasets):
    """
    Summarize datasets into a Pandas DataFrame
    """
    summary_data = []
    for allele, (train_data, test_data) in datasets.items():
        train_counter = Counter([label for _, _, label in train_data])
        test_counter = Counter([label for _, _, label in test_data])
        
        train_total = len(train_data)
        test_total = len(test_data)
        total = train_total + test_total

        summary_data.append({
            'Allele': allele,
            'Train Size': train_total,
            'Test Size': test_total,
            'Train (%)': f"{100 * train_total / total:.2f}%",
            'Test (%)': f"{100 * test_total / total:.2f}%",
            'Train Positive (%)': f"{100 * train_counter[1] / train_total:.2f}%",
            'Train Negative (%)': f"{100 * train_counter[0] / train_total:.2f}%",
            'Test Positive (%)': f"{100 * test_counter[1] / test_total:.2f}%",
            'Test Negative (%)': f"{100 * test_counter[0] / test_total:.2f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    print("Dataset Summary:")
    print(summary_df)

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
        
        # Store processed data for this allele
        processed_datasets[allele] = (X_train, y_train, X_test, y_test)
    summarize_processed_datasets(processed_datasets)
    return processed_datasets

def peptide_to_indices(peptide):
    """
    Convert peptide (string of amino acids) to list of indices
    """
    return [AA_TO_IDX[aa] for aa in peptide]

def summarize_processed_datasets(processed_datasets):
    """
    Summarize processed tensor datasets into a Pandas DataFrame
    """
    summary_data = []

    for allele, (X_train, y_train, X_test, y_test) in processed_datasets.items():
        summary_data.append({
            'Allele': allele,
            'X_train shape': X_train.shape,
            'y_train shape': y_train.shape,
            'X_test shape': X_test.shape,
            'y_test shape': y_test.shape,
            'Example input (indices)': X_train[0].tolist(),
            'Example label (binary)': y_train[0].item()
        })

    summary_df = pd.DataFrame(summary_data)
    print("Processed Dataset Summary:")
    print(summary_df)
