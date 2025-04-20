# config.py
from pathlib import Path

# Data configuration
DATA_DIR = Path("Data/HLA_Dataset")
TRAIN_RATIO = 0.9

# Model configuration
PEPTIDE_LENGTH = 9
NUM_AMINO_ACIDS = 20
NUM_CLASSES = 7
EMB_DIM = 4

# Training configuration
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Amino acid representation
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'