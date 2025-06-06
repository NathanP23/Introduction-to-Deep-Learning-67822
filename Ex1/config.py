# ================================
#          config.py
# ================================
from pathlib import Path

# Data configuration
DATA_DIR = Path("Data/HLA_Dataset")
MODELS_DIR = Path("Models")
ALLELE_POSITIVE_FILES = [f for f in DATA_DIR.glob("*.txt") if "neg" not in f.name]
NEGATIVE_FILE = DATA_DIR / "negs.txt"
TRAIN_RATIO = 0.9

# Model configuration
PEPTIDE_LENGTH = 9
NUM_AMINO_ACIDS = 20
NUM_CLASSES = 7
EMB_DIM = 32
FC_HIDDEN_DIM = 128

# Training configuration
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

