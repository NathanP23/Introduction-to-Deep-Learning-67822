# constants.py
from config import AMINO_ACIDS

# Allele to label mapping
ALLELE_LABEL_MAP = {
    "A0101": 1,
    "A0201": 2,
    "A0203": 3,
    "A0207": 4,
    "A0301": 5,
    "A2402": 6,
    "NEG": 0
}

# Amino acid to index mapping
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
