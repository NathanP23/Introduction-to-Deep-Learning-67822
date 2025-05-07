# ================================
#          hyperparameters.py
# ================================
from config import (
    EMB_DIM, FC_HIDDEN_DIM, BATCH_SIZE,
    LEARNING_RATE, EPOCHS
)

# ---- base settings applied to every allele ----
BASE_HP = dict(
    EMB_DIM          = EMB_DIM,
    FC_HIDDEN_DIM    = FC_HIDDEN_DIM,
    loss_function    = 'BCEWithLogitsLoss',   # raw-logit version
    BATCH_SIZE       = BATCH_SIZE,
    LEARNING_RATE    = LEARNING_RATE,
    EPOCHS           = EPOCHS,
    THRESHOLD        = 0.50,                  # default τ
    PATIENCE         = 5,                     # early stopping patience
    DROPOUT_RATES    = [0.3, 0.2, 0.1],       # dropout rates for 3 layers
)

# ---- per-allele overrides ---------------------
allele_data_hyperparameters = {
    # A0101 - performance peaks earlier than other alleles
    'A0101': {
        **BASE_HP,
        'EPOCHS': 15,
        'LEARNING_RATE': 8e-4,
        'PATIENCE': 4,
        'DROPOUT_RATES': [0.4, 0.3, 0.2],  # Increase dropout to reduce overfitting
    },
    
    # A0201 - good performance but tends to favor negative class later
    'A0201': {
        **BASE_HP,
        'THRESHOLD': 0.45,  # Lower threshold to improve positive class prediction
        'DROPOUT_RATES': [0.4, 0.3, 0.2],  # Increased dropout
    },
    
    # A0203 - best performance, minimal changes needed
    'A0203': {
        **BASE_HP,
        'EPOCHS': 16,
        'PATIENCE': 6,  # Longer patience due to stable performance
    },

    # A0207 - test loss increases after epoch 7
    'A0207': {
        **BASE_HP,
        'EPOCHS': 15,
        'LEARNING_RATE': 5e-4,
        'THRESHOLD': 0.52,  # Balanced threshold
        'PATIENCE': 4,
    },

    # A0301 - severe overfitting, needs significant changes
    'A0301': {
        **BASE_HP,
        'EMB_DIM': 32,  # Reduce embedding size
        'FC_HIDDEN_DIM': 96,  # Reduce model capacity
        'EPOCHS': 15,
        'LEARNING_RATE': 3e-4,
        'THRESHOLD': 0.40,  # Lower threshold to improve recall
        'DROPOUT_RATES': [0.5, 0.4, 0.3],  # Heavy dropout
        'PATIENCE': 3,  # Shorter patience
    },

    # A2402 - good overall but test loss increases
    'A2402': {
        **BASE_HP,
        'THRESHOLD': 0.55,  # Slightly higher threshold
        'PATIENCE': 5,
    },
}