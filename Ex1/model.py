# model.py
import torch
import torch.nn as nn
from config import PEPTIDE_LENGTH, NUM_AMINO_ACIDS, NUM_CLASSES


class PeptideClassifier(nn.Module):
    """
    Feedforward classifier for 9-mer peptide sequences using fixed-width hidden layers.
    - Embedding layer: (20 → emb_dim)
    - Flattened input
    - Two hidden layers of same dimension (as required)
    - Output layer: 7 classes
    """

    def __init__(self, emb_dim=4):
        """
        Initializes the model layers.
        Args:
            emb_dim (int): Size of the embedding vector for each amino acid.
        """
        super().__init__()

        self.embedding = nn.Embedding(NUM_AMINO_ACIDS, emb_dim)
        input_dim = PEPTIDE_LENGTH * emb_dim

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, NUM_CLASSES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (Tensor): Peptide input of shape (B, 9)
        Returns:
            Tensor: Class logits of shape (B, 7)
        """
        x = self.embedding(x)  # (B, 9, emb_dim)
        return self.model(x)   # (B, 7)


def save_model(model, filename='fc_model.pt'):
    """
    Save the trained model to a file
    """
    torch.save(model.state_dict(), filename)


def load_model(emb_dim=4, filename='fc_model.pt'):
    """
    Load a trained model from a file
    """
    model = PeptideClassifier(emb_dim=emb_dim)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model
