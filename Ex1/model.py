# model.py
import torch
import torch.nn as nn
from config import PEPTIDE_LENGTH, NUM_AMINO_ACIDS, NUM_CLASSES, FC_HIDDEN_DIM


class PeptideClassifier(nn.Module):
    """
    Base class for peptide sequence models.
    - Embedding layer: (20 → emb_dim)
    - Basic structure for peptide classification
    """
    
    def __init__(self, emb_dim=4):
        """
        Initializes the base model with embedding layer.
        Args:
            emb_dim (int): Size of the embedding vector for each amino acid.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(NUM_AMINO_ACIDS, emb_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for embedding layer
        Args:
            x (Tensor): Peptide input of shape (B, 9)
        Returns:
            Tensor: Embedded representation
        """
        return self.embedding(x)  # (B, 9, emb_dim)


class PeptideClassifier2b(PeptideClassifier):
    """
    Standard feedforward classifier for 9-mer peptide sequences using fixed-width hidden layers.
    - Identical to the original PeptideClassifier implementation
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
        super().__init__(emb_dim)
        input_dim = PEPTIDE_LENGTH * self.emb_dim

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
        x = super().forward(x)  # (B, 9, emb_dim)
        return self.model(x)   # (B, 7)


class PeptideClassifier2c(PeptideClassifier):
    """
    Advanced classifier for 9-mer peptide sequences using Tal architecture.
    - Embedding layer: (20 → emb_dim)
    - Output layer: 7 classes
    """

    def __init__(self, emb_dim=4, num_filters=16):
        """
        Initializes the advanced model layers.
        Args:
            emb_dim (int): Size of the embedding vector for each amino acid.
        """
        super().__init__(emb_dim)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(PEPTIDE_LENGTH * emb_dim, FC_HIDDEN_DIM * 2),
            nn.Sigmoid(),
            nn.Linear(FC_HIDDEN_DIM * 2, FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_DIM, FC_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_DIM // 2, FC_HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_DIM // 4, FC_HIDDEN_DIM // 8),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_DIM // 8, NUM_CLASSES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.model(x)


def save_model(model, filename='fc_model.pt'):
    """
    Save the trained model to a file
    """
    torch.save(model.state_dict(), filename)


def load_model(model_class=PeptideClassifier2b, emb_dim=4, filename='fc_model.pt', **kwargs):
    """
    Load a trained model from a file
    
    Args:
        model_class: The model class to instantiate (default: PeptideClassifier2b)
        emb_dim (int): Size of the embedding vector for each amino acid
        filename (str): Path to the saved model file
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        The loaded model in evaluation mode
    """
    model = model_class(emb_dim=emb_dim, **kwargs)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model