# ================================
#          model.py
# ================================
import torch
import torch.nn as nn
from config import PEPTIDE_LENGTH, NUM_AMINO_ACIDS, NUM_CLASSES, FC_HIDDEN_DIM, EMB_DIM


class PeptideClassifier(nn.Module):
    """
    Base class for peptide sequence models.
    - Embedding layer: (20 → emb_dim)
    - Basic structure for peptide classification
    """
    
    def __init__(self, emb_dim=EMB_DIM):
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

    def __init__(self, emb_dim=EMB_DIM):
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


class PeptideToHLAClassifier_2C(PeptideClassifier):
    """
    Binary classifier for 9-mer peptides specific to the A0201 allele.
    Takes a 9-mer peptide, embeds it, and outputs confidence (0-1) of peptide positivity.
    """
    def __init__(self, emb_dim=EMB_DIM, fc_hidden_dim=FC_HIDDEN_DIM):
        """
        Initializes the model for binary classification of peptides.
        Args:
            emb_dim (int): Size of the embedding vector for each amino acid.
            fc_hidden_dim (int): Size of the hidden layers.
        """
        super().__init__(emb_dim)

        # More expressive network for binary classification
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(PEPTIDE_LENGTH * emb_dim, fc_hidden_dim * 2),
            nn.BatchNorm1d(fc_hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),  # Add dropout for regularization
            
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.BatchNorm1d(fc_hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(fc_hidden_dim // 2, 1),
            nn.Sigmoid()  # Outputs value between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (Tensor): Peptide input of shape (B, 9)
        Returns:
            Tensor: Confidence score (0-1) of shape (B)
        """
        x = self.embedding(x)  # Use the embedding from the parent class
        return self.model(x).squeeze(-1)  # Shape (B)
        
    def predict(self, peptide_indices):
        """
        Helper method to predict on a single peptide or batch.
        Args:
            peptide_indices: Tensor of shape (9) or (B, 9) with peptide amino acid indices
        Returns:
            Confidence score (0-1) that the peptide is positive for A0201
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Add batch dimension if needed
            if peptide_indices.dim() == 1:
                peptide_indices = peptide_indices.unsqueeze(0)
            return self(peptide_indices)
    
def save_model(model, filename='fc_model.pt'):
    """
    Save the trained model to a file
    """
    torch.save(model.state_dict(), filename)


def load_model(model_class=PeptideClassifier2b, emb_dim=EMB_DIM, filename='fc_model.pt', **kwargs):
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