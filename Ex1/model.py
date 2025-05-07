# ================================
#          model.py
# ================================
import torch
import torch.nn as nn
from config import PEPTIDE_LENGTH, NUM_AMINO_ACIDS, NUM_CLASSES, FC_HIDDEN_DIM, EMB_DIM, MODELS_DIR
import os


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
    
class PeptideToHLAClassifier_2B(PeptideClassifier):
    """
    Simplified binary classifier with just two equal-sized hidden layers.
    No batch normalization, no LeakyReLU, no dropout.
    """
    def __init__(self, emb_dim=EMB_DIM, fc_hidden_dim=FC_HIDDEN_DIM):
        super().__init__(emb_dim)
        input_dim = PEPTIDE_LENGTH * emb_dim
        
        # Simple architecture with two equal-sized hidden layers
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),  # Same dimension as first layer
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
            # No sigmoid with BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)                # (B, 9, emb_dim)
        return self.model(x).squeeze(-1)     # (B,)

class PeptideToHLAClassifier_2C(PeptideClassifier):
    """
    Binary classifier for 9-mer peptides specific to an HLA allele.
    Takes a 9-mer peptide, embeds it, and outputs confidence (0-1) of peptide positivity.
    """
    def __init__(self, emb_dim=EMB_DIM, fc_hidden_dim=FC_HIDDEN_DIM, 
                 dropout_rates=None):
        """
        Initializes the model for binary classification of peptides.
        Args:
            emb_dim (int): Size of the embedding vector for each amino acid.
            fc_hidden_dim (int): Size of the hidden layers.
            dropout_rates (list): Dropout rates for each layer [layer1, layer2, layer3]
        """
        super().__init__(emb_dim)
        
        # Default dropout rates if not provided
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.1]
        
        # More expressive network for binary classification
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(PEPTIDE_LENGTH * emb_dim, fc_hidden_dim * 2),
            nn.BatchNorm1d(fc_hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rates[0]),  # Configurable dropout
            
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rates[1]),  # Configurable dropout
            
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.BatchNorm1d(fc_hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rates[2]),  # Configurable dropout
            
            nn.Linear(fc_hidden_dim // 2, 1),
            # no sigmoid with BCEWithLogitsLoss
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

class PeptideToHLAClassifier_2D(PeptideClassifier):
    """
    Binary classifier for 9-mer peptides specific to an HLA allele.
    Takes a 9-mer peptide, embeds it, and outputs confidence (0-1) of peptide positivity.
    """
    def __init__(self, emb_dim=EMB_DIM, fc_hidden_dim=FC_HIDDEN_DIM, 
                 dropout_rates=None):
        """
        Initializes the model for binary classification of peptides.
        Args:
            emb_dim (int): Size of the embedding vector for each amino acid.
            fc_hidden_dim (int): Size of the hidden layers.
            dropout_rates (list): Dropout rates for each layer [layer1, layer2, layer3]
        """
        super().__init__(emb_dim)
        
        # Default dropout rates if not provided
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.1]
        
        # More expressive network for binary classification
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(PEPTIDE_LENGTH * emb_dim, fc_hidden_dim * 2),
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.Linear(fc_hidden_dim // 2, 1),
            # no sigmoid with BCEWithLogitsLoss
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
    

class MultiAlleleEnsemble(nn.Module):
    """
    Wraps six PeptideToHLAClassifier_2C instances.
    Forward(x) → tensor of shape (B, 6) in a **fixed alphabetical order**:
        ['A0101', 'A0201', 'A0203', 'A0207', 'A0301', 'A2402']
    """

    DEFAULT_ORDER = ['A0101', 'A0201', 'A0203', 'A0207', 'A0301', 'A2402']

    def __init__(self, allele_models: dict):
        """
        allele_models: {allele_name: trained PeptideToHLAClassifier_2C}
                       **Must contain exactly the six keys above.**
        """
        super().__init__()
        # Preserve order for deterministic output vector
        self.allele_order = self.DEFAULT_ORDER
        self.models = nn.ModuleDict(allele_models)

    def forward(self, x):
        # Each sub-model returns shape (B); stack→(6,B) then transpose→(B,6)
        logits = [self.models[a](x) for a in self.allele_order]
        return torch.stack(logits).T    # (B,6)
