# ================================
#          training.py
# ================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import BATCH_SIZE, EPOCHS
from model import PeptideToHLAClassifier_2C, PeptideToHLAClassifier_2B, PeptideToHLAClassifier_2D

def _collect_outputs(model, loader):
    """
    Run `model` once over `loader` and return a single 1-D tensor
    of the raw sigmoid outputs (length = number of samples).
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            outs.append(model(x).cpu())
    return torch.cat(outs)

def create_data_loaders(X_train, y_train, X_test, y_test):
    """
    Create PyTorch DataLoaders for training and testing
    """
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader


def setup_training(model, loss_function, learning_rate, y_train):
    """
    Initialize loss function and optimizer
    """
    # For the binary classifier (PeptideToHLAClassifier_2C), use BCELoss
    if loss_function == 'BCEWithLogitsLoss':
        pos_frac   = y_train.float().mean()
        pos_weight = (1 - pos_frac) / pos_frac
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.clone().detach())
    else:
        # For multi-class classifiers, use CrossEntropyLoss
        # Loss Function: CrossEntropy
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum()  # Normalize

        # we dont use weights yet
        loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=EPOCHS, threshold=0.5, 
                patience=5, return_final_outputs=False):
    """
    Training loop for the model with early stopping and learning rate scheduling
    """
    # Metrics Tracking
    train_losses = []
    test_losses = []
    accuracies = []
    pos_accuracies = []
    neg_accuracies = []
    is_binary = isinstance(model, PeptideToHLAClassifier_2C) or isinstance(model, PeptideToHLAClassifier_2B) or isinstance(model, PeptideToHLAClassifier_2D)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()            # 1. Reset gradients
            
            outputs = model(batch_x)         # 2. Forward pass
            
            # Handle loss calculation differently for binary vs multi-class
            if is_binary:
                # Convert y to float for BCE loss
                batch_y_float = batch_y.float()
                loss = loss_fn(outputs, batch_y_float)
            else:
                loss = loss_fn(outputs, batch_y)
                
            loss.backward()                  # 4. Backpropagation
            optimizer.step()                 # 5. Update weights
            
            running_loss += loss.item() * batch_x.size(0)  # track loss
        
        # Epoch Summary
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluate on Test Set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        pos_correct = 0
        pos_total = 0
        neg_correct = 0
        neg_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                
                # Handle loss calculation differently for binary vs multi-class
                if is_binary:
                    # Convert y to float for BCE loss
                    batch_y_float = batch_y.float()
                    loss = loss_fn(outputs, batch_y_float)
                    
                    # For binary classification, threshold at 0.5
                    sig_outputs = torch.sigmoid(outputs)
                    preds = (sig_outputs >= threshold).long()
                    
                    # Track per-class metrics
                    pos_mask = batch_y == 1
                    neg_mask = batch_y == 0
                    
                    pos_correct += ((preds == 1) & pos_mask).sum().item()
                    pos_total += pos_mask.sum().item()
                    
                    neg_correct += ((preds == 0) & neg_mask).sum().item()
                    neg_total += neg_mask.sum().item()
                else:
                    loss = loss_fn(outputs, batch_y)
                    # For multi-class, take argmax
                    preds = outputs.argmax(dim=1)
                
                test_loss += loss.item() * batch_x.size(0)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        # After calculating test_loss:
        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_accuracy = 100 * correct / total
        
        # Calculate per-class accuracy
        epoch_pos_accuracy = 100 * pos_correct / pos_total if pos_total > 0 else 0
        epoch_neg_accuracy = 100 * neg_correct / neg_total if neg_total > 0 else 0
        
        # Store all metrics
        test_losses.append(epoch_test_loss)
        accuracies.append(epoch_accuracy)
        pos_accuracies.append(epoch_pos_accuracy)
        neg_accuracies.append(epoch_neg_accuracy)
        
        # Update learning rate based on validation performance
        scheduler.step(epoch_test_loss)
        
        # Log Results with per-class accuracy
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Test Loss: {epoch_test_loss:.4f} | "
              f"Overall Accuracy: {epoch_accuracy:.2f}% | "
              f"Pos Accuracy: {epoch_pos_accuracy:.2f}% | "
              f"Neg Accuracy: {epoch_neg_accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
              
        # Check for early stopping
        if early_stopping(epoch_test_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
              
    final_outputs = _collect_outputs(model, test_loader) if return_final_outputs else None
    
    # Return all metrics including per-class accuracies
    return train_losses, test_losses, accuracies, pos_accuracies, neg_accuracies, final_outputs
    