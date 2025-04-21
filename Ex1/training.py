# training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, EMB_DIM
from model import PeptideToHLAClassifier


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
    if loss_function == 'BCELoss':
        # For the binary classifier (PeptideToHLAClassifier), use BCELoss
        loss_fn = nn.BCELoss()
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


def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=EPOCHS, threshold=0.5):
    """
    Training loop for the model
    """
    # Metrics Tracking
    train_losses = []
    test_losses = []
    accuracies = []
    is_binary = isinstance(model, PeptideToHLAClassifier)
    
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
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                
                # Handle loss calculation differently for binary vs multi-class
                if is_binary:
                    # Convert y to float for BCE loss
                    batch_y_float = batch_y.float()
                    loss = loss_fn(outputs, batch_y_float)
                    
                    # For binary classification, threshold at 0.5
                    preds = (outputs >= threshold).long()
                else:
                    loss = loss_fn(outputs, batch_y)
                    # For multi-class, take argmax
                    preds = outputs.argmax(dim=1)
                
                test_loss += loss.item() * batch_x.size(0)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_accuracy = 100 * correct / total
        test_losses.append(epoch_test_loss)
        accuracies.append(epoch_accuracy)
        
        # Log Results
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Test Loss: {epoch_test_loss:.4f} | "
              f"Accuracy: {epoch_accuracy:.2f}%")
    
    return train_losses, test_losses, accuracies
