# training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, EMB_DIM


def create_data_loaders(X_train, y_train, X_test, y_test):
    """
    Create PyTorch DataLoaders for training and testing
    """
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader


def setup_training(model, y_train):
    """
    Initialize loss function and optimizer
    """
    # Loss Function: CrossEntropy
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum()  # Normalize
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Model, optimizer, and loss function initialized!")
    return loss_fn, optimizer


def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=EPOCHS):
    """
    Training loop for the model
    """
    # Metrics Tracking
    train_losses = []
    test_losses = []
    accuracies = []
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()            # 1. Reset gradients
            logits = model(batch_x)          # 2. Forward pass
            loss = loss_fn(logits, batch_y)  # 3. Compute loss
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
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                
                preds = logits.argmax(dim=1)
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
