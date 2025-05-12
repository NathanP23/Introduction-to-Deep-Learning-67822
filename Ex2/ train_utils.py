import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm

def train_autoencoder(model, train_loader, val_loader=None, num_epochs=10, 
                      learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train an autoencoder model
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    output = model(data)
                    loss = criterion(output, data)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            print(f'Epoch: {epoch+1}/{num_epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Time: {time.time() - start_time:.2f}s')
        else:
            print(f'Epoch: {epoch+1}/{num_epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Time: {time.time() - start_time:.2f}s')
    
    return model, history


def evaluate_autoencoder(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate autoencoder on test data
    """
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += criterion(output, data).item()
    
    avg_loss = test_loss / len(test_loader.dataset)
    print(f'Test set: Average loss: {avg_loss:.4f}')
    
    return avg_loss