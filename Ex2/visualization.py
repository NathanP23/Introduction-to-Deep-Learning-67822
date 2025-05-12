import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_reconstructions(model, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu', num_samples=10):
    """
    Plot original images and their reconstructions
    """
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(data_loader))
        data = data[:num_samples].to(device)
        reconstruction = model(data)
        
        # Convert to numpy for plotting
        data = data.cpu().numpy()
        reconstruction = reconstruction.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        # Original images
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(data[i][0], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
            
        # Reconstructed images
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(reconstruction[i][0], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.tight_layout()
    plt.show()


def plot_latent_space(model, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu', num_samples=1000):
    """
    Visualize the latent space (for 2D latent space)
    """
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            z = model.encode(data)
            latent_vectors.append(z.cpu().numpy())
            labels.append(label.numpy())
            
            if len(labels) * len(label) >= num_samples:
                break
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    if latent_vectors.shape[1] != 2:
        print(f"Latent dimension is {latent_vectors.shape[1]}, not 2. Cannot visualize directly.")
        return
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, 
                         cmap='tab10', alpha=0.8, s=5)
    plt.colorbar(scatter)
    plt.title('2D Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(alpha=0.3)
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation loss curves
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()