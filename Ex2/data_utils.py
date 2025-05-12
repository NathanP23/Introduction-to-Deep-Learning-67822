import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=128, val_split=0.1, download=True):
    """
    Get MNIST training, validation, and test data loaders
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load the full training dataset
    full_train_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        download=download,
        transform=transform
    )
    
    # Split into training and validation sets
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load the test dataset
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False,
        download=download,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader