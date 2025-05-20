# mnistlib/train.py
import torch
import torch.nn as nn
import time
import tqdm

def train_autoencoder_model(autoencoder, train_dataloader, validation_dataloader,
               epochs=10, learning_rate=2e-3, weight_decay=0, device=None, 
               pixel_accuracy_threshold=0.1, print_every=5):
    """
    Train an autoencoder model using the provided data loaders.
    
    Args:
        autoencoder (nn.Module): The autoencoder model to train
        train_dataloader (DataLoader): DataLoader containing training data
        validation_dataloader (DataLoader): DataLoader containing validation data
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        weight_decay (float): L2 regularization strength
        device (str): Device to use for training ('cuda' or 'cpu')
        pixel_accuracy_threshold (float): Threshold for considering a pixel accurately reconstructed
        print_every (int): Print progress every N epochs
        
    Returns:
        float: Final L1 loss on validation set
        list: L1 losses for each epoch on training set
        list: L1 losses for each epoch on validation set
        list: Reconstruction accuracies for each epoch on training set
        list: Reconstruction accuracies for each epoch on validation set
    """
    # Use GPU if available and not explicitly specified
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    # Initialize loss and accuracy history trackers
    training_loss_history = []
    validation_loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    # Set optimizer and loss criterion
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.L1Loss()

    for epoch in range(1, epochs + 1):
        # Set model to training mode
        autoencoder.train()
        epoch_start_time = time.time()
        epoch_training_loss = 0.0
        epoch_training_accuracy = 0.0
        num_training_pixels = 0

        # Training loop
        for input_images, _ in train_dataloader:
            """
            Process a single batch during training
            
            Args:
                input_images (torch.Tensor): Batch of input images [batch_size, channels, height, width]
                _ (torch.Tensor): Ignored labels
            """
            # Move data to device
            input_images = input_images.to(device, non_blocking=True)
            
            # Forward pass
            reconstructed_images = autoencoder(input_images)
            reconstruction_loss = criterion(reconstructed_images, input_images)
            
            # Calculate pixel-wise accuracy (percentage of pixels within threshold)
            pixel_errors = torch.abs(reconstructed_images - input_images)
            accurate_pixels = (pixel_errors < pixel_accuracy_threshold).float().sum().item()
            total_pixels = input_images.numel()
            
            # Backward pass
            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()
            
            # Accumulate batch statistics
            epoch_training_loss += reconstruction_loss.item()
            epoch_training_accuracy += accurate_pixels
            num_training_pixels += total_pixels
            
        # Calculate average metrics for the epoch
        epoch_training_loss /= len(train_dataloader)
        epoch_training_accuracy = epoch_training_accuracy / num_training_pixels
        
        # Store history
        training_loss_history.append(epoch_training_loss)
        training_accuracy_history.append(epoch_training_accuracy)

        # ---------- validation ----------
        """
        Validation phase for current epoch
        
        Evaluates model performance on validation data without updating weights
        """
        # Set model to evaluation mode
        autoencoder.eval()
        with torch.no_grad():
            # Initialize validation metrics
            epoch_validation_loss = 0.0
            epoch_validation_accuracy = 0.0
            num_validation_pixels = 0
            
            # Validation loop
            for validation_images, _ in validation_dataloader:
                """
                Process a single validation batch
                
                Args:
                    validation_images (torch.Tensor): Batch of validation images [batch_size, channels, height, width]
                    _ (torch.Tensor): Ignored labels
                """
                validation_images = validation_images.to(device)
                reconstructed_validation_images = autoencoder(validation_images)
                
                # Calculate loss
                validation_batch_loss = criterion(reconstructed_validation_images, validation_images)
                epoch_validation_loss += validation_batch_loss.item()
                
                # Calculate pixel-wise accuracy
                validation_pixel_errors = torch.abs(reconstructed_validation_images - validation_images)
                validation_accurate_pixels = (validation_pixel_errors < pixel_accuracy_threshold).float().sum().item()
                validation_total_pixels = validation_images.numel()
                
                epoch_validation_accuracy += validation_accurate_pixels
                num_validation_pixels += validation_total_pixels
            
            # Calculate average validation metrics
            epoch_validation_loss /= len(validation_dataloader)
            epoch_validation_accuracy = epoch_validation_accuracy / num_validation_pixels
            
            # Store history
            validation_loss_history.append(epoch_validation_loss)
            validation_accuracy_history.append(epoch_validation_accuracy)

        # Print progress only every print_every epochs or on the last epoch
        epoch_duration = time.time() - epoch_start_time
        if epoch % print_every == 0 or epoch == epochs or epoch == 1:
            print(f"[{epoch:02d}/{epochs}]  "
                  f"train L1={epoch_training_loss:.4f}  "
                  f"train accuracy={epoch_training_accuracy:.4f}  "
                  f"validation L1={epoch_validation_loss:.4f}  "
                  f"validation accuracy={epoch_validation_accuracy:.4f}  "
                  f"({epoch_duration:.1f}s)")
              
    final_validation_loss = epoch_validation_loss
    final_validation_accuracy = epoch_validation_accuracy
    
    return (final_validation_loss, training_loss_history, validation_loss_history, 
           training_accuracy_history, validation_accuracy_history)


def train_MLP_model(classifier, train_dataloader, test_dataloader, loss_function, optimizer,
                num_epochs):
    """
    Train an MLP classifier model
    
    Args:
        classifier (nn.Module): The MLP classifier model to train
        train_dataloader (DataLoader): DataLoader containing training data
        test_dataloader (DataLoader): DataLoader containing test data
        loss_function (nn.Module): Loss function to optimize
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights
        num_epochs (int): Number of training epochs
        print_every (int): Print progress every N epochs
        
    Returns:
        nn.Module: Trained classifier model
        list: Training losses for each epoch
        list: Training accuracies for each epoch
        list: Test losses for each epoch
        list: Test accuracies for each epoch
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = classifier.to(device)
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        """
        Training loop for one epoch
        
        Processes all batches in the training dataloader and updates model weights
        """
        classifier.train()

        epoch_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for input_images, target_labels in train_dataloader:
            """
            Process a single training batch
            
            Args:
                input_images (torch.Tensor): Batch of input images [batch_size, channels, height, width]
                target_labels (torch.Tensor): Batch of ground truth labels [batch_size]
            """
            input_images = input_images.to(device)
            target_labels = target_labels.to(device)

            optimizer.zero_grad()

            predicted_logits = classifier(input_images)
            batch_loss = loss_function(predicted_logits, target_labels)

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * input_images.size(0)
            total_samples += input_images.size(0)
            correct_predictions += (torch.softmax(predicted_logits, dim=1).argmax(dim=1) == target_labels).sum().item()

        # Calculate epoch metrics
        average_train_loss = epoch_loss / len(train_dataloader.dataset)
        train_accuracy = (correct_predictions / total_samples) * 100

        train_losses.append(average_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test data
        test_MLP_model(classifier, test_dataloader, loss_function, test_losses, test_accuracies, epoch, num_epochs)

    return classifier, train_losses, train_accuracies, test_losses, test_accuracies

def test_MLP_model(classifier, test_dataloader, loss_function, test_losses, test_accuracies, epoch, num_epochs):
    """
    Evaluate an MLP classifier model on test data
    
    Args:
        classifier (nn.Module): The MLP classifier model to test
        test_dataloader (DataLoader): DataLoader containing test data
        loss_function (nn.Module): Loss function for evaluation
        test_losses (list): List to append test loss to
        test_accuracies (list): List to append test accuracy to
        epoch (int): Current epoch number
        num_epochs (int): Total number of epochs
        print_every (int): Print progress every N epochs
        
    Returns:
        list: Updated list of test losses
        list: Updated list of test accuracies
    """
    classifier.eval()

    test_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        """
        Evaluation loop without gradient computation
        
        Processes all batches in the test dataloader without updating model weights
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for input_images, target_labels in test_dataloader:
            """
            Process a single test batch
            
            Args:
                input_images (torch.Tensor): Batch of input images [batch_size, channels, height, width]
                target_labels (torch.Tensor): Batch of ground truth labels [batch_size]
            """
            input_images = input_images.to(device)
            target_labels = target_labels.to(device)

            predicted_logits = classifier(input_images)
            batch_loss = loss_function(predicted_logits, target_labels)
            predicted_classes = torch.softmax(predicted_logits, dim=1).argmax(dim=1)

            test_loss += batch_loss.item() * input_images.size(0)
            total_samples += input_images.size(0)
            correct_predictions += (predicted_classes == target_labels).sum().item()

        # Calculate test metrics
        average_test_loss = test_loss / len(test_dataloader.dataset)
        test_accuracy = (correct_predictions / total_samples) * 100

        test_losses.append(average_test_loss)
        test_accuracies.append(test_accuracy)

    return test_losses, test_accuracies

def train_decoder_only(decoder, encoder, train_dataloader, validation_dataloader,
                      epochs=10, learning_rate=2e-3, weight_decay=0,
                      device=None, print_every=5):
    """
    Train a decoder to reconstruct images from a fixed encoder.
    
    Args:
        decoder (nn.Module): The decoder model to train
        encoder (nn.Module): The fixed encoder model (weights will not be updated)
        train_dataloader (DataLoader): DataLoader containing training data
        validation_dataloader (DataLoader): DataLoader containing validation data
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        weight_decay (float): L2 regularization strength
        device (str): Device to use for training ('cuda' or 'cpu')
        print_every (int): Print progress every N epochs
        
    Returns:
        nn.Module: Trained decoder model
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    decoder.to(device).train()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.L1Loss()

    for epoch in range(1, epochs+1):
        """
        Training and validation loop for one epoch
        
        Processes all batches in the training and validation dataloaders
        """
        # --- TRAIN ---
        """
        Training phase for current epoch
        
        Updates decoder weights based on reconstruction loss
        """
        training_loss = 0.0
        for input_images, _ in train_dataloader:
            """
            Process a single training batch
            
            Args:
                input_images (torch.Tensor): Batch of input images [batch_size, channels, height, width]
                _ (torch.Tensor): Ignored labels
            """
            input_images = input_images.to(device)
            with torch.no_grad(): latent_vectors = encoder(input_images)
            reconstructed_images = decoder(latent_vectors)
            reconstruction_loss = criterion(reconstructed_images, input_images)
            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()
            training_loss += reconstruction_loss.item()
        training_loss /= len(train_dataloader)

        # --- VALIDATE ---
        """
        Validation phase for current epoch
        
        Evaluates decoder performance on validation data without updating weights
        """
        validation_loss = 0.0
        decoder.eval()
        with torch.no_grad():
            for validation_images, _ in validation_dataloader:
                """
                Process a single validation batch
                
                Args:
                    validation_images (torch.Tensor): Batch of validation images [batch_size, channels, height, width]
                    _ (torch.Tensor): Ignored labels
                """
                validation_images = validation_images.to(device)
                latent_vectors = encoder(validation_images)
                reconstructed_validation_images = decoder(latent_vectors)
                validation_loss += criterion(reconstructed_validation_images, validation_images).item()
        validation_loss /= len(validation_dataloader)
        decoder.train()

        if epoch==1 or epoch%print_every==0 or epoch==epochs:
            print(f"[{epoch:02d}/{epochs}] train L1={training_loss:.4f}  validation L1={validation_loss:.4f}")
    return decoder

import torch
from torch.nn import L1Loss

def compute_batch_l1_loss(encoder, decoder, data_loader, device=None, max_samples=64):
    """
    Compute the L1 reconstruction loss for a single batch of data
    
    Args:
        encoder (nn.Module): The encoder model
        decoder (nn.Module): The decoder model
        data_loader (DataLoader): DataLoader containing the data
        device (str): Device to use for computation ('cuda' or 'cpu')
        max_samples (int): Maximum number of samples to evaluate
        
    Returns:
        float: Mean L1 reconstruction error for the batch
    """
    device = device or next(decoder.parameters()).device
    criterion = L1Loss()

    # Get a single batch
    input_batch, _ = next(iter(data_loader))
    input_batch = input_batch[:max_samples].to(device)  # for speed

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        """
        Evaluation loop without gradient computation
        
        Processes a single batch to compute reconstruction error
        """
        latent_vectors = encoder(input_batch)
        reconstructed_images = decoder(latent_vectors)
        reconstruction_error = criterion(reconstructed_images, input_batch).item()

    return reconstruction_error