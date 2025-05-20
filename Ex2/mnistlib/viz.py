# mnistlib/viz.py
import matplotlib.pyplot as plt, torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def show_reconstructions(model, loader, n=8, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    x,_ = next(iter(loader))
    with torch.no_grad():
        xr = model(x.to(device)).cpu()
    f,axs = plt.subplots(2,n, figsize=(n*1.2,3))
    for i in range(n):
        for row,img in enumerate([x[i,0], xr[i,0]]):
            axs[row,i].imshow(img, cmap="gray"); axs[row,i].axis("off")
    axs[0,0].set_title("input"); axs[1,0].set_title("reconstruction")
    plt.tight_layout()
    plt.show()

# Sagie:
def plot_training_curves(train_losses,
                         test_losses,
                         train_accuracies=None,
                         test_accuracies=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if train_accuracies is not None and test_accuracies is not None:
      plt.subplot(1, 2, 2)
      plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy Rate')
      plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy Rate')
      plt.title('Accuracy Rate vs. Epochs')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy Rate (%)')
      plt.legend()
      plt.grid(True)

    plt.tight_layout()
    
    plt.show()

def plot_sklearn_confusion_matrix(model, dataloader, normalize, class_names):
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            preds = torch.softmax(logits, dim=1).argmax(dim=1)

            # Handle one-hot encoded labels if necessary
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels = labels.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Normalization method {'true', 'pred', 'all'} or None
    cm = confusion_matrix(all_labels, all_preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='.2f' if normalize else 'd')
    plt.title('Confusion Matrix')
    plt.tight_layout()

def compare_reconstructions(enc1, dec1, enc2, dec2, loader, n=6, device=None):
    import matplotlib.pyplot as plt
    device = device or next(dec1.parameters()).device
    enc1.eval(); enc2.eval(); dec1.eval(); dec2.eval()

    x_batch, _ = next(iter(loader))
    x_batch = x_batch.to(device)

    with torch.no_grad():
        xr1 = dec1(enc1(x_batch))
        xr2 = dec2(enc2(x_batch))

    fig, axes = plt.subplots(3, n, figsize=(n*2, 6))
    for i in range(n):
        axes[0, i].imshow(x_batch[i,0].cpu(), cmap="gray"); axes[0,i].axis("off")
        axes[1, i].imshow(xr1[i,0].cpu(), cmap="gray");  axes[1,i].axis("off")
        axes[2, i].imshow(xr2[i,0].cpu(), cmap="gray");  axes[2,i].axis("off")
    axes[0,0].set_title("Input")
    axes[1,0].set_title("AE encoder + decoder")
    axes[2,0].set_title("CLF encoder + decoder")
    plt.tight_layout()
    plt.show()
