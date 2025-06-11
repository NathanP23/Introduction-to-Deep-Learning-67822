
import matplotlib.pyplot as plt
def plot_losses(log_data, model_name):
    epochs = [entry["epoch"] for entry in log_data]
    steps = [entry["step"] for entry in log_data]
    train_losses = [entry["train_loss"] for entry in log_data]
    test_losses = [entry["test_loss"] for entry in log_data]
    
    # Create continuous x-axis showing interval progression
    x_values = list(range(len(log_data)))  # 0, 1, 2, 3, 4...
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, train_losses, label="Train Loss", marker='o', markersize=3)
    plt.plot(x_values, test_losses, label="Test Loss", marker='x', markersize=3)
    
    # Create custom x-tick labels showing epoch numbers
    # Show epoch boundaries more clearly
    epoch_boundaries = []
    epoch_labels = []
    current_epoch = epochs[0]
    
    for i, epoch in enumerate(epochs):
        if epoch != current_epoch:
            epoch_boundaries.append(i)
            epoch_labels.append(f"Epoch {epoch}")
            current_epoch = epoch
    
    # Set major ticks at epoch boundaries
    if epoch_boundaries:
        plt.xticks(epoch_boundaries, epoch_labels)
    
    # Add minor ticks to show intervals within epochs
    plt.gca().set_xticks(x_values, minor=True)
    plt.grid(True, which='major', alpha=0.7)
    
    plt.title(f"{model_name} - Loss Progression (Intervals within Epochs)")
    plt.xlabel("Training Progress (Epoch Boundaries Marked)")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_loss_plot.png")
    plt.show()