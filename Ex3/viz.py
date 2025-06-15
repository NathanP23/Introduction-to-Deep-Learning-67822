
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
            epoch_labels.append(f"Ep{epoch}")
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
    plt.savefig(f"plots/{model_name}_loss_plot.png")
    plt.show()


def plot_losses_and_accuracy(log_data, model_name):
    epochs = [entry["epoch"] for entry in log_data]
    train_losses = [entry["train_loss"] for entry in log_data]
    test_losses = [entry["test_loss"] for entry in log_data]
    train_accs = [entry["train_acc"] for entry in log_data]
    test_accs = [entry["test_acc"] for entry in log_data]
    
    x_values = list(range(len(log_data)))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Plot losses
    ax1.plot(x_values, train_losses, label="Train Loss", marker='o', markersize=3)
    ax1.plot(x_values, test_losses, label="Test Loss", marker='x', markersize=3)
    ax1.set_title(f"{model_name} - Loss Progression")
    ax1.set_xlabel("Training Progress")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.7)
    
    # Plot accuracies
    ax2.plot(x_values, train_accs, label="Train Accuracy", marker='o', markersize=3, color='green')
    ax2.plot(x_values, test_accs, label="Test Accuracy", marker='x', markersize=3, color='red')
    ax2.set_title(f"{model_name} - Accuracy Progression")
    ax2.set_xlabel("Training Progress")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.7)
    
    # Add epoch boundaries to both plots
    epoch_boundaries = []
    current_epoch = epochs[0]
    for i, epoch in enumerate(epochs):
        if epoch != current_epoch:
            epoch_boundaries.append(i)
            current_epoch = epoch
    
    if epoch_boundaries:
        for ax in [ax1, ax2]:
            ax.set_xticks(epoch_boundaries)
            ax.set_xticklabels([f"Ep{epochs[i]}" for i in epoch_boundaries])
    
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_loss_acc_plot.png")
    plt.show()