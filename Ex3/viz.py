import matplotlib.pyplot as plt


def plot_losses_and_accuracy(log_data, model_name):
    epochs = [entry["epoch"] for entry in log_data]
    train_losses = [entry["train_loss"] for entry in log_data]
    test_losses = [entry["test_loss"] for entry in log_data]
    train_accs = [entry["train_acc"] for entry in log_data]
    test_accs = [entry["test_acc"] for entry in log_data]

    x_values = list(range(len(log_data)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    ax1.plot(x_values, train_losses, label="Train Loss", marker='o', markersize=3)
    ax1.plot(x_values, test_losses, label="Test Loss", marker='x', markersize=3)
    ax1.set_title(f"{model_name} - Loss Progression")
    ax1.set_xlabel("Training Progress")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.7)

    ax2.plot(x_values, train_accs, label="Train Accuracy", marker='o', markersize=3, color='green')
    ax2.plot(x_values, test_accs, label="Test Accuracy", marker='x', markersize=3, color='red')
    ax2.set_title(f"{model_name} - Accuracy Progression")
    ax2.set_xlabel("Training Progress")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.7)

    # Custom epoch boundary labels
    epoch_boundaries = [0]
    epoch_labels = ["Epoch 1"]
    current_epoch = epochs[0]

    for i, epoch in enumerate(epochs):
        if epoch != current_epoch:
            epoch_boundaries.append(i)
            epoch_labels.append(f"Epoch {epoch}")
            current_epoch = epoch

    if epoch_boundaries:
        ax1.set_xticks(epoch_boundaries)
        ax1.set_xticklabels(epoch_labels)
        ax2.set_xticks(epoch_boundaries)
        ax2.set_xticklabels(epoch_labels)

    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_loss_acc_plot.png")
    plt.show()
