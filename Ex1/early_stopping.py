# ================================
#          early_stopping.py
# ================================
class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for patience epochs.
    """
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore model to best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        """
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): The model being trained
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model is not None:
                    model.load_state_dict(self.best_model)
                return True
        return False