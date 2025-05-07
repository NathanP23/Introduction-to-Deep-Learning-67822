# ================================
#          evaluation.py
# ================================
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio


def plot_training_results(selected_allele, train_losses, test_losses, accuracies, pos_accuracies, neg_accuracies, epochs):
    """
    Create plots for training/test loss and test accuracy
    """
    pyo.init_notebook_mode(connected=True)
    
    # Create an x-axis based on epoch numbers
    epoch_range = list(range(1, epochs + 1))
    x_ticks = [1] + list(range(5, epochs + 1, 5))
    
    # Axis and layout styles
    axis_style = dict(
        showgrid=True,
        gridcolor='gray',
        gridwidth=1,
        zeroline=True,
        zerolinewidth=2,
        linecolor='white',
        linewidth=2,
        mirror=True
    )
    
    layout_style = dict(
        template="plotly_dark",
        plot_bgcolor='rgba(40, 40, 40, 1)',  
        paper_bgcolor='rgba(30, 30, 30, 1)', 
        width=800,
        height=420,
        margin=dict(l=40, r=30, t=40, b=40),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='white',
            borderwidth=1
        )
    )
    
    # Loss Plot
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epoch_range,
        y=train_losses,
        mode='lines+markers',
        name='Train Loss',
        line=dict(width=3),
        hovertext=[f"Epoch {e}: {v:.4f}" for e, v in zip(epoch_range, train_losses)],
        hoverinfo="text"
    ))
    fig_loss.add_trace(go.Scatter(
        x=epoch_range,
        y=test_losses,
        mode='lines+markers',
        name='Test Loss',
        line=dict(width=3),
        hovertext=[f"Epoch {e}: {v:.4f}" for e, v in zip(epoch_range, test_losses)],
        hoverinfo="text"
    ))
    fig_loss.update_layout(
        title=f"Train vs Test Loss for {selected_allele}",
        xaxis=dict(title="Epoch", tickmode='array', tickvals=x_ticks, **axis_style),
        yaxis=dict(title="Loss", **axis_style),
        **layout_style
    )
    pio.renderers.default = 'notebook'
    pio.show(fig_loss)

    
    # Accuracy Plot - updated to include per-class accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epoch_range,
        y=accuracies,
        mode='lines+markers',
        name='Overall Accuracy',  # Changed from 'Test Accuracy'
        line=dict(width=3),
        hovertext=[f"Epoch {e}: {v:.2f}%" for e, v in zip(epoch_range, accuracies)],
        hoverinfo="text"
    ))
    
    # Add positive class accuracy
    fig_acc.add_trace(go.Scatter(
        x=epoch_range,
        y=pos_accuracies,
        mode='lines+markers',
        name=f'{selected_allele} Accuracy',
        line=dict(width=3, dash='dash'),
        hovertext=[f"Epoch {e}: {v:.2f}%" for e, v in zip(epoch_range, pos_accuracies)],
        hoverinfo="text"
    ))
    
    # Add negative class accuracy
    fig_acc.add_trace(go.Scatter(
        x=epoch_range,
        y=neg_accuracies,
        mode='lines+markers',
        name='NEG Accuracy',
        line=dict(width=3, dash='dot'),
        hovertext=[f"Epoch {e}: {v:.2f}%" for e, v in zip(epoch_range, neg_accuracies)],
        hoverinfo="text"
    ))
    
    fig_acc.add_shape(type='line', x0=1, x1=epochs, y0=70, y1=70,
                      line=dict(color='orange', width=2, dash='dash'))
    
    fig_acc.update_layout(
        title=f"Test Accuracy Over Time for {selected_allele}",
        xaxis=dict(title="Epoch", tickmode='array', tickvals=x_ticks, **axis_style),
        yaxis=dict(title="Accuracy (%)", **axis_style),
        **layout_style
    )
    pio.renderers.default = 'notebook'
    pio.show(fig_acc)
    
    
import torch
from torch.utils.data import DataLoader, TensorDataset

def per_class_accuracy(model, X, y, threshold=0.5, batch_size=512, device=None):
    """
    Computes accuracy on positives (label=1) and negatives (label=0).

    Returns: dict(pos_acc=float, neg_acc=float, overall=float)
    """
    if device is None:
        device = next(model.parameters()).device
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size)

    model.eval()
    n_pos = n_neg = tp = tn = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            probs   = torch.sigmoid(logits)        # (B,) for your binary nets
            preds   = (probs >= threshold).long()

            n_pos  += (yb == 1).sum().item()
            n_neg  += (yb == 0).sum().item()
            tp     += ((preds == 1) & (yb == 1)).sum().item()
            tn     += ((preds == 0) & (yb == 0)).sum().item()

    pos_acc = tp / n_pos if n_pos else float('nan')
    neg_acc = tn / n_neg if n_neg else float('nan')
    overall = (tp + tn) / (n_pos + n_neg)
    return dict(pos_acc=pos_acc, neg_acc=neg_acc, overall=overall)


def evaluate_ensemble(ensemble, allele_datasets, threshold=0.5):
    """
    ensemble     : MultiAlleleEnsemble (six heads)
    allele_datasets : {allele: (X_test, y_test)}  *y_test is binary*
    Prints a table of per-allele accuracies.
    """
    order = ensemble.allele_order
    print(f"{'Allele':<6} | {'Pos%':>6} | {'Neg%':>6} | {'Overall%':>8}")
    print("-" * 34)
    for allele in order:
        X, y = allele_datasets[allele]['X_test'], allele_datasets[allele]['y_test']
        stats = per_class_accuracy(ensemble.models[allele], X, y, threshold)
        print(f"{allele:<6} | {stats['pos_acc']*100:6.2f} | "
              f"{stats['neg_acc']*100:6.2f} | {stats['overall']*100:8.2f}")
