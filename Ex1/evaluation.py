# evaluation.py
import plotly.graph_objects as go
import plotly.offline as pyo


def plot_training_results(selected_allele, train_losses, test_losses, accuracies, epochs):
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
    pyo.iplot(fig_loss)
    
    # Accuracy Plot
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epoch_range,
        y=accuracies,
        mode='lines+markers',
        name='Test Accuracy',
        line=dict(width=3),
        hovertext=[f"Epoch {e}: {v:.2f}%" for e, v in zip(epoch_range, accuracies)],
        hoverinfo="text"
    ))
    fig_acc.add_shape(type='line', x0=1, x1=epochs, y0=70, y1=70,
                      line=dict(color='orange', width=2, dash='dash'))
    
    fig_acc.update_layout(
        title="Test Accuracy Over Time for {selected_allele}",
        xaxis=dict(title="Epoch", tickmode='array', tickvals=x_ticks, **axis_style),
        yaxis=dict(title="Accuracy (%)", **axis_style),
        **layout_style
    )
    pyo.iplot(fig_acc)
