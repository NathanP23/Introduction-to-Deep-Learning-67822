# mnistlib/viz.py
import matplotlib.pyplot as plt, torch

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
