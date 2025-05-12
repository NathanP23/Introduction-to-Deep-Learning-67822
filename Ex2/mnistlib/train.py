# mnistlib/train.py
import torch, time, contextlib
from torch import nn

def train_loop(model, loader_tr, loader_val, *,
               epochs=10, lr=2e-3, weight_decay=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.L1Loss()

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for xb, _ in loader_tr:
            xb   = xb.to(device, non_blocking=True)
            loss = crit(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        running /= len(loader_tr)

        # ---------- validation ----------
        model.eval()
        with torch.no_grad():
            val = torch.stack([
                crit(model(x.to(device)), x.to(device))
                for x, _ in loader_val
            ]).mean().item()

        print(f"[{ep:02d}/{epochs}]  train L1={running:.4f}  "
              f"val L1={val:.4f}  ({time.time()-t0:.1f}s)")
    return val
