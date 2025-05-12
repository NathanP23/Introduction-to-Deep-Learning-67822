# README: Exercise 2

## 📦 Installation & Environment Setup

1. **Create a virtual environment (optional but recommended):**

```bash
python3.11 -m venv .venv_Ex2    # torch won't work with higher python versions!
source .venv_Ex2/bin/activate   # On Windows: .venv_Ex2\Scripts\activate
```

2. **Install dependencies from `requirements.txt`:**

```bash
pip install -r requirements.txt
```

3. **Run the notebook:**

```bash
jupyter notebook Ex2Sol.ipynb
```
## 📦 Current Pipeline

```
        MNIST dataset (60k train / 10k test)
                  │
                  │  mnistlib/data.py: get_mnist_loaders()
                  │
                  ▼
           DataLoader objects
(train_loader, test_loader), batch=256
                  │
                  │───▶────┐
                  │        │
                  ▼        │
          Autoencoder      │
     mnistlib/ae.py: ConvAE│
   ┌───────────────────────│────────────┐
   │ Encoder               │ Decoder    │
   │ (Conv layers)         │ (ConvTrans │
   │ input → latent        │  layers)   │
   │ vector                │ latent →   │
   │                       │ output img │
   └───────────────▲───────┴────────────┘
                   │
                   ▼
         mnistlib/train.py: train_loop()
 (mean L1 Loss, Adam optimizer, epochs=10)
                   │
                   ▼
         Trained Autoencoder Model
                   │
                   ▼
  mnistlib/viz.py: show_reconstructions()
          (visualize images)
```

## 📦 Current network Architecture:
```
          Input Image (1×28×28)
                    │
                    ▼
┌───────────────────────────────────────┐
│ Encoder                               │
│ ┌───────────┐  ┌───────────┐  ┌─────┐ │
│ │Conv Layer │─▶│Conv Layer │─▶│Conv │ │
│ │1→base_c   │  │base_c→2c  │  │2c→4c│ │
│ │kernel=3×3 │  │kernel=3×3 │  │3×3  │ │
│ │stride=2   │  │stride=2   │  │s=2  │ │
│ │ReLU+BN    │  │ReLU+BN    │  │ReLU │ │
│ └───────────┘  └───────────┘  └─────┘ │
│       │             │           │     │
│   14×14×c       7×7×2c       4×4×4c   │
│                                       │
│ flatten→Linear→ latent vector (size=d)│
└───────────────────┬───────────────────┘
                    │
                    │ (latent vector: d-dim)
                    ▼
┌───────────────────────────────────────┐
│ Decoder                               │
│ latent vector (d-dim) → Linear        │
│ │                                     │
│ ▼                                     │
│ reshape: (4c×4×4)                     │
│ ┌──────────────┐ ┌───────────┐ ┌─────┐│
│ │ConvTrans     │─▶ConvTrans  │─▶ConvT││
│ │4c→2c         │ │2c→c       │ │c→1  ││
│ │kernel=3×3    │ │kernel=3×3 │ │3×3  ││
│ │stride=2      │ │stride=2   │ │s=2  ││
│ │ReLU+BN       │ │ReLU+BN    │ │Sigm ││
│ └──────────────┘ └───────────┘ └─────┘│
│     7×7×2c          14×14×c     28×28 │
└───────────────────┬───────────────────┘
                    │
                    ▼
          Reconstructed Image (1×28×28)
```
