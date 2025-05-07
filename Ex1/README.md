# README: Exercise 1 - Peptide Classification to HLA Alleles

## 📦 Installation & Environment Setup

1. **Create a virtual environment (optional but recommended):**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies from `requirements.txt`:**

```bash
pip install -r requirements.txt
```

3. **Run the notebook:**

```bash
jupyter notebook Ex1Sol.ipynb
```


## 🧠 Project Overview

This Jupyter notebook and codebase implement a full pipeline for classifying 9-mer peptides against six specific HLA alleles using PyTorch. The objective was to explore the representational and learning capacity of various neural architectures, and eventually apply the best-performing model to predict peptide binding from the SARS-CoV-2 spike protein.

The assignment required designing and training:

* **Basic MLP (Model 2B)** with minimal architecture.
* **Enhanced MLP (Model 2C)** with dropout, batch norm, LeakyReLU.
* **Linear-only MLP (Model 2D)** for ablation.

We also used the best model to:

* Predict binding peptides from a given COVID-19 protein.

---

## ✅ What Was Successfully Implemented

### ✔ Data Handling

* Wrote robust preprocessing pipeline (`dataset.py`) that:

  * Loads 6 allele-specific files and one shared negative file.
  * Splits each dataset into train/test using a consistent 90/10 split.
  * Converts peptides to tensors using amino acid-to-index mapping.
  * Created balanced binary datasets per allele.

### ✔ Models

* Implemented three network variants:

  * `2B`: Basic two-layer MLP.
  * `2C`: Dropout, batch norm, LeakyReLU (the main model used).
  * `2D`: Ablation study without non-linearities.

### ✔ Training

* Unified training function supporting early stopping, class balancing, and per-class accuracy.
* Early stopping and learning rate scheduling were integrated.

### ✔ Evaluation

* Created clear per-epoch performance logs.
* Plotted train/test loss and class-wise accuracies.
* Produced readable, stylized loss/accuracy plots using Plotly.

### ✔ Inference

* Applied the trained `Model2C` ensemble to scan 9-mer peptides from SARS-CoV-2 spike protein.
* Reported top candidates and peptides that bound to multiple alleles.

---

## ⚠️ What Was Missing / Incomplete

* ❌ **Explanation Section (`TODO`)**: Section asking for reasoning behind the model design was left as "TODO".
* ❌ **No performance summary table** comparing 2B, 2C, and 2D.
* ❌ **Hyperparameter search** was skipped or hardcoded manually.
* ❌ **Multiclass baseline** (one model for all classes) was **not attempted**.
* ❌ **Notebook Markdown polish** (some answers and plots should’ve been wrapped better, or more interpretation written).

---

## ⚡ What Went Well

* Used clean modular structure (e.g., `model.py`, `training.py`, `predict.py`).
* Built reusable infrastructure for datasets and loaders.
* Wrote intelligible, annotated visualizations.
* Good experimentation: tried both simple and more advanced networks.
* Predicted SARS-CoV-2 results with a clear filtering strategy.

---

## 🏃 Why I Rushed

* I underestimated how long the spike prediction and training would take — especially for **six alleles**.
* The final plotting + prediction section took more effort than expected.
* I initially over-engineered the training loop and had to refactor during submission week.
* I spent time debugging `plotly` and `.ipynb` rendering issues to ensure GitHub would render plots correctly.
* As a result, I didn’t fully polish the written answers and explanations in the last section.

---

## 🚧 What I'd Improve With More Time

* Write `explain all design choices` markdown section in more detail.
* Add a `table of performance` for Models 2B/2C/2D.
* Try an **end-to-end multiclass model** (single model predicting 7 classes).
* Visualize **class imbalance impact** directly.
* Add comments on why `pos_weight` is essential in BCEWithLogitsLoss.
* Use a config YAML instead of hardcoded per-allele hyperparameter overrides.

---

## 📁 Structure

```
.
├── dataset.py               # Load and convert data
├── model.py                 # Model classes (2B/2C/2D)
├── training.py              # Train loop w/ early stopping
├── evaluation.py            # Plotting and metrics
├── predict.py               # SARS-CoV-2 peptide scanning
├── config.py                # Constants
├── hyperparameters.py       # Per-allele HP config
├── early_stopping.py        # Early stopping logic
├── spike.txt                # SARS-CoV-2 protein sequence
├── *.ipynb                  # Final notebook
└── Data/HLA_Dataset/        # Input peptide files
```

---

## 🧬 Summary

This was a well-structured, multi-model peptide classification project built with modular code. While the technical implementation was mostly completed successfully, some theoretical and interpretive writeups were left incomplete due to time constraints.

The best-performing model (2C) was used to extract real biological insights by screening the SARS-CoV-2 spike protein, completing the assignment with a practical and interpretable application.
