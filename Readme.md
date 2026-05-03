# Scientific Image Forgery Detection: Copy-Move Anomaly Localization 🔬

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced Deep Learning pipeline designed specifically to detect and localize **Copy-Move Forgeries (CMFD)** within scientific microscopic images (e.g., western blots, gels, cell cultures). 

## 📑 Table of Contents
- [The Challenge](#-the-challenge)
- [Architecture Details](#-architecture-details)
- [Dataset](#-dataset)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Future Work](#-future-work)

---

## 🛑 The Challenge: Microscopy vs. Natural Images

State-of-the-art CMFD models (like BusterNet) are highly optimized for natural images. When applied to scientific microscopy, these dual-branch architectures often fail catastrophically (achieving Hard Dice scores < 0.10). This happens because microscopy images lack distinct semantic boundaries, causing standard correlation filters to hallucinate matches in the highly repetitive, uniform background noise.

**Key Dataset Difficulties Solved:**
1. **Extreme Class Imbalance:** In forged images, the manipulated pixels constitute **< 1%** of the total image area.
2. **High False-Positive Rate:** Distinguishing between maliciously duplicated biological structures and naturally occurring identical cells/bands.

---

## 🧠 Architecture Details: Efficient-Attention U-Net

To overcome the limitations of standard segmentation models, this repository implements a custom **EfficientNet-B0 encoded Attention U-Net** that decouples global forgery detection from pixel-level localization.

### 1. Feature Extractor (Encoder)
* **Backbone:** `EfficientNet-B0` (Pre-trained on ImageNet).
* **Purpose:** Acts as a lightweight, highly robust feature extractor capable of capturing subtle textural and noise-domain inconsistencies left behind by splicing.

### 2. Attention-Guided Decoder
* **Attention Gates:** Integrated at every skip-connection before the decoding blocks.
* **Purpose:** Actively suppresses feature activations in irrelevant, uniform background regions. This forces the network to focus strictly on salient structural anomalies, preventing the "background hallucination" problem seen in other CMFD networks.

### 3. Loss Function
* **Joint Loss Optimization:** `BCEWithLogitsLoss` + `Soft Dice Loss`
* **Purpose:** BCE handles the pixel-wise classification, while the Dice Loss explicitly tackles the extreme class imbalance by penalizing the network heavily for missing the <1% forged regions.

---

## 📊 Dataset

The model was trained and evaluated on a specialized dataset of 5,000 microscopic images:
* **Authentic Images (2,300):** Paired with programmatically generated blank masks to force the network to learn True Negatives and reduce False Positives.
* **Forged Images (2,700):** Containing visually identical copy-move splices, paired with pixel-perfect ground-truth localization masks.

---

## 🏆 Results & Performance

By leveraging attention mechanisms to filter uniform backgrounds, this custom architecture significantly outperformed standard natural-image CMFD models.

| Model | Domain | Hard Dice Score | Image-Level Recall |
| :--- | :--- | :---: | :---: |
| BusterNet (Baseline) | Natural Images | `< 0.10` | - |
| **Efficient-Attention U-Net (Ours)** | **Microscopy** | **`0.41`** | **`~96.8%`** |

*Note: Achieving a >0.4 Hard Dice on sub-1% microscopy anomalies represents a massive leap in domain-specific forgery localization.*

---

## 📂 Project Structure

├── data/                       # Dataset directory (Excluded from Git)
│   ├── train_images/           # Authentic and Forged RGB images
│   └── train_masks/            # Ground truth NumPy/PNG masks
├── Paper reproduction/         # Baselines and reproduced model weights
│   ├── best_cmseg_model.pth    # Saved cmsegnet model weights 
│   └── busternet.pth           # BusterNet baseline model weights
├── notebooks/                  # Jupyter notebooks for EDA and testing
│   ├── unet.ipynb              # Baseline U-Net implementation
│   └── attention_unet.ipynb    # Final Efficient-Attention U-Net code
├── src/                        # Core source code
│   ├── dataset.py              # Custom PyTorch Dataset and Dataloaders
│   ├── model.py                # EfficientAttentionUnet architecture
│   ├── loss.py                 # Joint BCE & Dice Loss functions
│   └── metrics.py              # IOU, Soft Dice, and Hard Dice evaluation
├── .gitignore                  # Git ignore rules (protects against large files)
└── README.md                   # Project documentation
