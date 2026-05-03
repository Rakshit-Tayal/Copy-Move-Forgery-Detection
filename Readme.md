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

Copy-move forgery detection in scientific literature is notoriously difficult. Traditional methods (like SIFT) suffer from "keypoint starvation" because biological backgrounds lack sharp geometric edges. Meanwhile, state-of-the-art Deep Learning models (like BusterNet or EfficientNet-backed architectures) are highly optimized for natural images. When applied to microscopy, they aggressively filter out the low-level noise required for forensic analysis, causing them to hallucinate matches and fail catastrophically (achieving Hard Dice scores < 0.10).

**Key Dataset Difficulties Solved:**
1. **Extreme Class Imbalance:** In forged images, the manipulated pixels constitute **< 1%** of the total image area.
2. **High False-Positive Rate:** Distinguishing between maliciously duplicated biological structures and naturally occurring identical cells/bands in highly uniform backgrounds.

---

## 🧠 Architecture Details: Correlation-Aware Attention U-Net

To overcome the limitations of natural-image models, this repository implements a custom **Correlation-Aware Attention U-Net** that retains high-frequency forensic artifacts while actively filtering uniform background noise.

### 1. High-Frequency Feature Extractor (Encoder)
* **Backbone:** Standard Hierarchical Convolutional Encoder.
* **Purpose:** Unlike highly compressed NAS models (e.g., EfficientNet) that destroy low-level textures to prioritize semantic meaning, this standard encoder preserves the subtle noise-domain inconsistencies and splicing artifacts crucial for microscopic forensic analysis.

### 2. Dense Self-Correlation Module
* **Location:** Network Bottleneck.
* **Purpose:** Computes the spatial similarity (dot-product) between all pairs of feature vectors in the dense feature map. This explicitly maps long-range dependencies, identifying perfectly matching sub-1% regions (the source and the clone) regardless of where they are pasted.

### 3. Attention-Guided Decoder
* **Mechanism:** Attention Gates (AGs) integrated at every skip-connection.
* **Purpose:** Actively suppresses feature activations in irrelevant, uniform background regions. This forces the network to focus strictly on salient structural anomalies, preventing the "background hallucination" problem seen in standard CMFD networks.

### 4. Loss Function
* **Joint Optimization:** `BCEWithLogitsLoss` + `Soft Dice Loss`
* **Purpose:** BCE handles pixel-wise classification, while the Dice Loss explicitly tackles the extreme class imbalance by penalizing the network heavily for missing the sparse (<1%) forged regions.

---

## 📊 Dataset

The model was trained and evaluated on a specialized dataset of 5,000 microscopic images:
* **Authentic Images (2,300):** Paired with programmatically generated blank masks to force the network to learn True Negatives and reduce False Positives.
* **Forged Images (2,700):** Containing visually identical copy-move splices, paired with pixel-perfect ground-truth localization masks.

---

## 🏆 Results & Performance

This project involved a systematic architectural search to solve the specific challenges of microscopic forgery detection. Standard segmentation models and natural-image SOTA architectures were evaluated and iteratively improved upon to handle extreme class imbalance and domain shift.

| Model / Architecture | Key Observation / Limitation | Hard Dice Score | Image-Level Recall |
| :--- | :--- | :---: | :---: |
| **BusterNet (SOTA Baseline)** | Fails on microscopy; hallucinates matches in uniform backgrounds. | `< 0.10` | - |
| **Standard U-Net** | Overwhelmed by the extreme <1% foreground class imbalance. | `Poor` | `Poor` |
| **U-Net + Classification Head** | Failed to learn global image authenticity (performed at ~50% random chance). | `N/A` | `~50.0%` |
| **Attention U-Net** | High recall, but yielded severe false positives on uniform biological backgrounds. | `0.29` | `~96.8%` |
| **EfficientNet + Attention U-Net** | Domain shift: aggressive compression destroyed crucial high-frequency forensic noise. | `Failed` | `Failed` |
| **Correlation Attention U-Net (Ours)**| **Dense feature matching + attention filtering successfully localized splices.** | **`0.41`** | **`~96.8%`** |

*Note: Achieving a >0.4 Hard Dice on sub-1% microscopy anomalies represents a massive leap in domain-specific forgery localization, proving that dense self-correlation combined with high-frequency convolutional encoding is necessary for this domain.*


---

├── Paper reproduction/         # Baselines and reproduced model weights
│   ├── best_cmseg_model.pth    # Saved CMSegNet model weights 
│   └── busternet.pth           # BusterNet baseline model weights
├── unet.ipynb                  # Main research notebook (Full Experimental Pipeline)
├── final_test.py               # Baseline CNN test script
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
