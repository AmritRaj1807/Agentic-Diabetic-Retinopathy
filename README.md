# Diabetic Retinopathy Grading
### Parallel EfficientNetB4 + Swin Transformer Base with CORN Ordinal Loss

> MSc Artificial Intelligence — University of Surrey | Individual Contribution to Group Project

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/DRG-Group-34/Diabetic_Retinopathy_Grading)

---

## Overview

This project develops an automated deep learning pipeline to grade **Diabetic Retinopathy (DR)** severity from retinal fundus images. DR is classified into five ordinal severity levels (0–4), ranging from no disease to proliferative retinopathy.

Because the labels represent monotonically increasing disease severity, the problem is framed as an **ordinal classification** task rather than standard multiclass classification. The final model is a parallel dual-backbone fusion of EfficientNetB4 and Swin Transformer Base, trained with **CORN (Conditional Ordinal Regression Networks) loss**, achieving a **Test QWK of 0.8077**.

---

## Live Demo

The final model is deployed as an interactive web application on Hugging Face Spaces. Upload a retinal fundus image and the app returns the predicted DR severity grade (0–4).

**[Try the app → Diabetic Retinopathy Grading on Hugging Face](https://huggingface.co/spaces/DRG-Group-34/Diabetic_Retinopathy_Grading)**

---

## DR Severity Scale

| Grade | Clinical Description |
|:---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## My Contribution

This repository documents my individual contribution within a larger group project:

- Established single-model baselines for **Swin Transformer Base** and **EfficientNetB4**
- Designed and evaluated a **preprocessing and augmentation pipeline** (RFOV Cropping, CLAHE, MixUp, Random Erasing)
- Investigated and compared **ensemble, series, and parallel fusion architectures**
- Implemented **CORN loss** for ordinal-aware training
- Evaluated all models using **Quadratic Weighted Kappa (QWK)** and **Macro F1-score**

---

## Architecture

### Final Model: Parallel Dual-Backbone Fusion

Both backbones process the same retinal image independently. Their feature vectors are concatenated and passed to a shared fusion head.

The output is **4 logits** for 5 ordinal classes, as required by the CORN loss formulation.

### Backbone Rationale

| Backbone | Rationale |
|---|---|
| **Swin Transformer Base** | Shifted-window attention captures both local lesion detail (haemorrhages, exudates) and global retinal context |
| **EfficientNetB4** | Efficient convolutional scaling extracts fine-grained local patterns: microaneurysms, vessel structures, and texture changes |

> Swin Transformer Base outperformed EfficientNetB4 as a standalone model. EfficientNetB4 was retained as a complementary feature extractor in the fusion architecture.

---

## Dataset & Preprocessing

- Retinal fundus images linked to severity labels via CSV
- **Stratified train/validation/test split** saved to disk to ensure consistent class balance across all experiments

**Preprocessing variants evaluated:**

| Step | Description |
|---|---|
| Baseline | Original images, no preprocessing |
| RFOV Cropping | Removes irrelevant black background; aligns the retinal field of view |
| CLAHE Enhancement | Contrast Limited Adaptive Histogram Equalisation (clip limits 1.0 and 2.0 tested) |

RFOV Cropping delivered the single largest preprocessing gain across experiments.

---

## Augmentation & Loss Functions

**Augmentation pipeline:**
- Horizontal flip, Colour Jitter, Random Affine translation
- MixUp (alpha=0.2), Label Smoothing, Random Erasing

**Loss functions evaluated:**

| Loss | Notes |
|---|---|
| Weighted Cross Entropy | Baseline loss; class weights to handle imbalance |
| Weighted Focal Loss | Down-weights easy examples |
| **CORN Loss** ✓ | Ordinal-aware; best suited to graded severity labels |

CORN loss was selected for the final model as it explicitly models the ordinal structure of DR severity grades.

---

## Evaluation Metrics

| Metric | Rationale |
|---|---|
| **Quadratic Weighted Kappa (QWK)** | Penalises larger grading errors more heavily — appropriate for ordered severity labels |
| **Macro F1-score** | Gives equal weight to each class — handles class imbalance fairly |

---

## Results

### Single-Model & Preprocessing Experiments

| Configuration | Val F1 | Val QWK | Test F1 | Test QWK |
|---|:---:|:---:|:---:|:---:|
| Swin-B + Weighted CE | 0.5809 | 0.7541 | 0.5512 | 0.7580 |
| Swin-B + Weighted CE + MixUp | 0.5769 | 0.7450 | 0.5792 | 0.7639 |
| Swin-B + Label Smoothing | 0.5860 | 0.7465 | 0.5757 | 0.7669 |
| RFOV Crop + Swin-B + MixUp | 0.5988 | 0.8021 | 0.5933 | 0.7887 |
| CLAHE 2.0 + RFOV + Swin-B + MixUp | 0.6154 | 0.8004 | — | — |

### Fusion & Ensemble Experiments

| Configuration | Val F1 | Val QWK | Test F1 | Test QWK |
|---|:---:|:---:|:---:|:---:|
| EfficientNetB4 (standalone) | 0.4981 | 0.6667 | — | — |
| Ensemble (Eff=0.2, Swin=0.8) | — | — | 0.6042 | 0.7951 |
| Series fusion | 0.4795 | 0.6606 | 0.4649 | 0.6572 |
| Parallel + linear + CORN | 0.6182 | 0.8061 | 0.6063 | 0.8017 |
| **Final: Parallel + CORN + Random Erasing** | **0.6250** | **0.8096** | **0.6188** | **0.8077** |


### Confusion Matrix — Final Model (Test Set)

![Confusion matrix of the final parallel EfficientNetB4 + Swin-B + CORN model on the test set](confusion_matrix.png)

Most predictions cluster on or near the diagonal, consistent with the ordinal nature of DR grading — errors tend to occur between adjacent severity grades rather than distant ones.

---

## Training Setup

| Technique | Role |
|---|---|
| Transfer learning (ImageNet) | Initialises both backbones with pretrained weights |
| AdamW + weight decay | Stable fine-tuning with decoupled regularisation |
| Cosine annealing scheduler | Gradual learning rate reduction for better convergence |
| Mixed precision (AMP) | Reduces GPU memory usage and improves throughput |
| Gradient checkpointing | Trades additional compute for reduced memory footprint |
| Gradient clipping | Stabilises optimisation during fine-tuning |
| Stratified saved split | Ensures fair, consistent comparison across all runs |
| Top-k checkpoint saving | Retains best model weights by validation QWK |
| Fixed random seeds | Full reproducibility across Python, NumPy, PyTorch, and CUDA |

---

## Key Findings

- **Swin Transformer Base** was the stronger standalone backbone, outperforming EfficientNetB4 by a clear margin
- **RFOV Cropping** provided the largest single preprocessing improvement
- **Series fusion** was ineffective; **parallel fusion** consistently outperformed all other architectures
- **CORN loss** improved over standard cross-entropy by respecting the ordinal structure of DR grades
- Checkpoint selection by **validation QWK** (rather than loss) was essential, as later epochs exhibited overfitting

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Preprocess retinal images (RFOV Cropping + CLAHE)
python Preprocessing.py

# 2. Train the model (update paths in Train.py Config before running)
python Train.py

# 3. Evaluate on the test set (update paths in Test.py Config before running)
python Test.py
```

> Update the `data_dir`, `csv_path`, and `model_path` fields in the `Config` dataclass at the top of each script to point to your local dataset and checkpoint.

---

## Tech Stack

- **Deep Learning:** PyTorch, timm (Swin Transformer), Torchvision
- **Ordinal Loss:** CORN (Conditional Ordinal Regression Networks)
- **Data & Analysis:** NumPy, Pandas, Scikit-learn, Matplotlib
- **Training Infrastructure:** CUDA GPU, Mixed Precision (AMP)
- **Language:** Python

---

## About

This project was completed as part of an MSc Artificial Intelligence group coursework submission at the **University of Surrey**. This repository covers my individual contribution to the overall group effort.

**Author:** Bilal Ahmad Sami  
**Programme:** MSc Artificial Intelligence, University of Surrey
