# Diabetic Retinopathy Grading
### EfficientNet-B4 + Swin Transformer Base with CORN Ordinal Classification

> Deep Learning-based Diabetic Retinopathy Severity Grading using the DeepDRiD dataset

---

## Overview

This project implements an automated deep learning system for grading **Diabetic Retinopathy (DR)** from retinal fundus images.

Diabetic Retinopathy is graded according to five ordered severity levels:

| Grade | Severity |
|:---:|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

Since these classes represent an increasing level of disease severity, the problem is treated as an **ordinal classification** task rather than a conventional multiclass classification problem.

The final model combines two complementary deep learning architectures:

- **EfficientNet-B4** for fine-grained visual feature extraction
- **Swin Transformer Base** for learning local and global retinal features
- **CORN (Conditional Ordinal Regression Networks)** for ordinal prediction

The two backbones process the retinal image in parallel. Their learned feature representations are fused before being passed to the ordinal classification head.

---

## Model Architecture

```text
                    Retinal Fundus Image
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
          EfficientNet-B4        Swin Transformer Base
                │                       │
                ▼                       ▼
        Feature Extraction      Feature Extraction
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
                     Feature Fusion
                            │
                            ▼
                       Fusion Head
                            │
                            ▼
                    CORN Ordinal Head
                            │
                     4 Ordinal Logits
                            │
                            ▼
                  DR Severity Grade 0–4
## Why Two Backbones?

The two architectures provide complementary representations of retinal images.

Backbone	Role
EfficientNet-B4	Extracts fine-grained local visual patterns, textures and retinal structures
Swin Transformer Base	Captures local and global contextual relationships using shifted-window attention

The parallel architecture allows the model to combine convolutional and transformer-based representations before classification.

## CORN Ordinal Classification

Instead of directly predicting five independent class probabilities, the final model uses CORN (Conditional Ordinal Regression Networks).

For five DR classes, the model produces four ordinal logits representing cumulative classification decisions:

P(y > 0)
P(y > 1)
P(y > 2)
P(y > 3)

These predictions are converted into the final severity grade from 0 to 4.

This formulation is appropriate for Diabetic Retinopathy because confusing Grade 2 with Grade 3 is less severe than confusing Grade 0 with Grade 4.

## Dataset

This implementation uses the DeepDRiD v1.1 dataset.

The regular fundus image dataset contains four images per patient and provides DR severity annotations for individual retinal images.

For this project:

1,200 images were used for training
400 images were used as the held-out evaluation set
1,600 images were processed in total
Five DR severity classes were used: 0–4

The official DeepDRiD training and validation sets were kept separate for evaluation.

## Class Distribution
### Training Set
| Class | Images |
| 0 | 540 |
| 1 | 140 |
| 2 | 234 |
| 3 | 214 |
| 4 | 72  |
Total	1,200
### Evaluation Set
| Class | Images |
| 0 | 174 |
| 1 | 46  |
| 2 | 92  |
| 3	| 68  |
| 4 | 20  |
Total	400

The evaluation set was not used during model training.

## Preprocessing

The preprocessing pipeline was applied to the retinal fundus images before training and evaluation.

RFOV Cropping

Region of Field of View (RFOV) cropping removes unnecessary black background surrounding the retinal field.

This allows the model to focus more strongly on the relevant retinal region.

CLAHE Enhancement

Contrast Limited Adaptive Histogram Equalisation (CLAHE) is applied to improve local contrast and enhance retinal structures.

The preprocessing pipeline therefore follows:

Original Fundus Image
        │
        ▼
RFOV Cropping
        │
        ▼
CLAHE Enhancement
        │
        ▼
384 × 384 Image
        │
        ▼
Model Input
## Data Augmentation

The training pipeline includes augmentation techniques designed to improve generalisation:

Horizontal flipping
Colour jitter
Random affine transformations
MixUp
Label smoothing
Random erasing

These augmentations introduce controlled variation while preserving the underlying DR severity label.

## Training Configuration
Parameter	Value
Image Size	384 × 384
Number of Classes	5
Batch Size	14
Epochs	20
Optimizer	AdamW
Learning Rate	3 × 10⁻⁵
Weight Decay	1 × 10⁻⁴
Scheduler	Cosine Annealing
EfficientNet	EfficientNet-B4
Transformer	Swin Transformer Base
Loss	CORN
Precision	Mixed Precision (AMP)
Hardware	NVIDIA GPU

Both EfficientNet-B4 and Swin Transformer Base use pretrained weights for transfer learning.

Gradient checkpointing and mixed precision were used to reduce GPU memory requirements during training.

## Model Selection

During training, model checkpoints were saved according to validation Quadratic Weighted Kappa (QWK).

The best checkpoint obtained during training was:

Epoch: 15
Validation QWK: 0.8943

Checkpoint:

model_epoch_015_qwk_0.8943.pth

The validation score above represents the model's internal training-time validation result.

The final performance reported below comes from the separate 400-image held-out DeepDRiD evaluation set.

## Results
Final Evaluation Results

The trained model was evaluated on the 400-image held-out DeepDRiD evaluation set.

| Metric | Result |
| Accuracy | 71.75% |
| QWK | 0.7813 |
| Micro F1 | 0.7175 |
|Weighted F1 | 0.7131 |
| Macro F1 | 0.6400 |
|### Per-Class Performance 
| DR Grade | F1 Score |	Recall |
| 0 — No DR | 0.8034 | 0.8103 |
| 1 — Mild | 0.4444 | 0.4348 |
| 2 — Moderate | 0.6772 | 0.6957 |
| 3 — Severe	| 0.7746 | 0.8088 |
| 4 — Proliferative | 0.5000 | 0.3500 |
## Confusion Matrix

The confusion matrix shows that the model performs particularly well on No DR and Severe DR cases.

Most incorrect predictions occur between neighbouring severity levels. For example:

Mild DR → No DR
Moderate DR → Severe DR
Severe DR → Moderate DR
Proliferative DR → Severe/Moderate DR

This behaviour is consistent with the ordinal nature of the problem.

Large jumps between distant severity grades are relatively uncommon.

## Why QWK Is Important

Quadratic Weighted Kappa (QWK) is used as the primary evaluation metric because DR severity is ordinal.

A prediction such as:

Actual:      Grade 2
Predicted:   Grade 3

is a smaller grading error than:

Actual:      Grade 2
Predicted:   Grade 0

QWK accounts for this difference, making it more suitable for evaluating ordinal DR grading than accuracy alone.

The final model achieved:

QWK = 0.7813

on the held-out evaluation set.

## Key Findings
EfficientNet-B4 + Swin Transformer Base provides a complementary dual-backbone architecture.
CORN is suitable for the ordinal structure of DR severity.
RFOV cropping helps focus the model on the relevant retinal field of view.
CLAHE improves local retinal contrast before model inference.
The model performs strongest on No DR and Severe DR.
Mild DR is more difficult to distinguish from No DR.
Proliferative DR has lower recall, partly due to the relatively small number of Grade 4 samples in the evaluation set.
Most classification errors occur between neighbouring severity grades rather than distant classes.
The final held-out evaluation achieved 71.75% accuracy and 0.7813 QWK.
## Project Structure
Diabetic-Retinopathy-Grading/
│
├── Preprocessing.py
├── Train.py
├── Test.py
├── README.md
├── requirements.txt
├── confusion_matrix.png
├── .gitignore
│
├── dataset/              # Local dataset - excluded from Git
│
├── outputs/              # Local training outputs - excluded from Git
│
└── test_outputs/         # Local evaluation outputs - excluded from Git

Large datasets, trained model checkpoints and generated outputs are intentionally excluded from the Git repository.

## Installation

Clone the repository and install the required dependencies:

git clone <YOUR_REPOSITORY_URL>
cd Diabetic-Retinopathy-Grading

pip install -r requirements.txt
## Usage
1. Prepare the Dataset

Download and extract the DeepDRiD dataset.

The project expects the relevant DeepDRiD regular fundus images and label CSV files to be available locally.

The dataset is intentionally not included in this repository because of its size and dataset licensing/distribution considerations.

2. Configure Preprocessing

Update the paths in the Config section of:

Preprocessing.py

Set:

input_dir
output_dir
csv_path

Then run:

python Preprocessing.py
3. Train the Model

Configure the dataset and output paths in:

Train.py

Then run:

python Train.py

The best checkpoints are saved according to validation QWK.

4. Evaluate the Model

Configure the following paths in:

Test.py
model_path
test_dir
csv_path

Then run:

python Test.py

The evaluation script generates:

test_metrics.csv
test_predictions.csv
confusion_matrix.png
test.log
## Example Output
================ TEST RESULTS ================

Accuracy        : 0.7175
QWK             : 0.7813
Micro F1        : 0.7175
Weighted F1     : 0.7131
Macro F1        : 0.6400

Per-class F1:
C0 = 0.8034
C1 = 0.4444
C2 = 0.6772
C3 = 0.7746
C4 = 0.5000
## Technologies
Python
PyTorch
Torchvision
timm
EfficientNet-B4
Swin Transformer
CORN Ordinal Regression
NumPy
Pandas
Scikit-learn
OpenCV
Matplotlib
CUDA / Mixed Precision
## Limitations

The current system has several limitations:

1.The evaluation dataset contains relatively few Grade 4 (Proliferative DR) samples.
2.Mild DR remains difficult to distinguish from No DR.
3.The current training pipeline uses an image-level validation split internally, while DeepDRiD contains multiple images per patient. Therefore, the internal validation score should not be interpreted as the final generalisation performance.
The reported 0.7813 QWK is therefore based on the separate held-out DeepDRiD evaluation set and is the primary result reported by this implementation.
## Future Work

Potential extensions include:

Patient-level train/validation splitting
Improved handling of class imbalance
Uncertainty estimation and confidence-aware predictions
Explainability using Grad-CAM and transformer attention
Automated image-quality assessment
Interactive web-based inference
Agentic AI orchestration for automated screening workflows
Integration of multiple retinal images for patient-level assessment
## Disclaimer

This project is intended for research and educational purposes only.

The model is not a medical diagnostic device and should not be used as a substitute for assessment by a qualified ophthalmologist or other healthcare professional.

## Acknowledgements

This project uses the DeepDRiD dataset for diabetic retinopathy grading research.

The implementation builds upon established deep learning architectures including EfficientNet and Swin Transformer, with CORN used for ordinal regression.
