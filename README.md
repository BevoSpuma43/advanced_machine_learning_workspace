# Advanced Machine Learning Project: Real vs. Synthetic Feature Classification

## Overview

This project investigates binary classification of **precomputed spatial feature maps** extracted from real and AI-generated images. The task is to distinguish **real** examples from **synthetic** examples while keeping data handling, model selection, and evaluation reproducible and leakage-aware.

The experimental design compares five classifiers spanning kernel methods, gradient-boosted trees, dense neural networks, and spatial convolutional architectures. It also evaluates a deployable soft-voting ensemble composed of the locally executable models. The central research question is whether an ensemble of heterogeneous local models can provide a competitive alternative to a higher-capacity ConvNeXt classifier trained on Google Colab.

The repository contains self-contained Jupyter notebooks designed to run from a clean kernel both locally and on Google Colab. Each experiment uses the same class mapping, sample index, stratified split, and artifact contract.

> **Scope:** This repository describes the experimental methodology and implementation. It intentionally does not report benchmark results in this landing page.

## Dataset

The dataset consists of two NumPy arrays containing `float32` spatial feature maps. Each sample has shape `(1280, 16, 16)`, i.e., 1,280 channels over a 16 × 16 spatial grid (327,680 values per sample). The original RGB images are not included; classification operates directly on the feature tensors.

| Source file | Label | Semantic class | Samples | Tensor shape | Origin |
|---|---:|---|---:|---|---|
| `features_0.npy` | 0 | real | 1,654 | `(1654, 1280, 16, 16)` | Pascal VOC and COCO |
| `features_1.npy` | 1 | fake | 2,383 | `(2383, 1280, 16, 16)` | DALL·E 3, Stable Diffusion 1.4, Stable Diffusion 1.5, and Midjourney |
| **Total** | — | — | **4,037** | per sample: `(1280, 16, 16)` | — |

The positive class is consistently defined as **fake** (`label = 1`). Raw feature arrays are loaded with NumPy memory mapping, so the two multi-gigabyte tensors are never concatenated in host memory.

### Derived representations

`dataset_exploration.ipynb` creates deterministic, reusable representations and metadata:

- **Global average pooling (GAP):** 1,280 features per sample, obtained by averaging each channel across the 16 × 16 grid.
- **Channel mean + standard deviation:** 2,560 features per sample, ordered as the 1,280 channel means followed by the 1,280 channel standard deviations.
- **Spatial normalization statistics:** per-channel mean and standard deviation estimated on training samples only, then reused by both spatial neural networks.
- **Sample index and split artifacts:** stable `sample_id` values, source-file references, source indices, labels, and a shared split hash.

## Experimental Protocol

All implemented models follow the same protocol:

1. A single stratified split is shared across notebooks: 70% training, 15% validation, and 15% test, with seed 42.
2. Training data alone is used to fit learned preprocessing components, including scalers and spatial normalization statistics.
3. Validation balanced accuracy is the primary criterion for hyperparameter selection, early stopping, ensemble-weight selection, and decision-threshold selection.
4. The test partition remains isolated until all model choices are frozen.
5. Predictions are persisted per sample using a common schema, enabling exact alignment across models and downstream ensemble analysis.

No input augmentation, synthetic oversampling, SMOTE, MixUp, CutMix, random masking, or per-sample normalization is applied. This ensures that observed differences can be attributed to the representation and model family rather than stochastic input transformations.

## Models Considered

| Model | Input representation | Role |
|---|---|---|
| RBF SVM | GAP and channel mean + standard deviation | Kernel baseline / ensemble component |
| XGBoost | GAP and channel mean + standard deviation | Gradient-boosted tree baseline / ensemble component |
| MLP | GAP and channel mean + standard deviation | Dense neural-network baseline |
| CNN spatial head | Normalized `(1280, 16, 16)` maps | Local spatial model / ensemble component |
| Compact ConvNeXt | Normalized `(1280, 16, 16)` maps | Higher-capacity spatial reference, primarily trained on Colab |
| Soft-voting ensemble | Calibrated SVM, XGBoost, and CNN probabilities | Local heterogeneous ensemble |

An additional logistic-regression model based only on global mean, standard deviation, minimum, and maximum is used as a **diagnostic confounder check**, not as a final candidate model. A DeconvNet image-reconstruction extension is deliberately out of scope because paired original images are unavailable.

## Model Pipelines

### 1. Dataset exploration and shared data contract

```text
Memory-mapped raw feature arrays
    → integrity checks, file inventory, and duplicate checks
    → stable sample index
    → stratified shared split
    → deterministic GAP and mean+std caches
    → training-only spatial channel statistics
    → near-duplicate, outlier, and confounder diagnostics
```

The exploration notebook is responsible for producing the dataset manifest, split artifact, tabular caches, spatial normalization artifact, and diagnostic figures. Analyses that may influence modelling decisions are restricted to training data after the split has been created.

### 2. RBF Support Vector Machine

```text
GAP or mean+std cache
    → shared train / validation / test indices
    → StandardScaler fitted on training data only
    → optional PCA fitted inside the training pipeline
    → RBF SVM hyperparameter search
    → cross-validated sigmoid probability calibration on training data
    → validation-selected decision threshold
    → frozen evaluation and per-sample predictions
```

The SVM pipeline searches `C`, `gamma`, and optional class weighting using validation balanced accuracy. PCA, when enabled, is encapsulated in the scikit-learn pipeline to prevent leakage.

### 3. XGBoost

```text
Mean+std cache (primary) or GAP cache (ablation)
    → independent validation of each pooling representation
    → Optuna TPE hyperparameter optimization
    → XGBoost histogram tree method with pruning and early stopping
    → best-trial refit
    → Platt scaling on validation margins
    → validation-selected decision threshold
    → frozen evaluation and portable model bundle
```

XGBoost consumes the original `float32` tabular features without `StandardScaler` or PCA. The mean+std and GAP representations are optimized independently so that pooling is itself selected without test-set access.

### 4. Multi-Layer Perceptron

```text
Mean+std cache (primary) or GAP cache (ablation)
    → StandardScaler fitted on training data only
    → float32 tensors and mini-batch data loaders
    → Linear(input, 512) → LayerNorm → GELU → Dropout
    → Linear(512, 128) → LayerNorm → GELU → Dropout
    → Linear(128, 1)
    → AdamW, learning-rate scheduling, gradient clipping, and early stopping
    → validation-selected threshold and TorchScript export
```

The MLP is trained with `BCEWithLogitsLoss`. Automatic mixed precision is enabled only when CUDA is available; otherwise the notebook uses a CPU-safe fallback.

### 5. CNN Spatial Head

```text
Memory-mapped spatial feature map
    → sample-wise float32 retrieval through sample_index.csv
    → training-only per-channel normalization
    → 1×1 channel projection and GroupNorm
    → residual depthwise-separable convolutional blocks
    → channel attention
    → global average pooling + global max pooling
    → MLP classifier head
    → AdamW, AMP when available, gradient clipping, and early stopping
```

The CNN is intentionally compact to remain compatible with a local GPU with limited VRAM. Group normalization is used instead of batch normalization to accommodate small batch sizes.

### 6. Compact ConvNeXt

```text
Memory-mapped spatial feature map
    → same training-only channel normalization as the CNN
    → 1×1 stem: 1280 → 256 channels
    → 3 ConvNeXt blocks at 16 × 16
    → downsample: 256 → 384 channels at 8 × 8
    → 3 ConvNeXt blocks
    → downsample: 384 → 512 channels at 4 × 4
    → 2 ConvNeXt blocks
    → LayerNorm, global average pooling, dropout, binary classifier
```

The compact ConvNeXt uses depthwise 7 × 7 convolutions, layer scaling, GELU MLP blocks, stochastic depth, warm-up plus cosine learning-rate decay, gradient accumulation, automatic mixed precision, and resumable checkpoints. It is optimized for Colab execution while maintaining the same data contract as local notebooks.

### 7. Calibrated Soft-Voting Ensemble

```text
Saved SVM probabilities ───────────────┐
Saved XGBoost probabilities ───────────┼→ probability alignment and contract validation
Saved CNN logits → temperature scaling ┘
                                        → uniform and weighted soft voting
                                        → validation-only weight and threshold search
                                        → frozen test evaluation
                                        → diversity and paired statistical analysis
```

The ensemble contains **only** the RBF SVM, XGBoost, and CNN spatial head. SVM and XGBoost probabilities are calibrated in their respective notebooks; CNN logits receive validation-fitted temperature scaling. Uniform voting and non-negative weighted voting are evaluated, with weights constrained to sum to one. ConvNeXt is kept outside the ensemble and used only as an external comparison target.

## Reproducibility and Artifact Management

- Every notebook is executable from a clean kernel and detects local versus Google Colab execution automatically.
- Random seeds, package versions, hardware information, configuration, and split hashes are persisted with each run.
- The best checkpoint is selected by validation balanced accuracy rather than by final epoch.
- Checkpoints, model bundles, TorchScript exports, histories, configurations, metrics, predictions, and figures are written to timestamped run directories under `results/`.
- On Colab, raw arrays can be cached in the transient runtime for faster reads, while processed data and outputs remain on Google Drive.

## Repository Layout

```text
.
├── artifacts/                         # Dataset manifest and shared split
├── dataset/                           # Raw arrays (not versioned) and derived caches
├── docs/
│   └── plan.md                        # Experimental specification
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── 01_svm_rbf.ipynb
│   ├── 02_xgboost.ipynb
│   ├── 03_mlp.ipynb
│   ├── 04_cnn_spatial_head.ipynb
│   ├── 05_convnext_colab.ipynb
│   └── 06_ensemble_svm_xgb_cnn.ipynb
├── results/                           # Run-specific artifacts and diagnostics
└── requirements.txt
```

## Execution Order

1. Run `dataset_exploration.ipynb` to validate the raw arrays and create all shared artifacts.
2. Run the SVM, XGBoost, MLP, CNN spatial-head, and ConvNeXt notebooks.
3. Run `06_ensemble_svm_xgb_cnn.ipynb` only after SVM, XGBoost, CNN, and ConvNeXt predictions are available.

The raw feature arrays should be excluded from version control because of their size. They must be placed in `dataset/` before executing the notebooks.
