# HoverNet-GNN-Classification
# Hybrid Deep Learning Model for Breast Cancer Tumor Segmentation and Size Estimation

## Using HoVerNet and Graph Neural Networks

A Flask-based web application that performs **automated breast cancer histopathology analysis** using a hybrid HoVerNet-GNN deep learning model. The system detects cell nuclei, segments tumor regions, estimates tumor size, constructs spatial cell relationship graphs, and provides explainable AI (XAI) heatmaps — all from a single uploaded histopathology image.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Spatial Connections — What They Are & How They Work](#3-spatial-connections--what-they-are--how-they-work)
4. [Thresholds & Grading System — How Predictions Are Made](#4-thresholds--grading-system--how-predictions-are-made)
5. [XAI (Explainable AI) Heatmaps](#5-xai-explainable-ai-heatmaps)
6. [Model Comparison](#6-model-comparison)
7. [Installation & Usage](#7-installation--usage)
8. [Dataset Information](#8-dataset-information)
9. [Project Structure](#9-project-structure)

---

## 1. Project Overview

### What This System Does

When a user uploads a breast cancer histopathology image (H&E stained tissue), the system performs the following pipeline:

1. **Image Preprocessing** — Resizes to 256×256, normalizes pixel values using ImageNet statistics.
2. **Nucleus Detection** — The HoVerNet encoder-decoder identifies individual cell nuclei by predicting a nucleus probability map (values 0–1 per pixel).
3. **Segmentation** — Pixels with nucleus probability > 0.5 are classified as nuclei; connected components are labeled as individual cells.
4. **HV Gradient Estimation** — Predicts horizontal and vertical distance maps to separate touching/overlapping nuclei.
5. **Density Classification** — A hybrid CNN + histogram feature classifier predicts whether the tissue is **Low Density**, **Medium Density**, or **High Density** (3-class classification).
6. **Spatial Graph Construction** — Detected nuclei become graph nodes; edges connect nearby cells to model spatial relationships (GNN component).
7. **Tumor Grading** — Combines density, cell size variation (pleomorphism), and tumor area coverage into a composite grade (Grade 1–3).
8. **XAI Explanation** — Generates heatmaps and natural language explanations of what the model focused on.

### Expected Inputs

- Histopathology images (JPG/PNG, max 16MB)
- H&E (Hematoxylin & Eosin) stained breast tissue slides
- Original dataset images are 1000×1000 pixels from TCGA-BRCA

### Expected Outputs

- Nuclei identification and count
- Nucleus segmentation masks (red overlay)
- Cell density classification (Low / Medium / High)
- Cell density and spatial relationship maps
- Tumor area/size estimation (percentage)
- Tumor grade (Grade 1, 2, or 3)
- Explainability heatmaps (XAI) with written explanation
- HV gradient visualization for nucleus boundaries
- Spatial relationship graph visualization
- Model comparison table and charts (5 models)

---

## 2. System Architecture

### HoVerNet-GNN Hybrid Model

```
Input Image (256×256×3)
        │
        ▼
┌──────────────────┐
│  HoVerNet Encoder │  ← ResNet-34 backbone (pretrained on ImageNet)
│  (4 residual      │     Extracts multi-scale feature maps:
│   blocks)         │     64 → 64 → 128 → 256 → 512 channels
└──────────────────┘
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────┐
│    U-Net Decoder │              │ CNN Feature Projector│
│ (Skip connections│              │  AdaptiveAvgPool →   │
│   from encoder)  │              │  Linear(512→256)     │
└──────────────────┘              └──────────────────────┘
        │                                     │
   ┌────┴────┐                                │
   ▼         ▼                                │
┌──────┐ ┌──────┐                             │
│ NP   │ │ HV   │                             │
│Branch│ │Branch│                             │
│(1ch) │ │(2ch) │                             │
└──────┘ └──────┘                             │
   │         │                                │
   ▼         ▼                    ┌───────────┴───────────┐
Nucleus   Horizontal/             │                       │
Prob Map  Vertical                ▼                       ▼
          Gradients      ┌──────────────┐      ┌──────────────┐
                         │ Histogram    │      │              │
                         │ Feature      │      │  COMBINED    │
                         │ Projector    │      │  (concat)    │
                         │ Linear(33→64)│      │  320 → 128   │
                         └──────────────┘      │  → 64 → 3    │
                                │              └──────────────┘
                                │                     │
                                └─────────────────────┘
                                                      │
                                                      ▼
                                            Classification Output
                                         (Low/Medium/High Density)
```

### Key Components

| Component | Purpose | Architecture |
|---|---|---|
| **Encoder** | Extract image features | ResNet-34 (pretrained) |
| **Decoder** | Reconstruct spatial maps | U-Net style with skip connections |
| **NP Branch** | Nucleus probability per pixel | Conv → Sigmoid (0–1 output) |
| **HV Branch** | Horizontal/Vertical gradients | Conv → 2-channel output |
| **CNN Projector** | Compress encoder features | AdaptiveAvgPool → Linear(512→256) |
| **Histogram Projector** | Process color distribution | Linear(33→128→64) |
| **Classifier** | Final density prediction | Linear(320→128→64→3) with dropout |

---

## 3. Spatial Connections — What They Are & How They Work

### What Are Spatial Connections?

Spatial connections represent the **Graph Neural Network (GNN)** aspect of this hybrid model. They model how cells relate to each other in physical space — a critical factor in tumor analysis because:

- **Tightly clustered cells** (many connections) → aggressive, proliferative tissue
- **Sparsely scattered cells** (few connections) → benign or low-activity tissue
- **Connection patterns** reveal tissue organization and potential malignancy

### How Spatial Connections Are Built (Step by Step)

#### Step 1: Detect Nuclei Centroids

After the HoVerNet model predicts the nucleus probability map, pixels with probability > 0.5 are treated as nucleus regions. Connected component labeling identifies individual nuclei, and each nucleus's **centroid** (center of mass) is computed.

```python
# From app.py — Nucleus detection
nuclei_mask = (np_map > 0.5).astype(np.uint8)          # Threshold probability map
labeled_array, num_nuclei = scipy_label(nuclei_mask)     # Label connected components

# Compute centroids for each nucleus
for i in range(1, num_nuclei + 1):
    coords = np.where(labeled_array == i)
    cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
    centroids.append((cx, cy))
```

#### Step 2: Create Graph Nodes

Each detected nucleus becomes a **node** in the graph. The node is positioned at its centroid coordinates. In the visualization:
- **Green dots** = nucleus centroids (graph nodes)

#### Step 3: Create Graph Edges (Connections)

Two nuclei are connected by an **edge** if their Euclidean distance is less than **50 pixels**. This threshold represents approximately 25–50 micrometers in real tissue (depending on magnification), which is the typical range for cell-to-cell interactions.

```python
# From app.py — Edge construction
for i, (x1, y1) in enumerate(centroids):
    for j, (x2, y2) in enumerate(centroids[i+1:], i+1):
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if dist < 50:  # 50-pixel proximity threshold
            cv2.line(spatial_img, (x1, y1), (x2, y2), (255, 255, 0), 1)
            num_connections += 1
```

In the visualization:
- **Yellow lines** = spatial edges connecting nearby nuclei

#### Step 4: Interpret the Graph

| Metric | Low Connections | High Connections |
|---|---|---|
| **Meaning** | Cells are far apart, sparse tissue | Cells are packed together, dense tissue |
| **Implication** | Likely benign or low-grade | Potentially aggressive or high-grade |
| **Visual** | Few yellow lines, scattered green dots | Dense mesh of yellow lines |

### Why This Matters in Cancer Diagnosis

In real pathology, cell clustering patterns are one of the hallmarks of tumor aggressiveness:
- **Well-differentiated tumors (Grade 1)**: Cells maintain organized spacing similar to normal tissue
- **Poorly-differentiated tumors (Grade 3)**: Cells lose spatial organization, form irregular dense clusters
- The spatial graph captures this organizational pattern quantitatively

---

## 4. Thresholds & Grading System — How Predictions Are Made

### Overview of All Thresholds

The system uses several thresholds at different stages. Here is a complete breakdown of every threshold, what it does, and how it affects the final prediction.

### 4.1 Nucleus Detection Threshold

| Parameter | Value | What It Does |
|---|---|---|
| **Nucleus probability** | **> 0.5** | If a pixel's nucleus probability exceeds 0.5, it's classified as "nucleus" (foreground). Below 0.5 = background tissue. |

This comes from the NP (Nucleus Probability) branch of the model, which outputs a Sigmoid activation (0 to 1) for each pixel.

### 4.2 Spatial Proximity Threshold

| Parameter | Value | What It Does |
|---|---|---|
| **Edge distance** | **< 50 pixels** | Two nuclei within 50 pixels of each other are connected in the spatial graph. Beyond 50 pixels = no connection. |

This threshold controls network density. A smaller value would create sparser graphs; a larger value would connect more distant cells.

### 4.3 Density Classification (Neural Network)

The model's CNN + histogram classifier outputs probabilities for three classes:

| Class | Label | Meaning |
|---|---|---|
| 0 | **Low Density** | Few nuclei detected, sparse tissue |
| 1 | **Medium Density** | Moderate number of nuclei |
| 2 | **High Density** | Many nuclei packed together |

The class with the highest probability becomes the prediction. This is the model's learned classification — it was trained on 4,000 histopathology patches labeled by `cell_index` values from the dataset (percentile-based binning into 3 classes).

### 4.4 Tumor Grading — The Composite Score System

Tumor grading combines **three independent scores** to produce a final grade. This mimics the clinical **Nottingham Grading System** used by pathologists.

#### Score 1: Density Score (d_score) — from model classification

| Model Classification | d_score |
|---|---|
| High Density | 3 |
| Medium Density | 2 |
| Low Density | 1 |

#### Score 2: Pleomorphism Score (p_score) — from nucleus size variation

| Size Variation (%) | p_score | Meaning |
|---|---|---|
| > 40% | 3 | Highly variable cell sizes (irregular) |
| > 20% | 2 | Moderately variable |
| ≤ 20% | 1 | Uniform cell sizes (regular) |

Size variation is calculated as: `(std of nucleus sizes / mean of nucleus sizes) × 100`

A high variation means cells are very different sizes — a hallmark of malignancy.

#### Score 3: Tumor Area Score (a_score) — from segmentation coverage

| Tumor Area (%) | a_score | Meaning |
|---|---|---|
| > 30% | 3 | Large tumor coverage |
| > 15% | 2 | Moderate coverage |
| ≤ 15% | 1 | Small tumor area |

Tumor area is: `(number of nucleus pixels / total pixels) × 100`

#### Final Grade Calculation

```
Total Score = d_score + p_score + a_score   (range: 3 to 9)
```

| Total Score | Grade | Clinical Interpretation |
|---|---|---|
| **≥ 8** | **Grade 3 (High Malignancy)** | Aggressive tumor — all three factors are high |
| **≥ 5** | **Grade 2 (Intermediate)** | Moderate concern — mixed signals |
| **< 5** | **Grade 1 (Low Malignancy)** | Low aggressiveness — mostly benign characteristics |

#### Grading Examples

| Example | Density | Size Var | Area | d | p | a | Total | Grade |
|---|---|---|---|---|---|---|---|---|
| Sparse normal tissue | Low | 15% | 4% | 1 | 1 | 1 | **3** | Grade 1 |
| Moderate tissue | Medium | 25% | 18% | 2 | 2 | 2 | **6** | Grade 2 |
| Small aggressive lesion | High | 45% | 8% | 3 | 3 | 1 | **7** | Grade 2 |
| Large aggressive tumor | High | 50% | 35% | 3 | 3 | 3 | **9** | Grade 3 |
| Dense but small & uniform | High | 10% | 5% | 3 | 1 | 1 | **5** | Grade 2 |

> **Key design decision**: Grade 3 requires a total score ≥ 8 (out of 9), meaning at least two factors must score 3. This prevents, for example, a 4% tumor area from being graded as Grade 3 even if density and pleomorphism are high.

---

## 5. XAI (Explainable AI) Heatmaps

### What the XAI Heatmap Shows

The XAI overlay is built by blending the **nucleus probability map** over the original image using the JET colormap:

- **Red/Yellow (hot) regions** → High nucleus probability — the model is confident nuclei exist here
- **Blue/Green (cool) regions** → Low probability — likely background or stromal tissue
- **The blend alpha is 0.5** — 50% original image, 50% heatmap

### Dynamic XAI Explanation

The system generates **context-aware 2–3 line explanations** that describe:

1. **What the heatmap shows** and the model's confidence level
2. **Clinical interpretation** based on the density category (high/medium/low)
3. **Quantitative summary** — tumor area percentage, nuclei count, spatial connections, and resulting grade

Example explanation for a High Density prediction:
> *"The heatmap highlights regions the model focused on most during classification. Warmer (red/yellow) areas indicate higher nucleus probability, contributing to the "High Density" prediction with 87.3% confidence. Dense clusters of hot regions suggest a high concentration of cell nuclei, which is a hallmark of aggressive tumor tissue requiring further clinical evaluation. Tumor area covers 28.5% of the image with 142 detected nuclei and 318 spatial connections, resulting in Grade 2 (Intermediate)."*

### How XAI Is Computed

```python
# Nucleus probability map from model → used as attention/saliency map
np_map = output['np'][0, 0].cpu().numpy()

# Create colored heatmap overlay
heatmap_colored = cv2.applyColorMap((np_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
xai_overlay = (1 - 0.5) * original_image + 0.5 * heatmap_colored
```

This is a form of **inherent explainability** — the nucleus probability map naturally shows where the model "looks" in the image. Unlike post-hoc methods (LIME, SHAP), this comes directly from the model's architecture.

---

## 6. Model Comparison

Five models are trained and compared:

| Model | Type | Architecture | Key Feature |
|---|---|---|---|
| **Simple CNN** | Baseline | 4 conv layers + FC | Simplest baseline |
| **ResNet50** | Transfer Learning | ResNet-50 (pretrained) | Deep residual features |
| **HoVerNet** | Segmentation + Classification | ResNet-34 encoder + U-Net decoder | Nucleus segmentation branches |
| **HoVerNet-GNN** | **Hybrid (Deployed)** | HoVerNet + histogram features | Combines visual + statistical features |
| **RNN-LSTM** | Sequential | CNN + Bidirectional LSTM | Treats spatial features as sequences |

### Current Performance (from test set)

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Simple CNN | 92.67% | 92.71% | 92.91% | 92.67% |
| ResNet50 | 90.83% | 90.83% | 90.89% | 90.83% |
| HoVerNet | 90.83% | 90.89% | 91.11% | 90.83% |
| **HoVerNet-GNN** | **90.17%** | **90.04%** | **90.10%** | **90.17%** |
| RNN-LSTM | 92.50% | 92.39% | 92.49% | 92.50% |

> **Note**: HoVerNet-GNN is deployed because it provides the richest feature set (segmentation masks, HV gradients, spatial graphs, density maps) even though its raw classification accuracy is comparable to simpler models. The segmentation and spatial analysis capabilities are essential for the clinical visualization requirements of this project.

---

## 7. Installation & Usage

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (optional, CPU works but slower)

### Setup

```bash
# Clone or navigate to project
cd C543

# Install dependencies
pip install -r requirements.txt

# Required packages (if requirements.txt is minimal):
pip install flask torch torchvision opencv-python numpy pillow scipy
```

### Running the Application

```bash
python app.py
```

Then open **http://localhost:5001** in your browser.

### Usage Steps

1. Upload a histopathology image (JPG/PNG) via drag-and-drop or click-to-browse
2. Click **"Analyze Image"**
3. View results:
   - Classification badge (Low / Medium / High Density)
   - Metrics cards (nuclei count, tumor area %, cell density, tumor grade, spatial connections)
   - Visualization tabs (Original, Segmentation, Nucleus Map, Density Map, XAI Heatmap, Spatial Graph, HV Gradient)
4. Scroll down to see model comparison table and charts

### Where to Get Test Images

- **Your Kaggle dataset**: `kaggle.com/datasets/prathameshanvekar/afssssssasf` (TCGA-BRCA patches, 1000×1000 JPG)
- **Google Images**: Search "breast cancer histopathology H&E" for sample stained tissue images
- **BreakHis dataset**: [kaggle.com/datasets/ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis)

---

## 8. Dataset Information

| Property | Value |
|---|---|
| **Source** | TCGA-BRCA (The Cancer Genome Atlas — Breast Cancer) |
| **Kaggle Path** | `/kaggle/input/afssssssasf/` |
| **Patient Folders** | 299 TCGA folders |
| **Images Used** | ~4,000 patches (40 folders × 100 images) |
| **Image Dimensions** | 1000×1000 pixels |
| **Color Mode** | RGB (H&E stained) |
| **Format** | JPEG (~315 KB average) |
| **CSV Features** | `patch_info.csv` per folder with `cell_index` and 765 RGB histogram bins |
| **Label Strategy** | 3-class via percentile binning of `cell_index` (33rd/66th percentile) |
| **Split** | 70% Train / 15% Validation / 15% Test (stratified) |

---

## 9. Project Structure

```
C543/
├── app.py                          # Flask backend (model loading, analysis pipeline, routes)
├── requirements.txt                # Python dependencies
├── breast-tumor-full-code-new.ipynb # Training notebook (Kaggle/Colab)
├── test_verify.py                  # Model verification script
├── README.md                       # This file
│
├── models/                         # Trained PyTorch model weights
│   ├── hovernet_gnn_model_FIXED.pth  # Deployed HoVerNet-GNN (90.17%)
│   ├── hovernet_model.pth            # HoVerNet standalone
│   ├── resnet50_model.pth            # ResNet50
│   ├── rnn_lstm_model.pth            # RNN-LSTM
│   └── simple_cnn_model.pth          # Simple CNN
│
├── results/                        # Evaluation metrics
│   ├── test_results.json             # All 5 models' test metrics
│   ├── hovernet_gnn_results.json     # HoVerNet-GNN specific results
│   └── model_comparison.csv          # CSV comparison
│
├── templates/
│   └── index.html                  # Frontend HTML template
│
└── static/
    ├── css/style.css               # Styling
    ├── js/main.js                  # Frontend JavaScript
    ├── uploads/                    # Uploaded images (auto-created)
    └── plots/                      # Generated plots (from training)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js, Font Awesome |
| **Backend** | Flask (Python) |
| **Deep Learning** | PyTorch, TorchVision |
| **Image Processing** | OpenCV, Pillow, SciPy |
| **Training** | Kaggle (NVIDIA Tesla T4 GPU), Mixed Precision (AMP) |
| **Dataset** | TCGA-BRCA via Kaggle |

---

## Disclaimer

This system is a **research prototype** for academic purposes. It is NOT a medical diagnostic tool. All results should be interpreted by qualified pathologists. Do not use this system for clinical decision-making.
