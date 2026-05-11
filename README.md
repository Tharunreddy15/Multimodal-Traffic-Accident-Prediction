# 🚦 Multimodal Traffic Accident Severity Prediction

> Predicting accident severity by fusing structured tabular data with accident scene images using deep learning.

---

##  Overview

This project tackles a real-world classification problem — predicting the severity of traffic accidents as **Slight**, **Serious**, or **Fatal** — using a **multimodal late-fusion deep learning model** that combines:

-  **Tabular data** — road type, weather, speed limit, time of day, number of casualties, etc.
-  **Accident scene images** — visual cues from the crash environment

Two models are implemented and benchmarked:

| Model | Approach | Macro F1 |
|---|---|---|
| Tabular Baseline | MLP + Focal Loss | ~81% |
| **Multimodal Fusion** | ResNet-18 + MLP → Late Fusion | **~96%** |

Adding visual context improved Macro F1 by **~15 percentage points**, especially for the minority Fatal class.

---

##  Project Structure

```
.
├── notebook.ipynb                  # Full training pipeline
├── predictions.csv                 # Validation set predictions with probabilities
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Excludes dataset and model weights
```

> **Note:** The dataset (`ACCIDENTDATA_PROJECT_UNIQUE_IMAGES.csv`) and model checkpoints (`.pt` files) are not included due to size. See Dataset section below.

---

##  Model Architecture

### Tabular Baseline (MLP)
```
Input (30 features) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(3)
Loss: Focal Loss with inverse class-frequency alpha weights
```

### Multimodal Late Fusion (Main Model)
```
Image Branch:   ResNet-18 (pretrained, fc removed) → 512-dim feature vector
Tabular Branch: MLP → 128-dim feature vector
Fusion:         Concat(512 + 128 = 640) → Linear(128) → ReLU → Linear(3)
```

**Why Focal Loss?**
Standard cross-entropy ignores minority classes (Fatal accidents are rare). Focal Loss applies higher weights to hard, misclassified examples — forcing the model to learn them.

---

##  Results

| Metric | Tabular Only | Multimodal Fusion |
|---|---|---|
| Overall Accuracy | ~82% | ~98% |
| Macro F1-Score | ~81% | ~96% |
| Fatal Class Recall | Low | Significantly improved |

Evaluation includes:
- Per-class classification report
- Confusion matrix
- Training vs. validation loss & accuracy curves
- Wrong-prediction visual inspection

---

##  Dataset

The dataset is **not included** in this repository due to size constraints.

**Tabular features include:**
- Speed limit, road type, road surface conditions
- Weather and light conditions
- Time features (hour, day of week)
- Number of vehicles and casualties

**Label mapping:**
```python
label_map = {1: 0, 2: 1, 3: 2}  # Slight → 0, Serious → 1, Fatal → 2
```

To run the notebook, place your dataset at:
```
data/ACCIDENTDATA_PROJECT_UNIQUE_IMAGES.csv
```
And update the path in Cell 2 of the notebook accordingly.

---

##  Setup & Installation

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow
```

---

##  How to Run

Open `notebook.ipynb` and run cells in order:

1. **Load & clean CSV** — drop duplicates, map labels
2. **Feature preprocessing** — `LabelEncoder` for categoricals, `StandardScaler` for numerics
3. **Train/val split** — stratified split preserving class ratios
4. **Train Tabular MLP** — with Focal Loss, saves `best_tabular_focal.pt`
5. **Build Multimodal Dataset** — aligns tabular rows with image paths, applies transforms
6. **Train Fusion Model** — ResNet-18 + MLP, saves `best_3class_fusion.pt`
7. **Evaluate** — confusion matrix, learning curves, wrong-prediction analysis
8. **Inference demo** — single-sample prediction with probability output

---

##  Inference Example

```python
pred_label, confidence, probs = predict_one_3class(X_val[0], img_val[0])

# Output:
# Pred: Slight  Conf: 0.91
# All probs [Slight, Serious, Fatal]: [0.91, 0.07, 0.02]
```

---

##  Limitations

- Performance depends on correct `Image_path` mapping in the CSV
- Serious vs. Fatal classes are visually similar — some confusion is expected
- ResNet-18 backbone is frozen; fine-tuning may further improve results
- Augmentation (rotation, lighting) not yet applied — listed as future work

---

##  Future Work

- Apply image augmentation (rotation, brightness, flipping) to improve generalization
- Replace ResNet-18 with ResNet-50 or EfficientNet for stronger image features
- Integrate YOLO-based cropping to focus on the vehicle before passing to CNN
- Build a Streamlit dashboard for interactive severity prediction

---

##  Authors

- Tharun Reddy Kondavabahalli Venkatarayareddy

*M.Sc. Data Science — University of Europe for Applied Sciences, Potsdam*
