# Crumble Vision · Biscuit Tray Inspector

**Real‑time defect detection on production biscuit trays.**  
Fine‑tuned ResNet‑50 | 99.7% accuracy | Edge‑ready inference

---
##  Overview

Crumble Vision automates quality control for biscuit production lines.  
Each 3×3 tray is sliced into individual biscuit cells. A ResNet‑50 model classifies each cell into one of four classes:

| Class            | Meaning                         |
|------------------|---------------------------------|
| `Defect_No`      | Perfect biscuit                 |
| `Defect_Shape`   | Broken, misshapen, or deformed  |
| `Defect_Object`  | Foreign object or contamination |
| `Defect_Color`   | Over‑/under‑baked, discoloured  |

The system outputs a **tray quality score** = (% of `Defect_No` cells).  
A Streamlit dashboard visualises results, confusion matrices, and per‑tray reports.

---

##  Model Card

### Model Overview
- **Developer:** Crumble Pakistan QA Team  
- **Model Type:** Fine‑tuned ResNet‑50 (image classification)  
- **Task:** Classify individual biscuit crops into 4 defect categories  

### Intended Use
- **Primary use:** Automated quality inspection on biscuit production lines  
- **Intended users:** QA engineers, production supervisors  
- **Out of scope:** Non‑biscuit food items, safety‑critical applications  

### Architecture
- Base: ResNet‑50 pretrained on ImageNet  
- Modifications: Replaced final FC layer → 4 outputs; all layers fine‑tuned  

### Training Data
- **Dataset:** Proprietary biscuit images (Crumble Pakistan)  
- **Classes:** 4 (Defect_No, Shape, Object, Color)  
- **Split:** 70% train / 15% val / 15% test (stratified)  
- **Preprocessing:** Resize to 224×224; training augmentations (flip, rotation, colour jitter)  

### Performance (Test Set – 735 cells)

| Class          | Precision | Recall | F1 | Support |
|----------------|-----------|--------|----|---------|
| Defect_No      | 1.00      | 1.00   | 1.00 | 284 |
| Defect_Shape   | 1.00      | 1.00   | 1.00 | 279 |
| Defect_Object  | 1.00      | 0.99   | 0.99 | 95 |
| Defect_Color   | 0.99      | 1.00   | 0.99 | 77 |

**Overall accuracy:** 99.7% (only 2 misclassifications)

Confusion matrix:
```text
[[284    0    0    0]
 [  1  278    0    0]
 [  0    0   94    1]
 [  0    0    0   77]]
```


### Limitations
- May not generalise to lighting conditions or biscuit shapes not seen during training  
- Rare false negatives (defect → `Defect_No`) – a secondary manual check is advised for critical holds  

### Ethical Considerations
Designed to reduce waste and improve consistency. No worker displacement or safety risks are intended.

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sobanmujtaba/crumble-vision.git
cd crumble-vision
```

### 2. Install dependencies
```bash
pip install torch torchvision opencv-python pandas streamlit scikit-learn matplotlib plotly seaborn
```
### 3. Prepare the dataset
Place your annotated dataset in data/ with the following structure:

```text
data/
├── Annotations.csv        # columns: file, classDescription
└── Images/                # all biscuit images
```
Run the preprocessing script to create train/val/test splits:

```bash
python src/preprocess.py
```
### 4. Train the model (optional)
```bash
python src/train_resnet.py
```
The best model will be saved to models/resnet50_biscuit.pth.

### 5. Generate simulated trays
```bash
python src/tray_simulator.py
```
This creates 82 tray images (3×3 grid) in assets/trays/ and a manifest file.

### 6. Run inference on all trays
```bash
python src/tray_inference.py
```
Outputs assets/trays/tray_predictions.csv with per‑cell predictions.

### 7. Launch the dashboard
```bash
streamlit run src/dashboard.py
```
Your browser will open an interactive dashboard with metrics, confusion matrix, and tray‑by‑tray inspection.

### Demo
[crumble-vision](https://sobanmujtaba.github.io/crumble-vision/)

### License
MIT © Soban Mujtaba
