# Tea Leaf Disease Classification

A research-oriented deep learning framework designed for the automated identification and classification of diseases in tea plants using state-of-the-art Computer Vision architectures.

## 🚀 Overview
This repository provides a modular and extensible pipeline to train, evaluate, and visualize deep learning models for tea leaf health assessment. [cite_start]It supports a variety of architectures—from standard CNNs to Vision Transformers (ViT) and Hybrid models—specifically tuned for agricultural image datasets.

### Target Classes
The system is configured to classify images into four primary categories:
* **Blight**
* **Healthy_Leaf**
* **Helopeltis**
* **Red_Rust**

---

##  Environment Setup
**Requirement:** This project requires **Python 3.13 or 3.13+**.

1. **Clone the Project:**
   ```bash
   git clone https://github.com/shohel1arman/Tea-Leaf-Disease-Classification
   cd Tea-Leaf-Disease-Classification


## 2. Create a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

## 3. Install Dependencies:
The project requires packages such as torch, torchvision, albumentations, and opencv-python.
```bash
pip install -r requirements.txt
```

## Core Components
1. Model Factory (model_factory.py)
A centralized model hub that allows for easy switching between architectures:  

Classic CNNs: resnet50, vgg19, densenet121.  
Mobile-Optimized: mobilenetv3.  
State-of-the-art: efficientnetb3, vitb16.  
Research Hybrid: hybridmodel (TeaLeafHybridModel).  

2. Augmentation Strategy (data_loader.py)
The pipeline supports three levels of data preparation via the Albumentations library:

None: Simple resizing and ImageNet normalization.

Standard: Geometric rotations, horizontal flips, and brightness adjustments.

Enhanced: Advanced preprocessing featuring CLAHE (Contrast Limited Adaptive Histogram Equalization) and Sharpening to highlight intricate leaf textures and lesion patterns.

3. Loss Functions
Integrated modules to handle class imbalance and prevent overfitting:  


Focal Loss: Addresses class imbalance by down-weighting well-classified examples.  

Label Smoothing: Improves generalization by preventing the model from becoming overconfident.  

Weighted CrossEntropy: Automatically adjusts for dataset distribution using calculated class weights.


## Evaluation & Results
All outputs are automatically organized in the Result/<model_name>_<aug_type>/ directory:

Training History: training_curves.png plots loss and accuracy over epochs.

Confusion Matrix: confusion_matrix.png visualizes per-class performance.

Detailed Metrics: final_trainMetrics.json stores AUC, Sensitivity, Specificity, and F1-macro scores.

Model Checkpoints: The best_model.pth file stores the weights with the highest validation accuracy.

## 📂 Project Structure
```bash
├── models/
│   ├── model_factory.py     # Model selection logic and custom loss functions 
│   └── ...                  # Individual architecture definitions
├── utils/
│   ├── data_loader.py       # Dataset class and Albumentations pipelines
│   └── train_evaluation.py  # Trainer class, EarlyStopping, and Metric tracking
├── run.py                   # Main entry point for training and testing
└── requirements.txt         # List of required Python libraries