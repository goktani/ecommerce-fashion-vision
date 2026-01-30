# 👔 E-Commerce Fashion Vision (EfficientNet-B3)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Overview
This project implements a high-performance image classification pipeline for men's clothing in an e-commerce context. Using **Transfer Learning** with an **EfficientNet-B3** architecture, the model achieves state-of-the-art accuracy (97%+) on the dataset.

The pipeline is designed to be robust and production-ready, featuring advanced techniques such as **RandAugment**, **Label Smoothing**, and **Mixed Precision Training (AMP)**.

## 📂 Dataset
The dataset consists of 8 categories of men's clothing, cleaned and balanced for machine learning tasks.
- **Source:** [Kaggle E-Commerce Men's Clothing Dataset](https://www.kaggle.com/datasets/prashantsharma526/e-commerce-mens-clothing-dataset)
- **Classes:** Jeans, Casual Shirts, Formal Shirts, Formal Pants, Cargos, Plain T-Shirts, Printed T-Shirts, Hoodies.

## 🚀 Key Features & Methodology

### 1. Model Architecture
We utilize **EfficientNet-B3** pretrained on ImageNet. EfficientNet was chosen for its superior parameter efficiency and accuracy trade-off compared to ResNet.

### 2. Advanced Augmentation
To prevent overfitting and improve generalization, we implement **RandAugment** (a Google research automated augmentation strategy) alongside standard transforms.

### 3. Two-Stage Training Strategy
- **Phase 1 (Warmup):** The feature extractor (backbone) is frozen. Only the custom classifier head is trained to align weights with the new classes.
- **Phase 2 (Fine-Tuning):** The entire model is unfrozen. A very low learning rate (`1e-4`) and **Cosine Annealing Scheduler** are used to fine-tune the weights without destroying learned features.

### 4. Stabilization Techniques
- **Label Smoothing:** Prevents the model from becoming over-confident (e.g., predicting 1.0 probability), which helps in reducing overfitting.
- **AMP (Automatic Mixed Precision):** Accelerates training and reduces memory usage by using float16 where possible.

## 📊 Results

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **~97.2%** |
| **F1-Score (Weighted)** | **0.97** |
| **Backbone** | EfficientNet-B3 |

### Confusion Matrix Analysis
The model shows high distinctiveness between visually similar classes (e.g., *Formal Shirts* vs. *Casual Shirts*), proving the effectiveness of the fine-tuning strategy.

## 🛠️ Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/goktani/ecommerce-fashion-vision.git
cd ecommerce-fashion-vision
```
2. Install dependencies

```Bash
pip install -r requirements.txt
```
3. Run the Training Script (Ensure you have the dataset downloaded in the correct path)

```bash
# Run the notebook or script provided in /src
```
## 📈 Visualizations
The project includes a visualization module to interpret model predictions:

- Green Labels: Correct Predictions (with confidence score).
- Red Labels: Incorrect Predictions (showing Ground Truth vs. Prediction).

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
MIT
