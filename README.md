# 🫁 Lung Cancer Classification using CT Scan Images

A Deep Learning project that classifies lung cancer types from Chest CT Scan images using **TensorFlow** and **DenseNet architecture**.

This project was trained using CT scan images and performs image-based prediction for identifying different lung cancer categories.

---

## 📌 Project Overview

This project uses:

* **TensorFlow / Keras**
* **DenseNet Preprocessing**
* **Transfer Learning**
* **Python**
* **CT Scan Image Classification**

The model predicts the cancer class from a CT scan image and returns:

* Predicted Cancer Type
* Confidence Score

---

## 📂 Dataset

Dataset used:

**Chest CT-Scan Images**

🔗 https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

Dataset contains multiple CT scan image categories for lung cancer classification.

### Classes

* Adenocarcinoma
* Large Cell Carcinoma
* Squamous Cell Carcinoma
* Normal

---

## 🏗️ Project Structure

```plaintext
LungCancerImg/
│
├── LungsCancer_Main.py          # Prediction Script
├── LungsCancer_Img.ipynb        # Training Notebook
├── lung_cancer_classifier.keras # Saved Model
├── lung_cancer_train.csv        # Training Metadata
├── lung_cancer_test.csv         # Testing Metadata
├── Data/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone repository:

```bash
git clone <https://github.com/HanumeshGupta/LungsCancer_Cell_Image>
cd LungCancerImg
```

Create virtual environment:

```bash
py -3.10 -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Create `requirements.txt`

```txt
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

Install:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Prediction

Update image path:

```python
img_path = r"your_image_path.png"
```

Run:

```bash
python LungsCancer_Main.py
```

Example Output:

```plaintext
Prediction: adenocarcinoma
Confidence: 96.83%
```

---

## 🧠 Model Pipeline

1. Load Dataset
2. Encode Labels
3. Preprocess CT Images
4. Train Deep Learning Model
5. Save `.keras` Model
6. Load Model
7. Predict New Image

---

## 📊 Future Improvements

* Streamlit Web App
* Model Explainability (Grad-CAM)
* Upload CT Scan Interface
* Cloud Deployment
* Better Evaluation Metrics

---

