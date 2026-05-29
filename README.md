# Deep Learning Based Waste Classification System

## Project Overview

This project presents a **Deep Learning Based Waste Classification System** designed to automatically classify waste images into their respective categories using state-of-the-art transfer learning models. Proper waste segregation is a critical step toward improving recycling efficiency and reducing environmental impact. This system leverages deep learning techniques to identify different types of waste from images with high accuracy.

The project evaluates and compares multiple pre-trained convolutional neural network (CNN) architectures to determine the most effective model for waste classification.

---

## Author

**Name:** Athish Raj Mohan

---

## Project Objectives

The primary goals of this project are:

* Develop an automated waste image classification system.
* Apply transfer learning using pre-trained deep learning models.
* Compare the performance of multiple CNN architectures.
* Evaluate models using classification metrics and confusion matrices.
* Identify the best-performing architecture for waste classification.

---

## Models Evaluated

The following transfer learning architectures were implemented and evaluated:

* ResNet50
* ResNet101
* EfficientNetB0
* VGG16

These models were fine-tuned on the waste classification dataset and compared using standard evaluation metrics.

---

## Dataset

The dataset consists of labeled waste images belonging to multiple waste categories.

Example categories include:

* Cardboard
* Food Organics
* Glass
* Metal
* Miscellaneous Trash
* Paper
* Plastic
* Textile Trash
* Vegetation

The dataset is provided as a compressed file (`data.zip`) and is automatically extracted during notebook execution.

---

## Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Machine Learning Utilities

* Scikit-learn

### Development Environment

* Google Colab
* Jupyter Notebook

---

## Repository Structure

```text
Deep-Learning-Based-Waste-Classification-System/
│
├── Deep_Learning_Based_Waste_Classification_System.ipynb
├── deep_learning_based_waste_classification_system.py
├── requirements.txt
└── README.md
```

### File Description

| File                                                  | Description                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------- |
| Deep_Learning_Based_Waste_Classification_System.ipynb | Complete project notebook with training, evaluation, and visualizations |
| deep_learning_based_waste_classification_system.py    | Python implementation of the project                                    |
| requirements.txt                                      | Required Python dependencies                                            |
| README.md                                             | Project documentation                                                   |

---

## Project Workflow

### 1. Data Loading

* Load waste image dataset
* Organize images by category

### 2. Data Preprocessing

* Image resizing
* Normalization
* Dataset splitting

### 3. Data Augmentation

* Rotation
* Flipping
* Zooming
* Translation

### 4. Transfer Learning

The following pre-trained architectures were fine-tuned:

* ResNet50
* ResNet101
* EfficientNetB0
* VGG16

### 5. Model Training

* GPU-accelerated training
* Hyperparameter tuning
* Early stopping and optimization

### 6. Model Evaluation

Performance evaluation using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

---

## Results

The project successfully trained and evaluated multiple transfer learning models for waste classification.

### Best Performing Model

**EfficientNetB0** demonstrated the strongest overall performance among the evaluated architectures.

### Sample Classification Performance

| Waste Category      | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Cardboard           | 0.854     | 0.986  | 0.916    |
| Food Organics       | 0.982     | 0.988  | 0.985    |
| Glass               | 0.944     | 0.955  | 0.950    |
| Metal               | 0.955     | 0.934  | 0.944    |
| Miscellaneous Trash | 0.951     | 0.884  | 0.916    |

### Key Findings

* Transfer learning significantly improved classification performance.
* EfficientNetB0 achieved the best overall balance of accuracy and computational efficiency.
* Deep CNN architectures effectively distinguished between visually similar waste categories.
* Data augmentation improved model generalization and reduced overfitting.

---

## How to Run the Project

### Option 1: Google Colab (Recommended)

1. Open the provided Colab notebook.
2. Upload the dataset (`data.zip`) when prompted.
3. Enable GPU:

   * Runtime → Change Runtime Type
   * Hardware Accelerator → GPU
4. Run all notebook cells sequentially.

### Option 2: Local Execution

#### Clone Repository

```bash
git clone https://github.com/ATHISH-RAJ-MOHAN/Deep-Learning-Based-Waste-Classification-System-.git
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run Python Implementation

```bash
python deep_learning_based_waste_classification_system.py
```

---

## Documentation

Detailed project execution instructions:

https://docs.google.com/document/d/1RdCMhfUInrzaYZ7ga8s3LZppwxE4_06YvgjaY4WNk3M/edit?usp=sharing

---

## Notebook Preview Notice

GitHub may occasionally fail to render Jupyter Notebook previews directly in the browser.

If the notebook preview does not load:

* Open the notebook using the Google Colab link provided above.
* Download the notebook and run it locally using Jupyter Notebook.
* Refer to the Python implementation file (`deep_learning_based_waste_classification_system.py`) which contains the complete project implementation.

The notebook file itself remains fully accessible and executable even if GitHub's preview renderer fails.

---

## Future Improvements

* Deploy the model as a web application.
* Develop a real-time waste classification system.
* Integrate object detection for multiple waste items.
* Expand the dataset for improved robustness.
* Optimize models for edge and mobile devices.

---
