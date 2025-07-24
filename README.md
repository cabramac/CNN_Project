# 🧠 CIFAR-10 Image Classification using CNN & Transfer Learning

This project builds an image classification model using Convolutional Neural Networks (CNN) on the CIFAR-10 dataset. It covers an end-to-end machine learning pipeline — from data preprocessing to evaluation and transfer learning.

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Model Architecture](#model-architecture)  
5. [Model Training](#model-training)  
6. [Model Evaluation](#model-evaluation)  
7. [Transfer Learning](#transfer-learning)  
8. [Conclusion & Learnings](#conclusion--learnings)
## 📌 Project Overview

This project trains a deep learning model to classify images from 10 categories (airplane, cat, dog, etc.) using the CIFAR-10 dataset. It uses both a custom CNN and transfer learning for performance comparison.

---

## 📦 Dataset

- **CIFAR-10**: 60,000 32×32 color images across 10 classes.
- 50,000 training images, 10,000 test images.
- Preprocessed and loaded manually from batch files.

---

## 🧹 Data Preprocessing

- Unpacked raw `.pkl` files using `pickle`
- Reshaped images from flat vectors to (32, 32, 3)
- Normalized pixel values to range [0, 1]
- Visualized sample images with labels

---

## 🧠 Model Architecture

- Built a custom CNN with:
  - 3 Convolutional + Pooling layers
  - Flatten → Dense (128) → Dropout → Output (10-class softmax)
- Used ReLU activations and max pooling

---

## 🏋️ Model Training

- Compiled with Adam optimizer and categorical crossentropy
- Trained on 50,000 images for 10 epochs
- Used validation data to monitor generalization
- Tracked loss and accuracy over epochs

---

## 📈 Model Evaluation

- Evaluated on unseen test set
- Reported:
  - Test Accuracy & Loss
  - Precision, Recall, F1-Score (per class)
  - Confusion Matrix

---

## 🚀 Transfer Learning

- Loaded a pre-trained model, utsing VGG16)
- Replaced top layers with CIFAR-10-specific classifier
- Fine-tuned on the dataset for faster, smarter learning

---

## ✅ Conclusion & Learnings

- Built an end-to-end AI pipeline
- Achieved 52% accuracy — 5× better than random
- Identified areas for improvement (augmentation, tuning)
- Gained practical experience with CNNs & transfer learning

---

## 🛠️ Dependencies

- Python 3.8+
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn
