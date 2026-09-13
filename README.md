# CNN-Based-Scene-Classification-Model

A deep learning project for **indoor scene classification** into 17 different scene categories. The project explores and compares multiple convolutional and transformer-based architectures, including a custom CNN, ResNet, DenseNet121 and EfficientNet-B3.

> **Academic Project — CSE281: Introduction to Artificial Intelligence**
> Ain Shams University — Computer and AI Engineering
> May 2026

---

## 📌 Overview

The goal of this project is to classify indoor images into different scene categories such as kitchens, bedrooms, offices, and living rooms.

Indoor scene classification is challenging because different scenes can share similar visual characteristics, while the same scene category can vary significantly in:

* Lighting conditions
* Object placement
* Image resolution
* Visual noise
* Layout and textures

The project investigates how different deep learning architectures perform on this classification task.

---

## 🗂️ Dataset & Preprocessing

The dataset contains labeled images belonging to **17 scene classes**.

The data was divided using a **stratified 80/20 train-validation split** to preserve class distribution.

### Image Preprocessing

The preprocessing pipeline included:

* Image validation and corruption checking
* RGB conversion
* Resizing images to **224 × 224**
* Image normalization

### Data Augmentation

Training images were augmented using:

* Random horizontal flipping
* Random rotation up to 15°
* Color jittering for brightness and contrast
* Random cropping

The models were trained using a batch size of **32**.

---

## 🧠 Models

Six different architectures were investigated.

### 1. StyleCNN

A custom four-layer convolutional neural network designed specifically for the scene classification task.

Architecture highlights:

* Convolutional layers with 32, 64, 128, and 256 filters
* Batch Normalization
* LeakyReLU activations
* Max Pooling
* Adaptive Average Pooling
* Fully connected layer with 512 neurons
* Dropout

**Validation Accuracy: 34.03%**

---

### 2. ResNet

A pretrained ResNet architecture using residual connections to improve feature learning and enable deeper networks.

The pretrained backbone was adapted for the 17-class classification problem and fine-tuned on the dataset.

**Validation Accuracy: 23.66%**

---

### 3. DenseNet121

DenseNet121 was used with pretrained ImageNet weights.

Dense connections allow each layer to receive feature maps from previous layers, encouraging feature reuse throughout the network.

The final classification layer was adapted for the 17 scene classes.

**Validation Accuracy: 34.94%**

---

### 4. EfficientNet-B3

EfficientNet-B3 was investigated using transfer learning from ImageNet.

The training process included:

* Initial backbone freezing
* Fine-tuning of the full network
* AdamW optimizer
* Label smoothing
* Cosine annealing warm restarts
* Automatic mixed precision (AMP)

**Validation Accuracy: 20.55%**

---

## 📊 Results

| Model           | Validation Accuracy |
| --------------- | ------------------: |
| StyleCNN        |              34.03% |
| ResNet          |              23.66% |
| DenseNet121     |              34.94% |
| EfficientNet-B3 |              20.55% |


## ⚙️ Training Methodology

The general training pipeline included:

1. Loading and validating the image data
2. Stratified train-validation splitting
3. Image preprocessing and augmentation
4. Loading pretrained models where applicable
5. Initially freezing pretrained backbones
6. Fine-tuning the models
7. Training with cross-entropy loss
8. Using optimizers with weight decay
9. Applying learning-rate scheduling
10. Using early stopping
11. Saving the best-performing model

Training was performed for up to **30 epochs**, with GPU acceleration used when available.

---

## 🔍 Challenges & Limitations

The project achieved moderate overall validation performance.

Several factors affected classification performance:

* Class imbalance within the dataset
* Similar visual characteristics between different scene categories
* Overlapping textures and layouts
* Computational and memory requirements of some architectures
* Difficulty distinguishing scenes with visually similar structures

DenseNet121, for example, required considerable computational and memory resources.

---

## 🛠️ Technologies

* Python
* PyTorch
* Deep Learning
* Convolutional Neural Networks
* Transfer Learning
* Computer Vision
* Vision Transformers
* ConvNeXt
* Image Classification

---

## 📌 Project Outcome

This project provided practical experience with:

* Building and training CNN architectures
* Transfer learning with pretrained models
* Image preprocessing and augmentation
* Fine-tuning deep learning models
* Comparing different architectures experimentally
* Working with GPU-based training
* Evaluating model performance on a multi-class computer vision problem
