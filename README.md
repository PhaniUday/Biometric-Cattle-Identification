# 🐄 Biometric Cattle Identification via Deep Metric Learning

A lightweight, vision-based cattle identification system using deep metric learning and computer vision to enable non-invasive individual cattle recognition without relying on traditional RFID tags or physical identification markers.

## 📌 Overview

Traditional cattle identification systems often rely on physical methods such as ear tags and RFID devices, which can be inconvenient, require maintenance, and may be lost or damaged.

This project explores a vision-based alternative that identifies individual cattle using their unique muzzle patterns.

The system learns discriminative feature embeddings from cattle muzzle images and performs identity recognition through embedding similarity.

The objective is to develop an accurate, lightweight, and scalable biometric identification system suitable for real-world farm environments and resource-constrained edge devices.

## 🎯 Objectives

- Develop a non-invasive biometric cattle identification system.
- Learn discriminative representations from cattle muzzle images.
- Support both **1:1 biometric verification** and **1:N identification**.
- Improve recognition accuracy through deep metric learning.
- Design a lightweight architecture suitable for edge deployment.
- Reduce dependence on conventional RFID-based identification systems.

## 🧠 Methodology

The project follows a deep metric learning approach, where images belonging to the same cattle identity are mapped to nearby points in an embedding space, while images belonging to different identities are pushed farther apart.

### 1. Image Processing

Cattle muzzle images are used as biometric inputs.

Image preprocessing and data augmentation are employed to improve model generalization and account for variations in image appearance.

### 2. Feature Extraction

A lightweight, MobileViT-inspired CNN architecture is used as the feature extraction backbone.

The architecture incorporates **Efficient Multi-Scale Attention (EMA)** to improve feature representation by emphasizing discriminative spatial and channel information.

### 3. Deep Metric Learning

The model is trained using Triplet Margin Loss.

Triplet learning consists of:

- **Anchor:** Reference image of a cattle identity.
- **Positive:** Another image belonging to the same cattle identity.
- **Negative:** Image belonging to a different cattle identity.

The objective is to minimize the distance between anchor-positive pairs while maximizing the distance between anchor-negative pairs.

Hard/semi-hard negative selection helps improve the discriminative capability of the learned embeddings.

### 4. Embedding Generation

The trained network generates a **256-dimensional L2-normalized feature embedding** for every input image.

These embeddings represent the biometric characteristics of individual cattle.

### 5. Recognition

Recognition is performed using embedding distance comparison.

- **1:1 Verification:** Determines whether two muzzle images belong to the same cattle identity using a similarity threshold.
- **1:N Identification:** Compares a query embedding against a gallery of registered cattle embeddings and retrieves the closest identity using nearest-neighbour matching.

## 🏗️ Model Architecture

The proposed architecture combines:

- Lightweight CNN feature extraction.
- MobileViT-inspired design principles.
- EMA attention mechanism.
- Triplet-based deep metric learning.
- 256-dimensional L2-normalized embeddings.
- Distance-based biometric matching.

The architecture aims to balance recognition accuracy and computational efficiency for potential edge-device deployment.

## 📊 Results

The proposed model was evaluated using a gallery-query identification protocol.

| Evaluation Metric | Result |
|---|---:|
| Initial baseline Rank-1 Accuracy | 83.19% |
| Improved Model Rank-1 Accuracy | **96.55%** |
| Gallery identities | 24 |
| Query images | 232 |
| Correct Rank-1 identifications | 224 |
| Incorrect Rank-1 identifications | 8 |
| Embedding dimension | 256 |

### Performance Improvement

The improved architecture achieved a Rank-1 identification accuracy of **96.55%**, compared to the initial baseline of 83.19%.

This represents a substantial improvement in individual cattle recognition performance.

The evaluation also demonstrates the effectiveness of metric learning in producing discriminative biometric representations.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Computer Vision:** Image processing and data augmentation
- **Model Architecture:** Lightweight CNN, MobileViT-inspired feature extraction
- **Attention Mechanism:** EMA Attention
- **Learning Strategy:** Deep Metric Learning
- **Loss Function:** Triplet Margin Loss
- **Recognition:** Euclidean Distance and Nearest-Neighbour Retrieval

## 📁 Project Structure

```text
CattleIdentification/
│
├── data_loader.py
├── download_data.py
├── model.py
├── train.py
├── finetune.py
├── evaluate.py
├── recognition.py
├── list_files.py
│
├── best_model.pth
├── triplet_best_model.pth
│
├── evaluation_results.png
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Dataset files and large experiment outputs are excluded from this repository to keep it lightweight.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/USERNAME/CattleIdentification.git
```

```bash
cd CattleIdentification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Setup

The cattle muzzle datasets are not included in this repository because of their large storage requirements.

Download and configure the required dataset separately, following the directory structure expected by the data-loading scripts.

### 5. Training and Evaluation

The repository includes separate scripts for:

| Script | Purpose |
|---|---|
| `data_loader.py` | Dataset loading and preprocessing |
| `download_data.py` | Dataset download utilities |
| `model.py` | Model architecture |
| `train.py` | Model training |
| `finetune.py` | Model fine-tuning |
| `evaluate.py` | Model evaluation |
| `recognition.py` | Cattle identification |
| `list_files.py` | Dataset/file inspection |

## 🚀 Applications

Potential applications of the proposed system include:

- Automated cattle identification at farm entry points.
- Individual animal recognition at feeding stations.
- Livestock monitoring and management.
- Non-invasive biometric verification.
- Automated livestock identification systems using computer vision.

## 🔬 Future Scope

- Improve recognition under varying lighting and environmental conditions.
- Expand evaluation across larger cattle populations.
- Optimize inference speed and memory consumption.
- Explore deployment on edge computing hardware.
- Integrate real-time camera-based identification.
- Develop a complete farm-level cattle monitoring solution.

## 👨‍💻 Project Information

**Domain:** Artificial Intelligence, Computer Vision and Deep Metric Learning

**Focus:** Biometric Identification and Livestock Recognition

**Primary Objective:** Non-invasive cattle identification using muzzle biometrics.

---

*This project explores the application of lightweight deep learning and biometric recognition techniques to improve livestock identification through computer vision.*
