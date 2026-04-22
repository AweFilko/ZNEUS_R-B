Semestral project for ZNEUS course on FIIT STU


**Topic 1: Magical telescope binary classification**

**Topic 2: Sports classification**


**Purpose:** Serves as log for saving our progress and way for instructors to see changes and contribution


owners:
- Anastasia Rerikh
- Filip Brutovský

## MAGIC gamma telescope binary classification

## Overview
This project focuses on the development of a neural network architecture for binary classification of high-energy particle events recorded by the MAGIC (Major Atmospheric Gamma Imaging Cherenkov) telescope.

The objective is to distinguish between gamma particles (signal) and hadron events (background noise). This classification task is important because incorrect classification of gamma events may lead to the loss of scientifically valuable observations.
Project implements a complete machine learning pipeline including:

- data preprocessing and cleaning  
- feature selection  
- neural network design  
- model training and evaluation  

Multiple neural network architectures were tested and compared to determine the most effective configuration.


## Dataset

- Source: UCI Machine Learning Repository  
- Dataset: MAGIC Gamma Telescope  
- Number of instances: 19,020  
- Number of features: 10 numerical attributes  
- Target variable: binary classification  

Class distribution:
- Gamma: 12,332 instances  
- Hadron: 6,688 instances  


## Data preprocessing

### Initial data exploration

The dataset was analyzed to identify potential issues:

- 115 duplicate records  
- no missing values  
- class imbalance (~65% gamma, ~35% hadron)  
- 7011 outliers detected  

Examples of anomalies:
- invalid geometric values (e.g. negative or zero width/length)  
- invalid angle ranges  
- extreme distance values  


### Data cleaning

Outlier detection was performed using the Interquartile Range (IQR) method:

- values outside the interval  
  [Q1 − 1.5 × IQR, Q3 + 1.5 × IQR] were considered outliers  
- outliers were replaced with NaN and imputed using the median  


### Normalization and scaling

Normalization was applied to features with skewed distributions:

- fLength, fWidth, fSize, fAlpha, fAsym, fM3Long, fM3Trans  

Features with acceptable distributions were left unchanged:

- fConc, fConc1, fDist  


### Target encoding

- gamma → 1  
- hadron → 0  


## Feature selection

The goal of feature selection was to identify the most informative variables for prediction.

### Methods used

- correlation analysis  
- forward selection  
- backward elimination  

### Results

Forward selection identified 9 important features:  
fAlpha, fSize, fWidth, fLength, fConc, fM3Long, fDist, fConc1, fAsym  

Backward elimination resulted in a reduced set of 7 features:  
fLength, fWidth, fSize, fConc, fM3Long, fAlpha, fDist  

Key observation:  
fAlpha was consistently the most important feature.


## Neural network architectures

Several neural network configurations were tested:

### MagicalNet 1
- two hidden layers  
- activation functions: Tanh and ReLU  

### MagicalNet 2
- batch normalization  
- ReLU activation  
- dropout regularization  

### MagicalNet 3
- three hidden layers  
- Leaky ReLU activation  
- improved performance but prone to overfitting  

### MagicalNet 4 (Best Model)
- optimized architecture  
- Leaky ReLU activation  
- batch normalization  
- dropout regularization  


## Training configuration

- train/test split: 80% / 20%  
- batch size: 256 and 512  
- optimizer: AdamW  
- learning rate: 1e-3  
- loss function: BCEWithLogitsLoss (pos_weight = 1.8)  
- epochs: 500  
- early stopping enabled  


## Experiments and observations

### Activation functions

- Tanh led to saturation and poor performance  
- ReLU caused loss of information for negative values  
- Leaky ReLU provided the best results  

### Regularization techniques

- dropout (0.2) improved generalization  
- higher dropout (0.3) reduced performance  
- batch normalization stabilized training  

### Loss Function tuning

- pos_weight helped address class imbalance  
- optimal value found: 1.8  

### Training techniques

- early stopping effectively prevented overfitting  
- learning rate scheduling did not significantly improve results  


## Results and evaluation

### Best model: MagicalNet 4

Final performance metrics:

- Accuracy: 87.5%  
- Precision: 85.61%  
- Recall: 96.43%  
- F1-score: 91.35%  

### Confusion matrix interpretation

- False positives are less critical (wasted telescope time)  
- False negatives are more critical (missed gamma events)  

The model prioritizes high recall to minimize missed detections.

---

# Sports classification (Deep Learning)

Multi-class image classification system for sports recognition using deep learning and feature engineering techniques. The project compares multiple neural network architectures and evaluates their performance on real-world image data.

## Overview

This project implements and evaluates several approaches for classifying sports activities from images. It combines:

* Convolutional Neural Networks (CNNs)
* Feature engineering with traditional computer vision methods
* Systematic experimentation and model comparison

The goal is to identify the most effective architecture for image classification in terms of accuracy, stability, and computational efficiency.

## Dataset

* Total images: 2077
* Number of classes: 14
* Image size: 224×224 (RGB)

The dataset contains diverse sports categories with varying visual characteristics, including:

* high intra-class variability (different angles, lighting, environments)
* inter-class similarity (visually similar sports)
* moderate class imbalance

## Project structure

```text
config/
  config.yaml              # configuration

data/
  train/                   # training images
  valid/                   # validation images
  test/                    # test images
  sports.csv               # dataset metadata
  feature_extracted.csv    # engineered features

src/
  main/                    # entry point
  model/                   # neural network architectures
  train/                   # training pipeline
  preprocess/              # preprocessing logic
  extraction/              # feature extraction methods
  evaluation/              # metrics and visualisation
  wandb/                   # experiment tracking

fs_outputs/
  selected features and encoders
```

## Approaches

The project compares four different model types:

### 1. CNN-based models

* SimpleCNN (SCNN) – baseline model
* DeepCNN (DCNN) – deeper architecture with batch normalization and dropout
* ResNet – residual network with skip connections

These models learn features directly from raw image data.

### 2. Feature-based model (FENN)

A hybrid approach using:

* manually extracted features
* fully connected neural network

Feature extraction methods include:

* Color histograms
* Histogram of Oriented Gradients (HOG)
* Local Binary Patterns (LBP)
* ORB descriptors

## Data processing

### Preprocessing

* Data augmentation:

  * random horizontal flip
  * rotation
  * color jitter
* Normalization using dataset statistics

### Feature engineering (FENN)

* Variance filtering to remove low-information features
* Mutual information for feature ranking
* PCA for dimensionality reduction (95% variance retained)

## Model architectures

### DeepCNN (best-performing model)

* multiple convolutional layers with batch normalization
* dropout for regularization
* global average pooling
* fully connected classification layer

### FENN

* multi-layer fully connected network
* operates on engineered feature vectors
* includes batch normalization and dropout

## Training

* Optimizer: AdamW
* Loss function: CrossEntropyLoss
* Learning rate scheduling: ReduceLROnPlateau
* Regularization:

  * dropout
  * weight decay

Two training modes:

* simple (no validation monitoring)
* full training with validation and early stopping

## Results

### Best model: DeepCNN

* Accuracy: 90.6%
* Precision: 91.3%
* Recall: 90.6%
* F1-score: 89.9%

### FENN (feature-based model)

* Accuracy: 80%
* Faster training but lower accuracy

## Key Findings

* CNN-based models outperform manual feature engineering
* DeepCNN provides the best balance of performance and efficiency
* ResNet is unnecessarily complex for this dataset
* Feature engineering can reduce computation cost but limits accuracy

Important observations:

* batch normalization stabilizes training
* dropout improves generalization
* AdamW provides more stable optimization than Adam
* adaptive learning rate scheduling improves convergence

## What this project demonstrates

* end-to-end machine learning pipeline design
* experience with deep learning (PyTorch)
* feature engineering and dimensionality reduction
* experimentation and model comparison
* handling real-world data issues (imbalance, variability, noise)

## Limitations

* relatively small dataset
* limited hyperparameter search due to compute constraints
* no deployment or inference API



