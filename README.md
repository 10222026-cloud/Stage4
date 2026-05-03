# Rare Event Detection for Industrial Sensor Data

## Overview

This project investigates the detection of rare fault events in industrial sensor systems using machine learning techniques, with a particular focus on addressing severe class imbalance in real-world data. The system is designed to improve the reliability and robustness of fault detection in industrial environments.

## Objectives

* Detect rare fault events from sensor data
* Handle class imbalance effectively
* Improve model performance using advanced techniques

## System Pipeline

The system follows these main steps:

1. Data Acquisition
2. Preprocessing (filtering, normalization)
3. Feature Extraction
4. Imbalance Handling (SMOTE, Class Weighting, Focal Loss)
5. Model Training
6. Evaluation (Precision, Recall, F1-score, ROC-AUC)
7. Threshold Tuning

## Project Structure

```
data/       -> sample dataset (CSV)  
src/        -> source code  
README.md   -> project documentation  
```

## Dataset

A sample dataset is included in CSV format with the following structure:

| time | vibration | label |
| ---- | --------- | ----- |
| 0.1  | 0.23      | 0     |
| 0.3  | 1.20      | 1     |

* label = 0 → normal
* label = 1 → fault (rare event)

## Methods Used

* SMOTE (Synthetic Minority Oversampling Technique)
* Cost-sensitive learning
* Focal Loss
* Threshold tuning

## How to Run

### Python

```bash
python src/main.py
```

### MATLAB

```matlab
run('src/main.m')
```

## Tools

* Python (Pandas)
* MATLAB
* GitHub

## Author

Nguyen Dao Phuoc
Electrical and Computer Engineering (ECE)
Vietnamese-German University (VGU)
