# Hybrid LSTM-Autoencoder Model for Wheat Price Prediction and Anomaly Detection

## Overview

This repository contains the complete implementation of a hybrid deep learning ensemble framework for wheat price prediction and anomaly detection in the Turkish wheat market. The system combines four regression models (Linear Regression, Random Forest, Support Vector Regression, and LSTM+Attention) with an Autoencoder-based anomaly detection module to provide accurate price predictions and identify anomalous market conditions.

**Study Period:** June 1, 2022 - May 4, 2023 (239 trading days)
**Dataset Size:** 38,019 records
**Test Set Performance (Ablation):** R² = 0.6563, MAE = 0.85 TL

---

## Key Updates (Based on Reviewer Feedback)

### 1. Ablation Analysis

An ablation analysis was conducted to address the high R² values reported in the original study. The analysis revealed that the price-derived features (Price-Quality Ratio, Price Trend, etc.) were highly correlated with the target variable. After removing these features, the model performance is as follows:

| Model | R² (Without Price-Derived Features) | MAE (TL) |
|---|---|---|
| Linear Regression | 0.7259 | 0.82 |
| Random Forest | 0.7049 | 0.94 |
| SVR | 0.0356 | 1.90 |
| LSTM+Attention | 0.1086 | 1.16 |
| **Ensemble** | **0.6563** | **0.85** |

### 2. Visualization Updates

All figures (Figures 1-4) have been updated to improve readability:

- Font sizes increased by 20-30%
- Line widths and marker sizes enlarged
- All figures saved at 600 DPI to meet MDPI requirements

The updated figure generation codes are available in the `11_VISUALIZATIONS` directory.

---

## Repository Structure

```
WHEAT_ANALYSIS_MEGA_PACKAGE/
│
├── 01_ORIGINAL_DATA/
│   ├── raw_data.csv
│   └── processed_data.csv
│
├── 02_MODELS/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── svr.pkl
│   ├── lstm_attention.h5
│   └── autoencoder.h5
│
├── 03_SCALERS_AND_ENCODERS/
│   ├── scaler_X_min.npy
│   ├── scaler_X_scale.npy
│   ├── scaler_y_min.npy
│   ├── scaler_y_scale.npy
│   └── anomaly_threshold.npy
│
├── 04_TRAINING_DATA/
│   ├── X_train_val.npy
│   ├── y_train_val.npy
│   ├── X_val.npy
│   └── y_val.npy
│
├── 05_TEST_DATA/
│   ├── X_test_holdout.npy
│   └── y_test_holdout.npy
│
├── 06_PREDICTIONS/
│   ├── y_pred_ensemble.npy
│   └── [additional prediction files]
│
├── 07_ANOMALY_DETECTION/
│   ├── test_mse.npy
│   └── y_test_anomaly.npy
│
├── 08_ANALYSIS_RESULTS/
│   ├── all_models_comparison.csv
│   ├── ablation_results.csv
│   └── [additional analysis files]
│
├── 09_LOGS/
│   ├── data_preprocessing.log
│   └── [additional log files]
│
├── 10_PRODUCTION_READY/
│   ├── predict.py
│   └── [additional production files]
│
├── 11_VISUALIZATIONS/
│   ├── Figure_1_DataSet.py
│   ├── Figure_2_Model_Architecture_Code.py
│   ├── Figure_3.py
│   ├── Figure_4.py
│   └── MDPI_VISUALIZATIONS_UPDATED/
│       ├── Figure_1_Dataset_Updated.py
│       ├── Figure_2_Model_Architecture_Updated.py
│       ├── Figure_3_Updated.py
│       └── Figure_4_Updated.py
│
├── 12-ablation_analysis_package/
│   ├── ablation_results.csv
│   ├── ablation_study.py
│   └── ablation_study_v2.py
│
└── PSEUDOCODE.docx
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yeldafrt/Hybrid-LSTM-Autoencoder-Model-for-Wheat-Prices.git
cd Hybrid-LSTM-Autoencoder-Model-for-Wheat-Prices

# Install dependencies
pip install numpy pandas scikit-learn tensorflow

# Python version requirement
python >= 3.11
```

### Usage

```python
from production_ready.predict import WheatPricePredictor
import numpy as np

# Initialize predictor
predictor = WheatPricePredictor(model_dir=\'02_MODELS\' )

# Load your data (31 features)
X = np.load(\'your_data.npy\')

# Make predictions with anomaly detection
results = predictor.predict_with_anomaly_detection(X)

# View results
print(results.head())
```

---

## Dataset Download

Due to GitHub file size limitations, the complete dataset and model files have been split into three parts. Please download all three parts and extract them in the same directory to reconstruct the complete package.

### Download Instructions

**Part I:** WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PART_I.zip
- Contains: Original data, training/test data, scalers, encoders, and documentation

**Part II:** WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PART_II.zip
- Contains: All trained models

**Part III:** WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PART_III.zip
- Contains: Data, predictions, analysis, logs, and documentation

### Extraction Steps

```bash
# Example extraction (Linux/macOS)
unzip WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PART_I.zip
unzip WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PART_II.zip -d WHEAT_ANALYSIS_MEGA_PACKAGE/
unzip WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PART_III.zip -d WHEAT_ANALYSIS_MEGA_PACKAGE/
```
