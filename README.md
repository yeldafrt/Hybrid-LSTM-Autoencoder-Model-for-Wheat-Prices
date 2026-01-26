# Hybrid LSTM-Autoencoder Model for Wheat Price Prediction and Anomaly Detection

## Overview

This repository contains the complete implementation of a hybrid deep learning ensemble framework for wheat price prediction and anomaly detection in the Turkish wheat market. The system combines four regression models (Linear Regression, Random Forest, Support Vector Regression, and LSTM+Attention) with an Autoencoder-based anomaly detection module to provide accurate price forecasts and identify anomalous market conditions.

| Metric | Value |
|--------|-------|
| **Study Period** | June 1, 2022 - May 4, 2023 (239 trading days) |
| **Dataset Size** | 38,019 records |
| **Test Set Performance** | R² = 0.9942, MAE = 0.1646 TL |

---

## Repository Structure

```
WHEAT_ANALYSIS_MEGA_PACKAGE/
│
├── 01_ORIGINAL_DATA/
│   ├── raw_data.csv                 # Original wheat price dataset (38,019 records)
│   ├── raw_data1.xlsx               # Alternative raw data format
│   └── processed_data.csv           # Preprocessed dataset with engineered features
│
├── 02_MODELS/
│   ├── linear_regression.pkl        # Trained Linear Regression model
│   ├── random_forest.pkl            # Trained Random Forest model (100 trees)
│   ├── svr.pkl                      # Trained SVR model (RBF kernel, C=100, ε=0.05)
│   ├── lstm_attention.h5            # Trained LSTM+Attention model (64-32 units)
│   ├── autoencoder.h5               # Trained Autoencoder for anomaly detection
│   └── encoder.h5                   # Encoder component of Autoencoder
│
├── 03_SCALERS_AND_ENCODERS/
│   ├── scaler_X_min.npy             # Feature normalization minimum value
│   ├── scaler_X_scale.npy           # Feature normalization scale value
│   ├── scaler_y_min.npy             # Price normalization minimum value
│   ├── scaler_y_scale.npy           # Price normalization scale value
│   ├── anomaly_threshold.npy        # Anomaly detection threshold (95th percentile)
│   ├── sinif_encoder.npy            # Category encoder for classification
│   └── urun_encoder.npy             # Product encoder
│
├── 04_TRAINING_DATA/
│   ├── X_train_val.npy              # Training features (22,811 samples)
│   ├── y_train_val.npy              # Training prices
│   ├── X_val.npy                    # Validation features (7,604 samples)
│   └── y_val.npy                    # Validation prices
│
├── 05_TEST_DATA/
│   ├── X_test_holdout.npy           # Holdout test features (7,604 samples)
│   └── y_test_holdout.npy           # Holdout test prices
│
├── 06_PREDICTIONS/
│   ├── y_pred_ensemble.npy          # Ensemble model predictions
│   ├── y_pred_test_lr.npy           # Linear Regression predictions
│   ├── y_pred_test_rf.npy           # Random Forest predictions
│   └── y_pred_test_lstm.npy         # LSTM predictions
│
├── 07_ANOMALY_DETECTION/
│   ├── test_mse.npy                 # Reconstruction error (MSE) for test set
│   ├── test_mse_external.npy        # Reconstruction error for external test set
│   ├── y_test_anomaly.npy           # Anomaly labels for test set
│   ├── y_test_anomaly_external.npy  # Anomaly labels for external test set
│   └── X_test_pred_external.npy     # Autoencoder predictions on external data
│
├── 08_ANALYSIS_RESULTS/
│   ├── all_models_comparison.csv    # Performance comparison of all 5 models
│   ├── external_data_performance.csv # Holdout test set performance metrics
│   ├── feature_importance_shap.csv  # SHAP-based feature importance analysis
│   ├── feature_importance_rf.csv    # Random Forest feature importance
│   ├── cv_linear_regression.csv     # 5-fold CV results for Linear Regression
│   ├── cv_random_forest.csv         # 5-fold CV results for Random Forest
│   ├── cv_svr.csv                   # 5-fold CV results for SVR
│   ├── cv_overfitting_gap.csv       # Overfitting analysis across folds
│   ├── anomaly_detailed_analysis.csv # Detailed anomaly detection results
│   ├── data_split_summary.csv       # Data split statistics
│   └── [additional analysis files]
│
├── 09_LOGS/
│   ├── data_preprocessing.log       # Data preprocessing execution log
│   ├── data_split.log               # Data splitting log
│   ├── baseline_models.log          # Baseline model training log
│   ├── lstm_attention.log           # LSTM model training log
│   ├── autoencoder.log              # Autoencoder training log
│   ├── autoencoder_external_test.log # External test evaluation log
│   └── ensemble.log                 # Ensemble model training log
│
├── 10_PRODUCTION_READY/
│   ├── predict.py                   # Production prediction script
│   ├── README.md                    # Production usage documentation
│   ├── linear_regression.pkl        # Production Linear Regression model
│   ├── random_forest.pkl            # Production Random Forest model
│   ├── svr.pkl                      # Production SVR model
│   ├── lstm_attention.h5            # Production LSTM model
│   ├── autoencoder.h5               # Production Autoencoder model
│   ├── encoder.h5                   # Production Encoder model
│   ├── scaler_*.npy                 # Normalization parameters
│   ├── anomaly_threshold.npy        # Anomaly threshold
│   ├── sinif_encoder.npy            # Category encoder
│   └── urun_encoder.npy             # Product encoder
│
├── 11_VISUALIZATIONS/
│   ├── Figure_1_DataSet.py          # Dataset overview and statistics
│   ├── Figure_2_Model_Architecture_Code.py # Model architecture visualization
│   ├── Figure_3.py                  # Model performance and interpretability
│   ├── Figure_4.py                  # Robustness and anomaly detection
│   └── MDPI_VISUALIZATIONS_UPDATED/ # Updated figures for MDPI (600 DPI, large fonts)
│       ├── Figure_1_LARGE_FONTS.py  # Dataset figure with large fonts
│       ├── Figure_2_LARGE_FONTS.py  # Architecture figure with large fonts
│       ├── Figure_3_LARGE_FONTS.py  # Performance figure with large fonts
│       └── Figure_4_LARGE_FONTS.py  # Robustness figure with large fonts
│
├── 12_ablation_analysis_package/    # Ablation analysis scripts and results
│   ├── ablation_results.csv         # Ablation analysis results (Table 4)
│   ├── ablation_study.py            # Ablation study script (v1)
│   └── ablation_study_v2.py         # Ablation study script (v2 - final)
│
├── 13_HYPERPARAMETER_TUNING/        # Hyperparameter tuning analysis
│   └── 13_HYPERPARAMETER_TUNING_v2/
│       ├── hyperparameter_tuning.py         # Hyperparameter tuning script
│       └── hyperparameter_tuning_results.csv # Tuning results (Table 5)
│
└── PSEUDOCODE.docx                  # Complete pseudocode documentation
```

---

## Dataset Description

| Metric | Value |
|--------|-------|
| **Total Records** | 38,019 |
| **Time Period** | June 1, 2022 - May 4, 2023 |
| **Training Set** | 22,811 samples (60%) |
| **Validation Set** | 7,604 samples (20%) |
| **Test Set (Holdout)** | 7,604 samples (20%) |

### Features (31 input features)

**Quality Parameters:**
- Moisture (Rutubet)
- Hectolitre Weight (Hektolitre)
- Protein Content (Protein)
- Defective Grains (Kusurlu Taneler)
- Broken Grains (Kırık Tane)
- Shriveled Grains (Çiliz Burışuk)
- Foreign Material (Yabancı Madde)
- Husk (Kavuz)
- And additional quality metrics

**Engineered Features (8 derived variables):**
1. **Price-Quality Ratio:** Unit price / Quality score
2. **Quality Score:** Weighted composite index (moisture, hectolitre, protein), normalized [0,1]
3. **Price Trend:** Current price − 7-day moving average
4. **Total Defect Ratio:** Sum of all defect percentages
5. **Seasonal Indicator:** Binary (1 if June-September, 0 otherwise)
6. **Week Number:** ISO week number from transaction date
7. **Price Volatility:** Rolling standard deviation (7 days)
8. **7-Day Price MA:** Arithmetic mean of prices over 7 days

**Target Variable:** Unit Price (BirimFiyati) in Turkish Lira (TL)

---

## Model Architecture

### Ensemble Framework

The system combines four regression models with optimized weights determined by validation R² performance:

| Model | Weight | Test R² | Test MAE (TL) | Test RMSE (TL) | MAPE (%) |
|-------|--------|---------|---------------|----------------|----------|
| Linear Regression | 0.255 | 1.0000 | 0.0000 | 0.0000 | 0.00 |
| Random Forest | 0.255 | 0.9995 | 0.0103 | 0.0571 | 0.44 |
| SVR | 0.237 | 0.9297 | 0.5844 | 0.7024 | 16.42 |
| LSTM+Attention | 0.253 | 0.9912 | 0.1782 | 0.2479 | 5.89 |
| **Ensemble** | **-** | **0.9942** | **0.1646** | **0.2016** | **5.16** |

### Model Hyperparameters

| Model | Hyperparameters |
|-------|-----------------|
| **LSTM+Attention** | 64 units (Layer 1), 32 units (Layer 2), lr=0.001, batch=64, epochs=100, early_stop=10 |
| **Random Forest** | 100 trees, max_depth=None |
| **SVR** | RBF kernel, C=100, ε=0.05 |
| **Autoencoder** | Architecture: 31→16→8→4→8→16→31, threshold=95th percentile |

### Anomaly Detection

| Metric | Value |
|--------|-------|
| **Method** | Autoencoder with reconstruction error |
| **Threshold** | 95th percentile of training reconstruction errors |
| **Detection Rate** | 390 anomalies (5.13%) in test set |
| **Normal Records** | 7,214 (94.87%) |

---

## Model Performance

### Holdout Test Set Results (n=7,604)

**Regression Performance:**
- Ensemble R² Score: 0.9942
- Mean Absolute Error (MAE): 0.1646 TL
- Root Mean Squared Error (RMSE): 0.2016 TL
- Mean Absolute Percentage Error (MAPE): 5.16%

**Cross-Validation Stability (5-Fold):**

| Model | Mean Test R² | Std |
|-------|--------------|-----|
| Linear Regression | 1.0000 | 0.0000 |
| Random Forest | 0.9972 | 0.0028 |
| SVR | 0.9161 | 0.0078 |

### Hyperparameter Analysis (Table 5)

| Model | Hyperparameter Setting | Validation R² | Validation MAE (TL) |
|-------|------------------------|---------------|---------------------|
| LSTM + Attention | 32 units, lr=0.001 | 1.0000 | 0.024 |
| **LSTM + Attention** | **64 units, lr=0.001** | **1.0000** | **0.018** |
| LSTM + Attention | 128 units, lr=0.0005 | 1.0000 | 0.019 |
| Random Forest | 50 trees, depth=None | 0.9994 | 0.011 |
| **Random Forest** | **100 trees, depth=None** | **0.9994** | **0.011** |
| Random Forest | 200 trees, depth=20 | 0.9994 | 0.011 |
| SVR | RBF kernel, C=1 | 0.7901 | 0.931 |
| SVR | RBF kernel, C=10 | 0.7901 | 0.931 |
| **SVR** | **RBF kernel, C=100, ε=0.05** | **0.9174** | **0.596** |
| SVR | Polynomial kernel, C=10 | 0.7622 | 0.958 |

### Ablation Analysis (Table 4)

Performance comparison with and without price-derived features:

| Model | R² (All Features) | R² (Without Price Features) | Difference |
|-------|-------------------|----------------------------|------------|
| Linear Regression | 1.0000 | 0.7259 | -0.2741 |
| Random Forest | 0.9995 | 0.7049 | -0.2946 |
| SVR | 0.9297 | 0.0356 | -0.8941 |
| LSTM + Attention | 0.9912 | 0.1086 | -0.8826 |
| Ensemble | 0.9942 | 0.6563 | -0.3379 |

---

## Feature Importance (SHAP Analysis)

Top features by SHAP importance:

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 1 | Price-Quality Ratio (Fiyat_Kalite_Orani) | 0.0343 |
| 2 | Quality Score (Kalite_Skoru) | 0.0040 |
| 3 | Price Trend (Fiyat_Trend) | 0.0002 |
| 4 | Total Defects (Toplam_Kusur) | 0.00007 |
| 5 | Other Features | <0.00006 |

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

#### 1. Simple Prediction

```python
from production_ready.predict import WheatPricePredictor
import numpy as np

# Initialize predictor
predictor = WheatPricePredictor(model_dir='10_PRODUCTION_READY')

# Load your data (31 features)
X = np.load('your_data.npy')

# Make predictions with anomaly detection
results = predictor.predict_with_anomaly_detection(X)

# View results
print(results.head())
```

#### 2. Price Prediction Only

```python
prices = predictor.predict_price(X)
print(f"Predicted prices: {prices}")
```

#### 3. Anomaly Detection Only

```python
anomalies, mse = predictor.detect_anomalies(X)
print(f"Anomalies detected: {(anomalies == 1).sum()}")
```

---

## Output Format

### Prediction Output

```
DataFrame with columns:
- Predicted_Price_TL: Forecasted price in Turkish Lira
- Is_Anomaly: Binary anomaly label (0=normal, 1=anomaly)
- Reconstruction_Error: MSE value from Autoencoder
- Anomaly_Risk: Risk level (HIGH/LOW)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Ensemble (R² = 0.9942) |
| **Anomaly Detection Rate** | 5.13% (390 records) |
| **Cross-Validation Stability** | Mean R² = 0.9972 |
| **Feature Dominance** | Price-Quality Ratio (SHAP = 0.0343) |
| **Generalization** | No overfitting detected (Train R² ≈ Test R²) |

---

## Dataset Download

Due to GitHub file size limitations, the complete dataset and model files have been split into three parts. Please download all three parts and extract them to reconstruct the complete package.

### Download Instructions

| Part | Contents | Size |
|------|----------|------|
| **Part I** | Original data, training/test data, scalers, encoders, ablation analysis, PSEUDOCODE.docx | ~11 MB |
| **Part II** | All trained models (Linear Regression, Random Forest, SVR, LSTM, Autoencoder) | ~20 MB |
| **Part III** | Predictions, analysis results, logs, visualizations, hyperparameter tuning | ~21 MB |

**Part I Contents:**
- `01_ORIGINAL_DATA/`: raw_data.csv, raw_data1.xlsx, processed_data.csv
- `03_SCALERS_AND_ENCODERS/`: All scaler and encoder files
- `04_TRAINING_DATA/`: X_train_val.npy, y_train_val.npy, X_val.npy, y_val.npy
- `05_TEST_DATA/`: X_test_holdout.npy, y_test_holdout.npy
- `12_ablation_analysis_package/`: Ablation study scripts and results
- `PSEUDOCODE.docx`: Complete algorithm documentation

**Part II Contents:**
- `02_MODELS/`: All trained model files (.pkl, .h5)

**Part III Contents:**
- `06_PREDICTIONS/`: Model prediction outputs
- `07_ANOMALY_DETECTION/`: Anomaly detection results
- `08_ANALYSIS_RESULTS/`: All CSV analysis files
- `09_LOGS/`: Training and evaluation logs
- `10_PRODUCTION_READY/`: Production deployment package
- `11_VISUALIZATIONS/`: Figure generation scripts (including MDPI large fonts)
- `13_HYPERPARAMETER_TUNING/`: Hyperparameter analysis scripts and results

### Extraction Steps

```bash
# Download all three ZIP files from the repository

# Extract Part I (creates base directory structure)
unzip WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PARTI.zip

# Extract Part II (merge into same directory)
unzip WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PARTII.zip

# Extract Part III (merge into same directory)
unzip WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PARTIII.zip

# Rename directory if needed
mv "WHEAT_ANALYSIS_COMPLETE_MEGA_PACKAGE_PART I" WHEAT_ANALYSIS_MEGA_PACKAGE
```

### Total Package Size

| Metric | Value |
|--------|-------|
| **Part I** | ~11 MB |
| **Part II** | ~20 MB |
| **Part III** | ~21 MB |
| **Total Download** | ~52 MB |
| **Extracted Size** | ~100 MB |

---

## Requirements

| Package | Version |
|---------|---------|
| Python | >= 3.11 |
| NumPy | >= 1.21 |
| Pandas | >= 1.3 |
| Scikit-learn | >= 1.3.2 |
| TensorFlow | >= 2.15 |
| Keras | >= 3.0 |
| SHAP | >= 0.44.0 |

---

## Citation

If you use this code or dataset in your research, please cite:

```
@article{firat2026hybrid,
  title={A Hybrid Deep Learning Model for Wheat Price Prediction: LSTM-Autoencoder Ensemble Approach with SHAP-Based Interpretability},
  author={Fırat, Yelda and Sarıkaya, Hüseyin Ali},
  journal={Sustainability},
  year={2026},
  publisher={MDPI}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Yelda Fırat** - Department of Computer Engineering, Mudanya University
  - Email: yelda.firat@mudanya.edu.tr
  - GitHub: [@yeldafrt](https://github.com/yeldafrt)

- **Hüseyin Ali Sarıkaya** - Department of Industrial Engineering, Mudanya University
  - Email: huseyin.sarikaya@mudanya.edu.tr

---

## Acknowledgments

- Turkish Grain Board (TMO) for providing the wheat market data
- MDPI Sustainability journal reviewers for their valuable feedback
- TensorFlow and scikit-learn development teams
