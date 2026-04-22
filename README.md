# Clustering and Time-Series Forecasting Project using Machine Learning (R)

A machine learning project covering two distinct tasks: unsupervised clustering of white wine varieties based on chemical properties, and neural network forecasting of USD/EUR exchange rates using an autoregressive approach.

---

## Project Overview

### Part 1 — White Wine K-Means Clustering
Applies K-means clustering to a dataset of white wine chemical properties. The analysis is performed in two subtasks:
- **Subtask 1:** Clustering on all 11 original features after preprocessing
- **Subtask 2:** Dimensionality reduction via PCA, followed by clustering on the transformed dataset

### Part 2 — USD/EUR Exchange Rate Forecasting
Uses a Multilayer Perceptron (MLP) neural network to perform one-step-ahead forecasting of USD/EUR exchange rates. Multiple MLP architectures are tested with varying input vectors (up to t-4 autoregressive lags) and internal structures.

---

## Datasets

| Dataset | Description | Source |
|---|---|---|
| `Whitewine_Dataset.xlsx` | 2700 white wine samples with 11 chemical properties + quality score | UCI Machine Learning Repository |
| `ExchangeUSD_Dataset.xlsx` | 500 daily USD/EUR exchange rate observations (Oct 2011 – Oct 2013) | Provided as part of coursework |

### White Wine Features
- Fixed acidity, Volatile acidity, Citric acid, Residual sugar, Chlorides
- Free sulfur dioxide, Total sulfur dioxide, Density, pH, Sulphates, Alcohol
- *Quality (not used in clustering)*

---

## Techniques Used

**Clustering (Part 1)**
- Data scaling and outlier removal (Z-score method)
- Number of cluster determination: NBclust, Elbow, Gap Statistic, Silhouette
- K-means clustering with BSS/TSS, WSS evaluation
- Silhouette plot analysis
- PCA (eigenvalues, eigenvectors, cumulative variance > 85%)
- Calinski-Harabasz Index

**Forecasting (Part 2)**
- Autoregressive (AR) input vectors with time delays up to t-4
- MLP neural networks with various hidden layer configurations
- Normalisation using training set parameters
- Evaluation metrics: RMSE, MAE, MAPE, sMAPE

---

## Repository Structure

```
wine-clustering-fx-forecasting/
│
├── datasets/
│   ├── Whitewine_Dataset.xlsx
│   └── ExchangeUSD_Dataset.xlsx
│
├── whitewine_kmeans_pca.R
├── exchangerate_mlp_forecasting.R
└── README.md
```

---

## Requirements

- R (version 4.0 or higher recommended)
- RStudio

### R Packages

Install all required packages by running:

```r
install.packages(c("readxl", "NbClust", "factoextra", "cluster", "fpc", "neuralnet", "ggplot2"))
```

---

## How to Run

1. Clone the repository:
```bash
git clone git@github.com:Minduli-Lasandi/ML-Clustering-and-Time-Series-Forecasting.git
```

2. Open RStudio and set your working directory to the project folder:
```r
setwd("path/to/wine-clustering-fx-forecasting")
```

3. Make sure both dataset files (`Whitewine_Dataset.xlsx` and `ExchangeUSD_Dataset.xlsx`) are in the dataset folder.

4. Run the clustering script:
```r
source("whitewine_kmeans_pca.R")
```

5. Run the forecasting script:
```r
source("exchangerate_mlp_forecasting.R")
```

---

## Results Summary

### Clustering
- Optimal number of clusters: **k = 2** (determined by NBclust, Elbow, Gap Statistic and Silhouette methods)
- PCA retained the first **N principal components** explaining > 85% of variance
- Calinski-Harabasz Index used as additional internal validation metric

### Forecasting
- 12 MLP models tested with varying input vectors and hidden layer structures
- Best model evaluated using RMSE, MAE, MAPE and sMAPE on the test set (samples 401–500)
- Results visualised as a scatter plot of predicted vs. desired output

---

## Notes

- All test set normalisation is performed using training set parameters (mean and SD) to prevent data leakage
- `linear.output = TRUE` is used in all MLP models as this is a regression task
