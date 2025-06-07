# GexpNet: Deep Residual CNN for Cancer Classification using Gene Expression Data

## Overview
GexpNet is a deep learning model based on residual multi-head residual blocks designed to classify cancer subtypes from high-dimensional microarray gene expression data. The project implements a full preprocessing pipeline, cross-validation framework, and model training & evaluation procedures.
The model achieves high performance on the Mendeley gene expression dataset and can be adapted to other similar datasets.

## Features
* Deep residual architecture with multi-head residual blocks
* Robust preprocessing pipeline (median imputation, quantile normalization, supervised feature selection)
* Cross-validation with k=10 folds
* Detailed performance metrics with statistical analysis
* Visualization utilities for results reporting

## Dataset
The project uses the Mendeley Cancer Gene Expression Dataset:
* [Link to dataset](https://doi.org/10.17632/sf5n64hydt.1)


## How to Run
```bash
python main_mendeley.py
```

This will:
* Load and preprocess the dataset
* Perform 10-fold cross-validation
* Train and evaluate the GexpNet model on each fold
* Report metrics and confidence intervals
* Plot boxplots for performance metrics
 
