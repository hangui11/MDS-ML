# Project Notebooks README

This repository contains several Jupyter notebooks for graph-based machine learning tasks, including feature engineering, model training, hyperparameter tuning, and evaluation. Below is a description of each notebook and instructions for running them.

---

## 1. Hypertuning.ipynb

**Purpose:**  
This notebook performs hyperparameter tuning for a Random Forest classifier on graph-based features. It includes data preparation, feature extraction, normalization, resampling for class imbalance, cross-validation, hyperparametr tuning, and model evaluation.

**Key Steps:**
- Loads and processes training and test data.
- Extracts graph centrality features for each node.
- Normalizes features per sentence/graph.
- Resamples the training set using SMOTEENN to address class imbalance.
- Performs randomized hyperparameter search with cross-validation.
- Trains the final model with the best parameters.
- Evaluates the model and generates predictions for the test set.
- Outputs a submission file in CSV format.

**How to Run:**
1. Ensure you have the required data files (`train-random.csv`, `test-random.csv`) in the same directory.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn networkx joblib
   ```
3. Open the notebook and run all cells sequentially.

---

## 2. ModelComparison.ipynb

**Purpose:**  
Trains and evaluates the classifiers proposed. This notebook is useful for baseline comparisons and understanding feature importance in a more interpretable model for to be after hypertuned.

**Key Steps:**
- Loads and processes the data.
- Extracts and normalizes features.
- Trains several classifiers.
- Evaluates performance on validation data.
- Analyzes feature importances.

**How to Run:**
1. Ensure data files are present.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn networkx
   ```
3. Open the notebook and run all cells.

---

## 3. PCA.ipynb

**Purpose:**  
Performs Principal Component Analysis (PCA) on the features to reduce dimensionality and visualize the data. Useful for exploratory data analysis and understanding feature distributions.

**Key Steps:**
- Loads and processes the data.
- Applies PCA to the feature set.
- Visualizes explained variance and principal components.
- Optionally, uses reduced features for downstream modeling.

**How to Run:**
1. Ensure data files are present.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open the notebook and run all cells.

---

## 4. CreateGraphs.ipynb

**Purpose:**  
This notebook is foundational for visualizing the performance between the models and it generates graph structures from this data. 

**Key Steps:**
- Loads model performance data.
- visualizes graphs.

**How to Run:**
1. Install dependencies:
   ```bash
   pip install pandas matplotlib
   ```
2. Open the notebook and run all cells.

---

## 5. LogisticRegression.ipynb

**Purpose:**  
Analyzes Logistic Regression model predictions for the first feature selection without take account the most informative ones.

**Key Steps:**
- Loads model predictions and ground truth.
- Compares predictions to actual values.
- Highlights correct and incorrect predictions.
- Provides visualizations or explanations for model decisions.

**How to Run:**
1. Ensure you have the required data files (`train.csv`, `test.csv`) in the same directory.
2. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn
   ```
3. Open the notebook and run all cells.

---

## General Notes

- All notebooks assume the presence of the required data files in the working directory.
- Results (such as trained models and submission files) are saved in the same directory by default.
- For best results, use Python 3.8+ and the latest versions of the required libraries.

---