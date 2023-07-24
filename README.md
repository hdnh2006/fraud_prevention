# Featurespace fraud prevention
Repo for the featurespace assigment in fraud prevention

# Models

This directory contains all the machine learning models used in the project.

## Files

* `tree_based_models.py` - This script contains the implementation of tree-based machine learning models such as Random Forest and XGBoost. It has a class `train_tree_cls` that trains Random Forest and XGBoost models on provided training data, applies Synthetic Minority Over-sampling Technique (SMOTE), finds the best parameters for the models using GridSearchCV, and also explains the models using SHAP values.

* `train_autoencoder.py` - This script is responsible for training an autoencoder neural network model. The model is defined, compiled, and trained in this script. It takes in a preprocessed dataset, trains the model, and saves the trained model.

```
.
├── EDA
│   ├── eda.py
│   └── preprocess.py
├── Models
│   ├── tree_based_models.py
│   ├── train_xgboost.py
│   └── train_autoencoder.py
├── Deployment
│   ├── app.py
│   └── Dockerfile
├── Utils
│   ├── utils.py
│   └── Dockerfile
├── Tests
│   └── test_utils.py
├── Reports
│   └── final_report.pdf
└── Data
    ├── raw
    └── processed


```