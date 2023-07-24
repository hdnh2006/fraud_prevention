# Models

This folder contains scripts for training machine learning models for fraud detection. The models include an Autoencoder and Tree-Based Models such as Random Forest and XGBoost. These models are trained on a transaction dataset where the target variable is a binary feature indicating whether a transaction is fraudulent or not.

# Scripts
The repository consists of the following main scripts:

## 1. tree_based_models.py

This script trains tree-based models, specifically a Random Forest classifier and an XGBoost model. The script also uses SMOTE to address class imbalance issues. The XGBoost model is retrained using SMOTE data and the best hyperparameters for the model are optimized. The script also provides an explainability analysis using SHAP.

Command-line arguments can be used to customize the behavior of the script. For example, the path to the input data, the name of the Weights & Biases project, the path to save plots, and a flag to avoid hyperparameter optimization can all be specified as command-line arguments.

Usage:

```bash
python tree_based_models.py --data-processed <path_to_data> --wandb-project <wandb_project_name> --out-plots <plots_output_path> --avoid-optimize
```

- `data-processed`: Specifies the path to the preprocessed data CSV file. The default value is `data/processed/data_processed.csv`.
- `wandb-project`: Specifies the name of the project in Weights & Biases. The default value is `tree-based-models`.
- `out-plots`: Specifies the path to save plots. The default value is `reports/eda`.
- `avoid-optimize`: If included, this flag indicates that the script should not perform hyperparameter optimization. Hyperparameter optimization can take a long time, even on a GPU.


## 2. autoencoder_model.py
This script is used to train an autoencoder model on the provided transaction data. The purpose of an autoencoder is to learn a compressed, distributed representation of the dataset, often for the purposes of anomaly detection or dimensionality reduction. The script also applies the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance issues in the training data.

Command-line arguments can be used to customize the behavior of the script. For example, the path to the input data, the name of the Weights & Biases project, the path to save the trained model, the batch size for training, the number of training epochs, and the patience for early stopping can all be specified as command-line arguments.

Usage: 

```bash
python autoencoder_model.py --data-processed <path_to_data> --wandb-project <wandb_project_name> --name <experiment_name> --output-model <model_output_path> --batch-size <batch_size> --epochs <number_of_epochs> --patience <early_stopping_patience>
```

Where:

- `data-processed`: Specifies the path to the preprocessed data CSV file. The default value is `data/processed/data_processed.csv`.
- `wandb-project`: Specifies the name of the project in Weights & Biases. The default value is `autoencoder`.
- `name`: Specifies the name of the experiment in Weights & Biases.
- `output-model`: Specifies the path to save the trained model. The default value is `models/pretrained/autoencoder`.
- `batch-size`: Specifies the total batch size for all GPUs. The default value is 4096.
- `epochs`: Specifies the total number of training epochs. The default value is 100.
- `patience`: Specifies the EarlyStopping patience (number of epochs without improvement). The default value is 20.














