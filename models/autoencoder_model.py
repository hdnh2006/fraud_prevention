#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:18:44 2023

@author: henry
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from models_utils.train_utils import train_autoencoder, apply_SMOTE
from sklearn.preprocessing import LabelEncoder


# To get main directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Setup logging
script_name = Path(__file__).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{script_name}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def main(args):
    
    # Read data
    data = pd.read_csv(args.data_processed)
    
    # Initialize a label encoder
    le = LabelEncoder()
    
    # List of categorical columns to encode
    cat_cols = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode']
    
    # Encode each column
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])
    
    
    # Separate the features (X) and the target variable (y)
    X = data.drop(['transactionTime', 'isFraud'], axis=1)
    y = data['isFraud']
    
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check the sizes of the training set and the test set
    logging.info(f'Data splitted with Train shape: {X_train.shape}, Test_shape: {X_test.shape}')
    
    # Initialize training class
    autoencoder = train_autoencoder(X_train, y_train, X_test, y_test, project= args.wandb_project)
    
    # Dataset assigned is scaled so the model will be able to read it
    autoencoder.scale_dataset()
    
    # Train the autoencoder with pre-set architecture
    autoencoder.train(args)
    
    # Find best threshold
    autoencoder.find_best_cutoff()
    
    # Predict
    autoencoder.classify()
    
    
    # Retrain with SMOTE
    logging.info('🟢🟢🟢 Training eXtreme Gradient Boosting machine with SMOTE data')
    autoencoder.X_train, autoencoder.y_train = apply_SMOTE(X_train, y_train)
    args.epochs = 300
    
    # Dataset assigned is scaled so the model will be able to read it
    autoencoder.scale_dataset()
    
    # Train the autoencoder with pre-set architecture
    autoencoder.train(args)
    
    # Find best threshold
    autoencoder.find_best_cutoff()
    
    # Predict
    autoencoder.classify()


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on transaction and labels data. Output: report plots and data_processed")
    parser.add_argument('--data-processed', type=str, help='Path to transaction data CSV file.', default= ROOT / 'data/processed/data_processed.csv')
    parser.add_argument('--wandb-project', type=str, help='Name of the project in wandb', default = 'autoencoder')
    parser.add_argument('--name', type=str, help='Name of the experiment in wandb')
    parser.add_argument('--out-plots', type=str, help='Path save plots', default = ROOT / 'reports/eda')
    parser.add_argument('--batch-size', type=int, default=4096, help='total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--patience', type=int, default=20, help='EarlyStopping patience (epochs without improvement)')
    args = parser.parse_args()
    main(args)



