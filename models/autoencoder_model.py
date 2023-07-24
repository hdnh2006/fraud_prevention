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
import joblib
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
    le_dict = {}
    
    # List of categorical columns to encode
    cat_cols = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode']
    
    # Encode each column
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le  # store the encoder
    
    # Now save all encoders
    path_save = Path(args.output_model)
    path_save.mkdir(parents=True, exist_ok=True)  # make dir
    path_out = os.path.join(path_save, 'label_encoders.pkl')
    joblib.dump(le_dict, path_out)
    
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
    logging.info('Scaling dataset')
    autoencoder.scale_dataset(args)
    
    # Train the autoencoder with pre-set architecture
    logging.info('🟢🟢🟢 Training Autoencoder neural network.')
    autoencoder.train(args)
    
    # Find best threshold
    logging.info('🟢🟢🟢 Finding the best cutoff for prediction')
    autoencoder.find_best_cutoff()
    
    # Predict
    logging.info('Testing the cutoff in test dataset')
    autoencoder.classify(args)
    
    # Retrain with SMOTE
    logging.info('🟢🟢🟢 Training Autoencoder neural network with SMOTE data')
    autoencoder.X_train, autoencoder.y_train = apply_SMOTE(X_train, y_train)
    
    logging.info('Dataset bigger so the data will be passed 3 times the number of epochs initially set')
    args.epochs = args.epochs*3
    
    # Dataset assigned is scaled so the model will be able to read it
    logging.info('Scaling the new dataset with SMOTE')
    autoencoder.scale_dataset(args)
    
    # Train the autoencoder with pre-set architecture
    logging.info('🟢🟢🟢 Training the neural network with SMOTE data')
    autoencoder.train(args)
    
    # Find best threshold
    logging.info('🟢🟢🟢 Finding the best cutoff for prediction with SMOTE dataset')
    autoencoder.find_best_cutoff()
    
    # Predict X_test
    logging.info('Testing the cutoff in test dataset with model trained using SMOTE')
    autoencoder.classify(args)
    
    # Saving model
    logging.info('🟢🟢🟢 The false positives values are reduced so we will save this model')
    path_save = Path(args.output_model)
    path_save.mkdir(parents=True, exist_ok=True)  # make dir
    path_out = os.path.join(path_save, 'autoencoder.keras')
    autoencoder.autoencoder.save(path_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on transaction and labels data. Output: report plots and data_processed")
    parser.add_argument('--data-processed', type=str, help='Path to transaction data CSV file.', default= ROOT / 'data/processed/data_processed.csv')
    parser.add_argument('--wandb-project', type=str, help='Name of the project in wandb', default = 'autoencoder')
    parser.add_argument('--name', type=str, help='Name of the experiment in wandb')
    parser.add_argument('--output-model', type=str, help='Path save the model', default = ROOT / 'models/pretrained/autoencoder')
    parser.add_argument('--batch-size', type=int, default=4096, help='total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--patience', type=int, default=20, help='EarlyStopping patience (epochs without improvement)')
    args = parser.parse_args()
    main(args)



