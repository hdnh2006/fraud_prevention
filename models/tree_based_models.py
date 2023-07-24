#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:49:20 2023

@author: henry
"""

import argparse
import logging
import sys
import os
import time
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from models_utils.train_utils import train_tree_cls, apply_SMOTE
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
    
    if not args.avoid_optimize:
        logging.warning("""
                        ⚠️ ⚠️ ⚠️  ALERT! The process to get the best hyperparameters can take more than one
                        hour even in GPU, if you don\'t have that time set --avoid-optimize.
                        Press Ctrl+C to cancel this process and add the flag suggested
                        """)
        time.sleep(10)
    
    
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
    
    # Separate the features (X) and the target variable (y)
    X = data.drop(['transactionTime', 'isFraud'], axis=1)
    y = data['isFraud']
    
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check the sizes of the training set and the test set
    logging.info(f'Data splitted with Train shape: {X_train.shape}, Test_shape: {X_test.shape}')
    
    # Initialize training class
    tree_cls = train_tree_cls(X_train, y_train, X_test, y_test, project= args.wandb_project)
         
    # Random forest, decision tree-based model
    logging.info('🟢🟢🟢 Training Random Forest Classifier...')
    rf = tree_cls.train_RF(name='1st-rf')
    
    # Train a XGBoost model
    logging.info('🟢🟢🟢 Training eXtreme Gradient Boosting machine...')
    xgb = tree_cls.train_XGBoost(name='1st-XGBoost')
    
    # Message
    logging.info('🟢🟢🟢 Due to the metrics of XGBoost, we will retrain the model using SMOTE')
    
    # Retrain with SMOTE
    logging.info('🟢🟢🟢 Training eXtreme Gradient Boosting machine with SMOTE data')
    X_train_smote, y_train_smote = apply_SMOTE(X_train, y_train)
    tree_cls.X_train = X_train_smote
    tree_cls.y_train = y_train_smote
    xgb_smote = tree_cls.train_XGBoost('XGBoost-SMOTE')
        
    # Message
    logging.info('🟢🟢🟢 The improvement is good for false negatives, now the model will be trained with the best hyperparameters')
    
    if not args.avoid_optimize:
        logging.warning('⚠️ ⚠️ ⚠️ ALERT! This process can take more than one hour even in GPU')
        best_params = tree_cls.get_best_params(X_train, y_train) # It apply XGBoost
        logging.info(f'These are the best hyperparameters for the dataset provided: \n {best_params}')
    else:
        logging.warning('🟢🟢🟢 Using the best hyperparameters gotten in the past')
        
        # The code was already ran, an these are the results:
        best_params = {'colsample_bytree': 0.5,
                      'gamma': 0,
                      'learning_rate': 0.1,
                      'max_depth': 20,
                      'n_estimators': 100,
                      'subsample': 1.0}
    
    xgb_best = tree_cls.train_XGBoost(name= 'XGBoost-best-params', params = best_params)
    path_save = Path(args.output_model)
    path_save.mkdir(parents=True, exist_ok=True)  # make dir
    path_out = os.path.join(path_save, 'xgboost.joblib')
    # Save the model
    joblib.dump(xgb_best, path_out)
    # Save encoder
    path_out = os.path.join(path_save, 'label_encoders.pkl')
    joblib.dump(le_dict, path_out)
    
    
    # Explainability using SHAP and feature importance
    logging.info('🟢🟢🟢 Looking for explainability of the model')
    tree_cls.SHAP_explain(xgb_best, name_model='xgboost')
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on transaction and labels data. Output: report plots and data_processed")
    parser.add_argument('--data-processed', type=str, help='Path to transaction data CSV file.', default=  'data/processed/data_processed.csv')
    parser.add_argument('--wandb-project', type=str, help='Name of the project in wandb', default = 'tree-based-models')
    parser.add_argument('--out-plots', type=str, help='Path save plots', default =  'reports/eda')
    parser.add_argument('--output-model', type=str, help='Path save the model', default =  'models/pretrained/xgboost')
    parser.add_argument('--avoid-optimize',  action='store_true', help='If you want to get the best hyperparameters (Alert! it will take a long time even in GPU)')
    args = parser.parse_args()
    main(args)



