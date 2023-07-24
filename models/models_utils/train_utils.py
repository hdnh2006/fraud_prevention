#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:20:35 2023

@author: henry
"""
import logging
import sys
import os
import io
import contextlib
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .plots import plot_metrics
from sklearn.model_selection import GridSearchCV


# To get main directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def train_RF(X_train, y_train, X_test, y_test):

    try:        
        # Initialize a new Random Forest model
        rf = RandomForestClassifier(random_state=42)
        
        # Train the model
        rf.fit(X_train, y_train)
        
        # Plot metrics for rf
        plot_metrics(rf, X_test, y_test, ROOT / 'reports/rf' )
        
        return rf
        
    except Exception as e:
        logging.error(f'An Error has occured: {e}')


def train_XGBoost(X_train, y_train, X_test, y_test, params=None):

    if params is None:
        params = {
            'eval_metric': 'logloss', 
            'random_state': 42
        }

    try:        
        # Attempt to xgboost on GPU with default hyperparameters
        try:
            params.update({'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': "gpu_predictor"})
            xgb = XGBClassifier(**params)
            xgb.fit(X_train, y_train)
            logging.info('GPU is available for XGBoost.')
            
        except Exception as e:
            logging.info('GPU is not available for XGBoost:', str(e), 'using CPU')
            params.pop('tree_method', None)
            params.pop('gpu_id', None)
            params.pop('predictor', None)
            xgb = XGBClassifier(**params)
            xgb.fit(X_train, y_train)
        
        # Plot metrics for rf
        plot_metrics(xgb, X_test, y_test, ROOT / 'reports/xgboost' )
        
        return xgb        
        
    except Exception as e:
        logging.error(f'An Error has occured: {e}')
    
    



def apply_SMOTE(X_train, y_train):
    
    try:
        # Initialize SMOTE
        smote = SMOTE(random_state=42)
    
        # Fit SMOTE on the training data
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
        # Check the number of fraudulent and non-fraudulent transactions after SMOTE
        logging.info(f'After apply SMOTE, the new sample is distributed like this: \n {y_train_smote.value_counts()}')
        
        return X_train_smote, y_train_smote
    
    except Exception as e:
        logging.error(f'An Error has occured: {e}')
    


def get_best_params(X_train, y_train):

    # Define the hyperparameters and their values that we want to tune
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.25, 0.5, 1.0]
    }
    
    # Attempt to train a model on the GPU
    try:
        print('GPU is available for XGBoost.')
        xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0, predictor="gpu_predictor", eval_metric='logloss', random_state=42)
    except Exception as e:
        print('GPU is not available for XGBoost:', str(e), 'using CPU')
        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    
    
    # Initialize the Grid Search model
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    
    
    # Redirect standard output to a string stream
    stdout = io.StringIO()
    
    # This is done in order to save everything in my logs
    with contextlib.redirect_stdout(stdout):
        # Fit the Grid Search model
        grid_search.fit(X_train, y_train)
    
    # Get the standard output as a string
    output = stdout.getvalue()
    
    # Log the output
    for line in output.split("\n"):
        logging.info(line)
    
    
    # Get the best parameters
    best_params = grid_search.best_params_
    
    return best_params


   
