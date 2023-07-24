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
import wandb
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from .plots import plot_metrics
from sklearn.model_selection import GridSearchCV
from wandb.xgboost import WandbCallback

# To get main directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# set WANDB_DIR environment variable
os.environ['WANDB_DIR']= str(ROOT)

class train_tree_cls:
    
    def __init__(self, X_train, y_train, X_test, y_test, project):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.project = project
    

    def train_RF(self, name = None):
    
        try:
    
            # Start a wandb run
            run = wandb.init(project=self.project, name = name)
            
            # Initialize a new Random Forest model
            rf = RandomForestClassifier(random_state=42)
            
            # Train the model
            rf.fit(self.X_train, self.y_train)
            
            # Get predictions
            y_pred = rf.predict(self.X_test)
            
            # Log metrics to wandb
            wandb.log({
                "XGBoost": {
                    "Accuracy": accuracy_score(self.y_test, y_pred),
                    "ROC AUC": roc_auc_score(self.y_test, y_pred),
                    "F1 Score": f1_score(self.y_test, y_pred)
                }
            })
            
            # Plot metrics for rf
            plot_metrics(rf, self.X_test, self.y_test, ROOT / 'reports/rf' )
            
            # Close your wandb run
            run.finish()
            
            return rf
            
        except Exception as e:
            logging.error(f'An Error has occured: {e}')
    
    
    def train_XGBoost(self, name, params=None):
    
        if params is None:
            params = {
                'eval_metric': 'logloss', 
                'random_state': 42
            }
        
        try:
            # Start a wandb run
            run = wandb.init(project=self.project, name = name)
            
            # Attempt to xgboost on GPU with default hyperparameters
            try:
                params.update({'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': "gpu_predictor"})
                xgb = XGBClassifier(**params)
                xgb.fit(self.X_train, self.y_train, callbacks=[WandbCallback(log_model=True)])
                logging.info('GPU is available for XGBoost.')
                
            except Exception as e:
                logging.info('GPU is not available for XGBoost:', str(e), 'using CPU')
                params.pop('tree_method', None)
                params.pop('gpu_id', None)
                params.pop('predictor', None)
                xgb = XGBClassifier(**params)
                xgb.fit(self.X_train, self.y_train, callbacks=[WandbCallback(log_model=True)])
            
            # Get predictions
            y_pred = xgb.predict(self.X_test)
            
            # Log metrics to wandb
            wandb.log({
                "XGBoost": {
                    "Accuracy": accuracy_score(self.y_test, y_pred),
                    "ROC AUC": roc_auc_score(self.y_test, y_pred),
                    "F1 Score": f1_score(self.y_test, y_pred)
                }
            })
            
            # Plot metrics for rf
            plot_metrics(xgb, self.X_test, self.y_test, ROOT / 'reports/xgboost' )
            
            # Close your wandb run
            run.finish()
            
            return xgb        
            
        except Exception as e:
            logging.error(f'An Error has occured: {e}')
        
    
    
    def apply_SMOTE(self):
        
        try:
            # Initialize SMOTE
            smote = SMOTE(random_state=42)
        
            # Fit SMOTE on the training data
            self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
            # Check the number of fraudulent and non-fraudulent transactions after SMOTE
            logging.info(f'After apply SMOTE, the new sample is distributed like this: \n {self.y_train_smote.value_counts()}')
            
            return self.X_train_smote, self.y_train_smote
        
        except Exception as e:
            logging.error(f'An Error has occured: {e}')
        
    
    
    def get_best_params(self):
    
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
            grid_search.fit(self.X_train, self.y_train)
        
        # Get the standard output as a string
        output = stdout.getvalue()
        
        # Log the output
        for line in output.split("\n"):
            logging.info(line)
        
        
        # Get the best parameters
        best_params = grid_search.best_params_
        
        return best_params


   
