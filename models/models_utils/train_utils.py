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
import shap
import joblib
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .plots import plot_metrics, plot_SHAP_and_importance, plot_history
from sklearn.model_selection import GridSearchCV
from wandb.xgboost import WandbCallback
from keras_core.models import Sequential
from keras_core.layers import Input, Dense, Dropout, BatchNormalization
from keras_core.callbacks import EarlyStopping
from keras_core.losses import mean_squared_error
from keras_core.optimizers import Adam

# To get main directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# set WANDB_DIR environment variable
os.environ['WANDB_DIR']= str(ROOT)

class train_tree_cls:
    
    """
    A class to train tree-based models and perform various related tasks.
    
    Attributes
    ----------
    X_train : pandas.DataFrame
        Features for the training data.
    y_train : pandas.Series
        Target variable for the training data.
    X_test : pandas.DataFrame
        Features for the test data.
    y_test : pandas.Series
        Target variable for the test data.
    project : str
        The name of the project in wandb.
    
    Methods
    -------
    train_RF(name=None):
        Trains a Random Forest model and logs the metrics using wandb.
    train_XGBoost(name, params=None):
        Trains an XGBoost model with the given parameters and logs the metrics using wandb.
    apply_SMOTE():
        Applies Synthetic Minority Over-sampling Technique (SMOTE) to the training data.
    get_best_params():
        Gets the best parameters for an XGBoost model using GridSearchCV.
    SHAP_explain(model, name_model=None):
        Creates a SHAP explainer for the model and plots the SHAP values.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, project):
        
        """
        Constructs all the necessary attributes for the train_tree_cls object.
        
        Parameters
        ----------
        X_train : pandas.DataFrame
            Features for the training data.
        y_train : pandas.Series
            Target variable for the training data.
        X_test : pandas.DataFrame
            Features for the test data.
        y_test : pandas.Series
            Target variable for the test data.
        project : str
            The name of the project in wandb.
        """
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.project = project
    

    def train_RF(self, name = None):
        
        """
        Trains a Random Forest model and logs the metrics using wandb.
        
        Parameters
        ----------
        name : str, optional
            The name of the experiment in wandb.
        
        Returns
        -------
        rf : RandomForestClassifier
            The trained Random Forest model.
        """
    
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
        
        """
        Trains an XGBoost model with the given parameters and logs the metrics using wandb.

        Parameters
        ----------
        name : str
            The name of the experiment in wandb.
        params : dict, optional
            The parameters for the XGBoost model. If None, default parameters will be used.

        Returns
        -------
        xgb : XGBClassifier
            The trained XGBoost model.
        """
    
        if params is None:
            params = {
                'eval_metric': 'logloss', 
                'random_state': 42
            }
        
        try:
            # Start a wandb run
            run = wandb.init(project=self.project, name = name)
            
            # Attempt to xgboost on GPU with default hyperparametersfrom wandb.keras import WandbCallback

            try:
                params.update({'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': "gpu_predictor"})
                xgb = XGBClassifier(**params, callbacks=[WandbCallback(log_model=True)])
                xgb.fit(self.X_train, self.y_train)
                logging.info('GPU is available for XGBoost.')
                
            except Exception as e:
                logging.info('GPU is not available for XGBoost:', str(e), 'using CPU')
                params.pop('tree_method', None)
                params.pop('gpu_id', None)
                params.pop('predictor', None)
                xgb = XGBClassifier(**params, callbacks=[WandbCallback(log_model=True)])
                xgb.fit(self.X_train, self.y_train)
            
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
            
    def get_best_params(self):
        
        """
        Gets the best parameters for an XGBoost model using GridSearchCV.

        Returns
        -------
        best_params : dict
            The best parameters for an XGBoost model.
        """
    
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
            logging.info('GPU is available for XGBoost.')
            xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0, predictor="gpu_predictor", eval_metric='logloss', random_state=42)
        except Exception as e:
            logging.info('GPU is not available for XGBoost:', str(e), 'using CPU')
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
    
    def SHAP_explain(self, model, name_model=None):
        
        """
        Creates a SHAP explainer for the model and plots the SHAP values.

        Parameters
        ----------
        model : XGBClassifier or RandomForestClassifier
            The model to explain.
        name_model : str, optional
            The name of the model.

        Returns
        -------
        None
        """
        # Create SHAP explainer
        explainer = shap.Explainer(model)

        # Calculate SHAP values
        shap_values = explainer(self.X_test)

        # Plot SHAP values
        shap.summary_plot(shap_values, self.X_test)

        # Plot SHAP and importance
        plot_SHAP_and_importance(model, self.X_test, ROOT / f'reports/{name_model}' )
   


class train_autoencoder:
    
    def __init__(self, X_train, y_train, X_test, y_test, project):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.project = project
    
    def scale_dataset(self, args):
   
        # Initialize a MinMaxScaler
        scaler = StandardScaler()
        
        # Fit the scaler on the training data and transform it
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        
        # Transform the test data
        self.X_test_scaled = scaler.transform(self.X_test)
        
        # Split the training data into non-fraudulent and fraudulent transactions
        self.X_train_non_fraud = self.X_train_scaled[self.y_train == 0]
        self.X_train_fraud = self.X_train_scaled[self.y_train == 1]
                
        self.y_train_nonfraud = self.y_train[self.y_train == 0]
        
        # Export the scaler
        path_save = Path(args.output_model)
        path_save.mkdir(parents=True, exist_ok=True)  # make dir
        path_out = os.path.join(path_save,'scaler.pkl')
        joblib.dump(scaler, path_out)
  
    
    def train(self, args):
        
        # Start a wandb run
        run = wandb.init(project=args.wandb_project, name = args.name)
        
        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience= args.patience, restore_best_weights=True)

        
        # Define the dimensionality of the input data and the hidden layers
        input_dim = int(self.X_train_non_fraud.shape[1])
        hidden_dim = int(input_dim * 2.0)
        hidden_dim_1 = int(input_dim * 4.0)
        hidden_dim_2 = int(hidden_dim_1 * 8.0)

        # Input shape
        input_shape = Input(shape=[input_dim])

        # Encoder part
        encoder = Sequential(name='encoder')
        encoder.add(input_shape)
        encoder.add(layer=Dense(units=hidden_dim, activation= "relu"))
        encoder.add(BatchNormalization())  # Add Batch Normalization after the first layer
        # encoder.add(Dropout(0.1))
        encoder.add(layer=Dense(units=hidden_dim_1, activation= "relu"))
        encoder.add(BatchNormalization())  # Add Batch Normalization after the second layer
        encoder.add(layer=Dense(units=hidden_dim_2, activation= "relu"))
        encoder.add(BatchNormalization())  # Add Batch Normalization after the second layer


        # Decoder part
        decoder = Sequential(name='decoder')
        decoder.add(Input(shape=[hidden_dim_2]))
        decoder.add(layer=Dense(units=hidden_dim_1, activation= "relu"))
        decoder.add(BatchNormalization())  # Add Batch Normalization after the second layer
        decoder.add(layer=Dense(units=hidden_dim, activation= "relu"))
        decoder.add(BatchNormalization())  # Add Batch Normalization after the second layer
        # decoder.add(Dropout(0.1))
        decoder.add(layer=Dense(units=input_dim, activation= "relu"))
        decoder.add(BatchNormalization())  # Add Batch Normalization after the second layer

        # Add a final layer with 1 unit and a sigmoid activation
        final_layer = Dense(units=1, activation='sigmoid')
        
        self.autoencoder = Sequential([encoder, decoder, final_layer])

        # self.autoencoder.compile(
        # 	loss= "mean_squared_error",
        # 	optimizer= "adam",
        # 	metrics=["mean_squared_error"])
        
        # Choose optimizer and compile the NN
        optimizer = Adam(learning_rate=0.001)

        self.autoencoder.compile(
         	loss= "binary_crossentropy",
         	optimizer= optimizer,
         	metrics=["accuracy"])


        # Fit the model and store training info in history
        # history = self.autoencoder.fit(self.X_train_non_fraud, self.X_train_non_fraud, batch_size=4096, epochs=100, verbose=1, shuffle=True, validation_split=0.2, callbacks=[early_stopping])
        history = self.autoencoder.fit(self.X_train_scaled, self.y_train, batch_size= args.batch_size,
                                       epochs=args.epochs, verbose=1, shuffle=True, validation_split=0.2,
                                       callbacks=[early_stopping])
        
        # plot history loss
        plot_history(history, ROOT / 'reports/autoencoder' )
        
        # Close your wandb run
        run.finish()
        
        return history
    
    def find_best_cutoff(self):
        
        logging.info('Finding the best cutoff in the training dataset.')
        # Make predictions
        y_probs = self.autoencoder.predict(self.X_train_scaled)
    
        # Create a list to store the accuracies for each threshold
        accuracies = []
    
        # Test thresholds between 0 and 1
        thresholds = np.linspace(0, 1, 100)
    
        # For each threshold, calculate the accuracy and append it to the accuracies list
        for threshold in thresholds:
            y_pred = np.where(y_probs >= threshold, 1, 0)
            accuracies.append(accuracy_score(self.y_train, y_pred))
    
        # Find the optimal threshold: the one that maximizes accuracy
        optimal_idx = np.argmax(accuracies)
        self.cutoff = thresholds[optimal_idx]
        logging.info(f"""Best threshold: {self.cutoff}""")

        

    
    def classify(self, args):
    
        
        # train_predicted_x = self.autoencoder.predict(x=self.X_train_scaled)
        # train_events_mse = mean_squared_error(self.X_train_scaled, train_predicted_x)
        # cut_off = np.percentile(train_events_mse.cpu(), 95)

        # # abnormal event
        # predicted_x = self.autoencoder.predict(self.X_test_scaled)
        # abnormal_events_mse = mean_squared_error(self.X_test_scaled, predicted_x)

        # pred_test = (abnormal_events_mse.cpu().numpy() > cut_off).astype(int)
        
        logging.info('Evaluating test dataset')
        
        # Evaluating test dataset
        probs = self.autoencoder.predict(self.X_test_scaled, batch_size=4096)
        y_pred = np.where(probs >= self.cutoff, 1, 0)

        logging.info(f'Confusion Matrix for this model: {confusion_matrix(self.y_test, y_pred)}')


def apply_SMOTE(X_train, y_train):
    
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to the training data.

    Returns
    -------
    X_train_smote : pandas.DataFrame
        The training data features after applying SMOTE.
    y_train_smote : pandas.Series
        The training data target variable after applying SMOTE.
    """
    
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
        
        
        














