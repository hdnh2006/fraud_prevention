#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:34:54 2023

@author: henry

# Usage:
    python app.py
"""

import argparse
import logging
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template, send_file
from keras.models import load_model

# To get main directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# This import must be done after choose the ROOT path
from EDA.eda import process_data


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

# Arguments
parser = argparse.ArgumentParser(description='Flask API to deploy pretrained models')
parser.add_argument('--xgboost', default =  ROOT / 'models/pretrained/xgboost/xgboost.joblib', help='Path to the XGBoost model')
parser.add_argument('--autoencoder', default = ROOT / 'models/pretrained/autoencoder/autoencoder.keras', help='Path to the Autoencoder model')
parser.add_argument('--output-path', default = ROOT / 'deployment/results', help='Path to the save results')
parser.add_argument('--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--host', default='0.0.0.0', help='Host to run on')
parser.add_argument('--port', default=5000, help='Port to run on')
#parser.add_argument('--device', default = 'cpu', help='device to use: cuda, gpu or cpu')
args = parser.parse_args()


app = Flask(__name__)

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# TODO
# if args.device == 'cpu':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# elif args.device in ['gpu', 'cuda']:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# else:
#     raise ValueError(f'Value of device is \'{args.device}\'. Value accepted are: \'cpu\', \'cuda\', \'gpu\'')

# Create output path
path_save = Path(args.output_path)
path_save.mkdir(parents=True, exist_ok=True)  # make dir

# Load autoencoder
if args.autoencoder != "":
    try: 
        autoencoder = load_model(args.autoencoder)
        
        autoencoder_path = os.path.split(args.autoencoder)[0]
        
        # Load the dictionary of encoders
        le_dict_NN = joblib.load(os.path.join(autoencoder_path,'label_encoders.pkl'))
        
        # Load scaler
        scaler = joblib.load(os.path.join(autoencoder_path, 'scaler.pkl'))
        
        # Load optimal cutoff
        with open(os.path.join(autoencoder_path, 'cutoff.txt'), 'r') as f:
            cutoff = f.read()
            
        cutoff = float(cutoff) 
        
        
    except Exception as e:
        logging.error(f'An error has occured: {e}')
    

# Load XGboost
if args.xgboost != "":
    try:
        xgboost_model = joblib.load(args.xgboost)
        
        # Load the dictionary of encoders
        le_dict_xgb = joblib.load(os.path.join(os.path.split(args.xgboost)[0],'label_encoders.pkl'))
    except Exception as e:
        logging.error(f'An error has occured: {e}')


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    """Handle file uploads and predictions.

    This function is responsible for handling file uploads via POST requests,
    making predictions with the uploaded data, and returning a CSV file with the predictions.
    """
    
    if request.method == 'POST':
        try:
            csv_file = request.files.get('file')
            
            if csv_file is not None:
                logging.info('Loading dataset')
                data = pd.read_csv(csv_file) #csv_file="data/raw/transactions_obf.csv"
                eventId = data.eventId
                data = process_data(data)
                logging.info('Dataset loaded and correctly preprocessed')
                
                # Drop date transaction
                data = data.drop(['transactionTime'], axis=1)
                cat_cols = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode']
                
                logging.info('Running XGBoost model')
                if args.xgboost != "":
                    data_xgb = data.copy()
                    # Load the dictionary of encoders
                    for col in cat_cols:
                        print(col)
                        le = le_dict_xgb[col]
                        data_xgb[col] = le.transform(data_xgb[col])
                        
                    # Predict
                    xgboost_predictions = xgboost_model.predict(data_xgb)
                logging.info('Data succesfully evaluated with XGBoost')
                
                logging.info('Running Autoencoder model')
                if args.autoencoder != "":
                    data_autoencoder = data.copy()
                    # Load the dictionary of encoders
                    for col in cat_cols:
                        print(col)
                        le = le_dict_NN[col]
                        data_autoencoder[col] = le.transform(data_autoencoder[col])
    
                    # Transform data and predict
                    data_scaled = scaler.transform(data_autoencoder)   
                    probs = autoencoder.predict(data_scaled, batch_size=1000)
                    autoencoder_predictions = np.where(probs >= cutoff, 1, 0)
                    autoencoder_predictions = autoencoder_predictions.reshape(-1)
                    
                logging.info('Data succesfully evaluated with Autoencoder')
    
                # Combine predictions into a single DataFrame
                results = pd.DataFrame({
                    'eventId' : eventId,
                    'Autoencoder Predictions': autoencoder_predictions,
                    'XGBoost Predictions': xgboost_predictions
                })
                
                # Save to CSV
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                current_directory = Path(os.getcwd())
                filename = current_directory / f'predictions_{timestamp}.csv'
                print("HEEEEREEEE!")
                print(filename)
                results.to_csv(filename, index=False)
                
                return send_file(str(filename), as_attachment=True)                                                             
        
            logging.info('Data succesfully returned to the user')
        except Exception as e:
            logging.error(f'An error has occured: {e}')
            return "Data couln't be processed"
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=args.debug, host=args.host, port=args.port)