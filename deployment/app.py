#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:34:54 2023

@author: henry
"""

import argparse
import logging
import sys
import os
import joblib
import xgboost as xgb
import pandas as pd
from pathlib import Path
from flask import Flask, request, render_template, send_file
from keras.models import load_model
from EDA.eda import process_data


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

# Arguments
parser = argparse.ArgumentParser(description='Flask API to deploy pretrained models')
parser.add_argument('--xgboost', default =  'models/pretrained/xgboost/xgboost.joblib', help='Path to the XGBoost model')
parser.add_argument('--autoencoder', default =  'models/pretrained/autoencoder/autoencoder.keras', help='Path to the Autoencoder model')
parser.add_argument('--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--host', default='0.0.0.0', help='Host to run on')
parser.add_argument('--port', default=5000, help='Port to run on')
args = parser.parse_args()


app = Flask(__name__)

# Load autoencoder
if args.autoencoder != "":
    autoencoder = load_model(args.autoencoder)
    
    # Load the dictionary of encoders
    le_dict_NN = joblib.load(os.path.join(os.path.split(args.autoencoder)[0],'label_encoders.pkl'))
    
    # Load scaler
    scaler = joblib.load(os.path.join(os.path.split(args.autoencoder)[0], 'scaler.pkl'))
    

# Load XGboost
if args.xgboost != "":
    xgboost_model = joblib.load(args.xgboost)
    
    # Load the dictionary of encoders
    le_dict_xgb = joblib.load(os.path.join(os.path.split(args.xgboost)[0],'label_encoders.pkl'))


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        csv_file = request.files.get('file')
        if csv_file is not None:
            data = pd.read_csv(csv_file)
            data = process_data(data)
            # Drop date transaction
            data = data.drop(['transactionTime'], axis=1)
            cat_cols = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode']


            if args.xgboost != "":
                
                # Load the dictionary of encoders
                for col in cat_cols:
                    print(col)
                    le = le_dict_xgb[col]
                    data[col] = le.transform(data[col])
                    
                # make predictions
                xgboost_predictions = xgboost_model.predict(data)
            
            
            if args.autoencoder != "":
                # Load the dictionary of encoders
                for col in cat_cols:
                    print(col)
                    le = le_dict_NN[col]
                    data[col] = le.transform(data[col])

                # Transform data
                data_scaled = scaler.transform(data)
    
                autoencoder_predictions = autoencoder.predict(data_scaled)
                        
            
            # Combine predictions into a single DataFrame
            results = pd.DataFrame({
                'Autoencoder Predictions': autoencoder_predictions,
                'XGBoost Predictions': xgboost_predictions
            })
            
            # Save to CSV
            results.to_csv('predictions.csv', index=False)
            return send_file('predictions.csv', as_attachment=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=args.debug, host=args.host, port=args.port)