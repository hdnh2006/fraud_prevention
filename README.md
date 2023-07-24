# Featurespace fraud prevention
This repository contains a machine learning solution for the problem of card fraud detection. A potential customer in the banking sector has approached FeatureSpace to develop a solution that can help identify fraudulent transactions and consequently improve customer satisfaction. The solution consists of a machine learning model that scores transactions based on their likelihood of being fraudulent. This scoring system aids a small team of fraud analysts to identify and review risky transactions, thereby preventing as much fraudulent activity as possible.

## Files

The repository is structured as following:

```
.
├── EDA
│   └── eda.py
├── data
│   ├── raw
│   └── processed
├── deployment
│   ├── templates
│   ├── app.py
│   └── Dockerfile
├── documents
│   ├── Description.docx
├── models
│   ├── tree_based_models.py
│   └── train_autoencoder.py
└──  reports
    └── autoencoder
    └── eda
    └── rf
    └── xgboost
    └── final_report.pdf

```

Where:

- EDA: This directory contains scripts for exploratory data analysis and preprocessing.
- data: This directory contains the raw and processed data.
- models: This directory contains scripts for training the machine learning models.
- deployment: This directory contains a Flask API for deploying the models and a Dockerfile to containerize the application.
- reports: This directory contains the final report explaining the model performance and business impact.
- Other folders has been added as information and documentation


## Usage

It is importante to follow these instruction to run all the functionalities of this repo.

### 1. Install requirements
Use a virtual environment for this purpose

```
pip install -r requirements.txt
```

### 2. Run the exploratory data analysis
To start understanding the nature of the data and to guide your future preprocessing steps, you should first run a thorough exploratory data analysis (EDA). In this project, we've prepared a dedicated script for this purpose.

To execute the EDA script, simply run the following command:
```
python eda.py
```
* `eda.py` - This script is responsible for conducting exploratory data analysis (EDA) on the provided transactional data. It reads the raw data, performs various statistical and graphical analysis operations, and outputs the results as a PDF report. It also handles preprocessing tasks such as encoding categorical variables and scaling numerical ones.


### 2. Train tree based model

This will provide you a XGBoost model with the best possible performance
```
python tree_based_models.py --avoid-optimize
```

Use avoid optimize in order to avoid the hyperparameter optimization. This process can take a long time even in GPU.

* `tree_based_models.py` - This script contains the implementation of tree-based machine learning models such as Random Forest and XGBoost. It has a class train_tree_cls that trains Random Forest and XGBoost models on provided training data, applies Synthetic Minority Over-sampling Technique (SMOTE), finds the best parameters for the models using GridSearchCV, and also explains the models using SHAP values.

### 3. Train an autoencoder neural network

This code will provide you an autoencoder with a hybrid architecture, able to classify fraud transactions

```
python train_autoencoder.py

```

* `train_autoencoder.py` - This script is responsible for training an autoencoder neural network model. The model is defined, compiled, and trained in this script. It takes in a preprocessed dataset, trains the model, and saves the trained model.

### 4 Deployment of one (or both) models

This will run a Flask API. The user will provide an `.csv` file and the API will return the results

```
python app.py
```

* `app.py` - This script contains the code for a Flask application. The application serves an API endpoint where users can upload a CSV file containing transaction data. The application then uses the trained machine learning models (XGBoost and autoencoder) to predict the likelihood of each transaction being fraudulent. The predictions are returned to the user in the form of a downloadable CSV file.


# About

This project was created by Henry, a dedicated and passionate Data Scientist. He has applied his expertise in machine learning and data analysis to develop this solution for preventing fraudulent transactions in the banking sector.

If you have any questions about the project, need help with implementation, or would like to discuss anything related to machine learning and data science, feel free to reach out to him.

Email: contact@henrynavarro.org

Please note that this project is a product of hard work and dedication. If you find it helpful, consider giving credit where it's due. Enjoy!