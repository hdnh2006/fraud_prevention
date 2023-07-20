#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:34:50 2023

@author: henry
"""

import pandas as pd
import numpy as np

# ============= Show data ================== #

# Load the data
data = pd.read_csv('data-new/transactions_obf.csv')

# Display the first few rows of the data
data.head()


# Load the labels data
labels = pd.read_csv('data-new/labels_obf.csv')

# Display the first few rows of the labels
labels.head()


# =========== Create variable fraud ============ #

# Add a 'isFraud' column to the labels data
labels['isFraud'] = 1

# Merge the data and the labels on the 'eventId' column
data = pd.merge(data, labels, on='eventId', how='left')

# Fill NaN values in the 'isFraud' column with 0 (indicating non-fraudulent transactions)
data['isFraud'] = data['isFraud'].fillna(0)

# Display the first few rows of the merged data
data.head()


# ========= Extra variables ============= #


# Convert transactionTime to datetime and extract features
data['transactionTime'] = pd.to_datetime(data['transactionTime'])
data['transactionHour'] = data['transactionTime'].dt.hour
data['transactionDayOfWeek'] = data['transactionTime'].dt.dayofweek
data['transactionDayOfMonth'] = data['transactionTime'].dt.day
data['transactionMonth'] = data['transactionTime'].dt.month

# Create binary features for posEntryMode
pos_entry_mode_dummies = pd.get_dummies(data['posEntryMode'], prefix='posEntryMode')
data = pd.concat([data, pos_entry_mode_dummies], axis=1)

# Create ratio of transaction amount to available cash
data['transactionAmountToAvailableCashRatio'] = data['transactionAmount'] / (data['availableCash'] + 1)

# Create feature for whether the transaction occurred in the most common country
most_common_country = data['merchantCountry'].mode()[0]
data['isMostCommonCountry'] = (data['merchantCountry'] == most_common_country).astype(int)

# Create feature for whether the transaction occurred in the most entre mode
most_common_country = data['posEntryMode'].mode()[0]
data['isMostEntryMode'] = (data['posEntryMode'] == most_common_country).astype(int)


# Display the first few rows of the dataset with the new features
data.head()


# =============== EDA =================== #

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set_style("whitegrid")

# Plot the distribution of fraudulent vs non-fraudulent transactions
plt.figure(figsize=(6, 6))
sns.countplot(x='isFraud', data=data)
plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
plt.show()




# Plot the distribution of transaction amounts for fraudulent and non-fraudulent transactions
plt.figure(figsize=(10, 6))
sns.boxplot(x='isFraud', y='transactionAmount', data=data)
plt.title('Transaction Amounts for Fraudulent vs Non-Fraudulent Transactions')
plt.ylim(0, 200)  # Limit the y-axis to better visualize the data
plt.show()



# Count the number of fraudulent transactions for each country
fraud_counts = data[data['isFraud'] == 1]['merchantCountry'].value_counts()

# Get the top 10 countries with the highest number of fraudulent transactions
top_countries = fraud_counts.head(10)

# Plot the number of fraudulent transactions for the top 10 countries
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.index, y=top_countries.values, order=top_countries.index)
plt.title('Top 10 Countries with Highest Number of Fraudulent Transactions')
plt.xlabel('Country Code')
plt.ylabel('Number of Fraudulent Transactions')
plt.show()



# Count the number of fraudulent transactions for each Entry Mode
fraud_counts = data[data['isFraud'] == 1]['posEntryMode'].value_counts()

# Get the top of fraudulent transactions
top_entry_modes = fraud_counts

# Plot the number of fraudulent transactions for the top 10 countries
plt.figure(figsize=(10, 6))
sns.barplot(x=top_entry_modes.index, y=top_entry_modes.values, order=top_entry_modes.index)
plt.title('Top Entry Modes with Highest Number of Fraudulent Transactions')
plt.xlabel('Entry Mode')
plt.ylabel('Number of Fraudulent Transactions')
plt.show()



# =============== data processing ================= # 

# Check for missing values
data.isnull().sum()


# Drop the 'merchantZip' column
data = data.drop('merchantZip', axis=1)

# Drop the 'eventId' column
data = data.drop('eventId', axis=1)

# Drop the 'reportedTime' column as it's no longer needed
data = data.drop('reportedTime', axis=1)

# Check the first few rows of the data
data.head()


from sklearn.preprocessing import LabelEncoder

# Initialize a label encoder
le = LabelEncoder()

# List of categorical columns to encode
cat_cols = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode']

# Encode each column
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

# Check the first few rows of the data
data.head()



# ============== training process ===================== #



# Logistic regression, most basic classifier
from sklearn.model_selection import train_test_split

# Separate the features (X) and the target variable (y)
X = data.drop(['transactionTime', 'isFraud'], axis=1)
y = data['isFraud']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the sizes of the training set and the test set
X_train.shape, X_test.shape


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Initialize a Logistic Regression model
lr = LogisticRegression(random_state=42, max_iter=1000)

# Train the model
lr.fit(X_train, y_train)

# Make predictions on the test set
lr_preds = lr.predict(X_test)

# Calculate the AUC-ROC score
lr_auc = roc_auc_score(y_test, lr_preds)

# Show confusion matrix
print(f'Confusion Matrix for Logistic Regression: \n {confusion_matrix(y_test, lr_preds)}')

# Print the AUC-ROC score
print(f'AUC-ROC Score for Logistic Regression: {lr_auc}')

# Print the classification report
print(classification_report(y_test, lr_preds))





# Random forest, decision tree-based model

from sklearn.ensemble import RandomForestClassifier


# Initialize a new Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
rf_preds = rf.predict(X_test)

# Calculate the AUC-ROC score
rf_auc = roc_auc_score(y_test, rf_preds)

# Show confusion matrix
print(f'Confusion Matrix for Random Forest: \n {confusion_matrix(y_test, rf_preds)}')

# Print the AUC-ROC score
print(f'AUC-ROC Score for Random Forest: {rf_auc}')

# Print the classification report
print(classification_report(y_test, rf_preds))



# ========================= SMOTE ========================== #


from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit SMOTE on the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the number of fraudulent and non-fraudulent transactions after SMOTE
y_train_smote.value_counts()




from sklearn.ensemble import RandomForestClassifier

# Initialize a new Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
rf_preds = rf.predict(X_test)

# Calculate the AUC-ROC score
rf_auc = roc_auc_score(y_test, rf_preds)

# Show confusion matrix
print(f'Confusion Matrix for Random Forest: \n {confusion_matrix(y_test, rf_preds)}')

# Print the AUC-ROC score
print(f'AUC-ROC Score for Random Forest: {rf_auc}')

# Print the classification report
print(classification_report(y_test, rf_preds))






# ============ XGBoost ================== # 

from xgboost import XGBClassifier

# Initialize a new XGBoost model
#xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0, predictor="gpu_predictor")

# Attempt to train a model on the GPU
try:
    print('GPU is available for XGBoost.')
    xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0, predictor="gpu_predictor", use_label_encoder=False, eval_metric='logloss', random_state=42)
except Exception as e:
    print('GPU is not available for XGBoost:', str(e), 'using CPU')
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


# Train the model
xgb.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
xgb_preds = xgb.predict(X_test)

# Calculate the AUC-ROC score
xgb_auc = roc_auc_score(y_test, xgb_preds)

# Show confusion matrix
print(f'Confusion Matrix for XGBoost: \n {confusion_matrix(y_test, xgb_preds)}')

# Print the AUC-ROC score
print(f'AUC-ROC Score for XGBoost: {xgb_auc}')

# Print the classification report
print(classification_report(y_test, xgb_preds))






# hyperparameter tunning for XGBoost

from sklearn.model_selection import GridSearchCV

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
    xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0, predictor="gpu_predictor", use_label_encoder=False, eval_metric='logloss', random_state=42)
except Exception as e:
    print('GPU is not available for XGBoost:', str(e), 'using CPU')
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)

# Fit the Grid Search model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

best_params










# ============== With autoencoders ==================== #

from sklearn.preprocessing import MinMaxScaler

# Initialize a MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)



# Split the training data into non-fraudulent and fraudulent transactions
X_train_non_fraud = X_train_scaled[y_train == 0]
X_train_fraud = X_train_scaled[y_train == 1]

# Check the number of non-fraudulent and fraudulent transactions
X_train_non_fraud.shape, X_train_fraud.shape

y_train_nonfraud = y_train[y_train == 0]


from keras_core.models import Model
from keras_core.layers import Input, Dense
# Define the dimensionality of the input data and the hidden layers
input_dim = X_train_non_fraud.shape[1]




hidden_dim = int(input_dim / 2.0)

# Define the encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(hidden_dim, activation='relu')(input_layer)

# Define the decoder
decoder = Dense(input_dim, activation='sigmoid')(encoder)

# Define the autoencoder as the composition of the encoder and the decoder
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_train_non_fraud, X_train_non_fraud, epochs=50, batch_size=4096, shuffle=True, validation_split=0.2, verbose=0)

# Use the model to predict on the test set
X_test_pred = autoencoder.predict(X_test)

# Compute the mean squared error of each prediction
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Classify transactions with an MSE above a certain threshold as fraudulent
#fraud_pred = mse > threshold







# Build autoencoder model
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(100, activation='tanh')(input_layer)
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

# Add a single neuron with a logistic activation function
output = Dense(1, activation='sigmoid')(decoded)

# Define the model as the composition of the input layer and the output layer
model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Fit the model
model.fit(X_train_non_fraud, y_train_nonfraud, epochs=50)

# Use the model to predict on the test set
fraud_pred = model.predict(X_test)










"""
