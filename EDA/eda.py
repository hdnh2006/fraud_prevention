#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:35:14 2023
@author: henry
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_data(data, out_plots):
    """
    Generate plots based on the provided data and save them in the specified directory.

    Parameters:
    data (pandas.DataFrame): The data to plot.
    out_plots (str): The directory to save the plots.

    Returns:
    None
    """
    
    logging.info("Starting report plots")
    
    # Create directory
    path_save = Path(out_plots)
    path_save.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Set the style of seaborn
    sns.set_style("whitegrid")
        
    # Plot the distribution of fraudulent vs non-fraudulent transactions
    plt.figure(figsize=(6, 6))
    sns.countplot(x='isFraud', data=data)
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
    #plt.show()
    path_out = f'{out_plots}/transaction_fraud_distribution.png'  # Save the plot as an image file
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Distribution of Fraudulent vs Non-Fraudulent Transactions saved in {path_out}')
  
    
    
    # Plot the distribution of transaction amounts for fraudulent and non-fraudulent transactions
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='isFraud', y='transactionAmount', data=data)
    plt.title('Transaction Amounts for Fraudulent vs Non-Fraudulent Transactions')
    plt.ylim(0, 200)  # Limit the y-axis to better visualize the data
    #plt.show()
    path_out = f'{out_plots}/transaction_amount_distribution.png'
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Transaction Amounts for Fraudulent vs Non-Fraudulent Transactions saved in {path_out}')
    
    
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
    #plt.show()
    path_out = f'{out_plots}/top_10_countries_with_fraud.png'
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Top 10 Countries with Highest Number of Fraudulent Transactions saved in {path_out}')
    
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
    #plt.show()
    path_out = f'{out_plots}/top_10_methods_with_fraud.png'
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Top Entry Modes with Highest Number of Fraudulent Transactions has been saved in {path_out}')
    
    
    # Convert the 'transactionTime' column to datetime
    data['transactionTime'] = pd.to_datetime(data['transactionTime'])
    data['YearMonth'] = data['transactionTime'].dt.to_period('M')
    transactions_per_month = data.groupby('YearMonth').size()

    # Plotting the number of transactions per month
    plt.figure(figsize=(10,6))
    transactions_per_month.plot(kind='line', linestyle='-', marker='o')
    plt.title('Number of Transactions per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Transactions')
    plt.grid(True)
    #plt.show()
    path_out = f'{out_plots}/transactions_per_month.png'
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Number of Transactions per Month has been saved in {path_out}')

    
    
    logging.info("Completed report plots.")


def process_data(data):
    """
    Process the provided data by generating new features, handling missing values, and dropping unnecessary columns.

    Parameters:
    data (pandas.DataFrame): The data to process.

    Returns:
    pandas.DataFrame: The processed data.
    """
    
    logging.info("Starting data preprocessing")
    
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
    
    # Check for missing values
    logging.info("Checking for missing values...")
    missing_values = data.isnull().sum()
    
    # Log the number of missing values per column
    for column, num_missing in missing_values.items():
        logging.info("Column {} has {} missing values".format(column, num_missing))
    
    # Drop the 'merchantZip' column, due to the number of NA's values
    if 'merchantZip' in data.columns:
        data = data.drop('merchantZip', axis=1)
        logging.info('\'merchantZip\' variable dropped')
    
    # Drop the 'eventId' column
    if 'eventId' in data.columns:
        data = data.drop('eventId', axis=1)
        logging.info('\'eventId\' variable dropped')
    
    # Drop the 'reportedTime' column as it's no longer needed
    if 'reportedTime' in data.columns:
        data = data.drop('reportedTime', axis=1)
        logging.info('\'reportedTime\' variable dropped')
    
    logging.info("Completed data processing.")
    
    return data


def main(args):

    """
    The main function that controls the flow of the script. It loads the data, merges it with the labels, 
    calls the functions to perform EDA and process the data, and saves the processed data.
    
    Parameters:
    args (argparse.Namespace): The command-line arguments.
    
    Returns:
    None
    """
    
    logging.info("Starting EDA script")
    
    # Load the data
    data = pd.read_csv(args.data_path)
    logging.info("Loaded data from {}".format(args.data_path))

    # Load the labels data
    labels = pd.read_csv(args.labels_path)
    logging.info("Loaded labels from {}".format(args.labels_path))

    # Add a 'isFraud' column to the labels data
    labels['isFraud'] = 1

    # Merge the data and the labels on the 'eventId' column
    data = pd.merge(data, labels, on='eventId', how='left')

    # Fill NaN values in the 'isFraud' column with 0 (indicating non-fraudulent transactions)
    data['isFraud'] = data['isFraud'].fillna(0)
    
    # Call function to plot data
    if not args.avoid_plots:
        plot_data(data, args.out_plots)
    
    # Call function to process data
    data = process_data(data)
    data.to_csv(args.out_data, index=False)
    
    logging.info("Completed EDA.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on transaction and labels data. Output: report plots and data_processed")
    parser.add_argument('--data-path', type=str, help='Path to transaction data CSV file.', default= ROOT / 'data/raw/transactions_obf.csv')
    parser.add_argument('--labels-path', type=str, help='Path to labels data CSV file.', default = ROOT / 'data/raw/labels_obf.csv')
    parser.add_argument('--out-data', type=str, help='Path to labels data CSV file.', default = ROOT / 'data/processed/data_processed.csv')
    parser.add_argument('--out-plots', type=str, help='Path to save plots', default = ROOT / 'reports/eda')
    parser.add_argument('--avoid-plots',  action='store_true', help='Do not save plots. If not, plots will be saved in report folder')
    args = parser.parse_args()
    main(args)
