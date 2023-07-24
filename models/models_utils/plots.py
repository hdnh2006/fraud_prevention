#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:32:47 2023

@author: henry
"""


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, auc, roc_curve
import seaborn as sns
import logging
from pathlib import Path
import shap
from xgboost import plot_importance
import os
import pandas as pd

def plot_metrics(model, X_test, y_test, path_save):
    """
    This function plots the confusion matrix and precision-recall curve given a trained model and test data.
    
    Parameters:
    model (sklearn.estimator): Trained classifier.
    X_test (numpy.ndarray or pandas.DataFrame): Test features.
    y_test (numpy.ndarray or pandas.Series): Test labels.
    path_save (str): path to save the plots
    
    Returns:
    None
    """
    
    path_save = Path(path_save)
    path_save.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Use the fitted model to predict on the test data
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Show confusion matrix
    logging.info(f'Confusion Matrix for Model: \n {cm}')
        
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    #plt.show()
    path_out = f'{path_save}/confusion_matrix.png'  # Save the plot as an image file
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Confusion matrix saved in {path_out}')

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    auc_score = auc(recall, precision)
    
    # Print the AUC-ROC score
    logging.info(f'AUC-ROC Score for this model: {auc_score}')
    
    
    plt.plot(recall, precision, label='Model (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    #plt.show()
    path_out = f'{path_save}/precision_recall.png'  # Save the plot as an image file
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Precision-Recall curve saved in {path_out}')
    
    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label='Model (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    #plt.show()
    path_out = f'{path_save}/ROC_curve.png'  # Save the plot as an image file
    plt.savefig(path_out)  # Save the plot as an image file
    logging.info(f'Plot: Receiver Operating Characteristic curve saved in {path_out}')

    # print classification report
    logging.info(f'Classification report: \n {classification_report(y_test, y_pred)}')



def plot_SHAP_and_importance(model, X_test, output_dir):
    """
    Calculate and plot SHAP values and feature importance for a model.

    Parameters:
    model: The trained model.
    X_test: The test data.
    output_dir: The directory where to save the plots.
    """
    try:
        logging.info("Starting to calculate SHAP values...")

        # Create SHAP explainer
        explainer = shap.Explainer(model)

        # Calculate SHAP values
        shap_values = explainer(X_test)

        logging.info("SHAP values calculated.")

        # Plot SHAP values
        logging.info("Starting to plot SHAP values...")
        shap.summary_plot(shap_values, X_test)
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
        logging.info("SHAP values plotted and saved.")

        # Get feature importances
        importances = model.feature_importances_
        
        # Convert the importances into a DataFrame
        feature_importances = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': importances
        })

        # Sort the DataFrame to make the plot easier to understand
        feature_importances.sort_values('Importance', inplace=True)

        # Plot feature importances
        logging.info("Starting to plot feature importance...")
        feature_importances.plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.title('Feature Importance')
        plt.savefig(os.path.join(output_dir, 'feature_importance_plot.png'))
        logging.info("Feature importance plotted and saved.")

    except Exception as e:
        logging.error(f'Error occurred: {e}')
    

    