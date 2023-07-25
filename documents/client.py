#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:35:41 2023

@author: henry

This is an example of how to use the api from python.
This code can be provided to the customer

"""

import requests

# Specify the url of the API
url = "http://0.0.0.0:5000/"

# Specify the path to the file to be sent
file_path = "data/raw/transactions_obf.csv"

# Open the file in binary mode
file = open(file_path, "rb")

# Send a POST request to the API with the file attached
response = requests.post(url, files={"file": file})

# Check if the request was successful
if response.status_code == 200:
    # If successful, save the received file
    with open("predictions.csv", "wb") as f:
        f.write(response.content)
    print("File saved successfully.")
else:
    print("Request failed.")
