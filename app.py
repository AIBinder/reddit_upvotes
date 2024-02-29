import streamlit as st
import requests
import json

st.title('Reddit Upvotes Prediction')

# Define textual input fields
title = st.text_input("Titel", "")
username = st.text_input("Username", "")
text = st.text_input("Text", "")

# Placeholder for model output
output_placeholder = st.empty()

# URL of your Flask app's prediction endpoint
url = 'http://127.0.0.1:5000/upvotes'

headers = {'Content-Type': 'application/json'}

# Button to trigger model call
if st.button('Submit'):
    # Sample data to send for prediction
    data = {'title': title, 'author': username, 'text': text}
    # Send a POST request with your input data
    response = requests.post(url, json=data, headers=headers)
    model_output = response.json()
    output_placeholder.write(model_output)