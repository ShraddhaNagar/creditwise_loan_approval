import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the Brain and the Translator
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("💳 CreditWise Loan Predictor")
st.write("Fill the details below to check loan eligibility.")

# 2. Create the Form (Input Fields)
income = st.number_input("Applicant Income", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)

# 3. Prediction Logic
if st.button("Predict"):
    # Create the same features we used in Colab
    dti_sq = dti_ratio ** 2
    credit_sq = credit_score ** 2
    income_log = np.log1p(income)
    
    # Pre-process the input (This must match your X_train columns)
    # Note: For now, I'm using a simplified input list. 
    # In real app, we will add all 28 columns.
    
    st.success("App structure is ready! Next, we will connect your actual model data.")