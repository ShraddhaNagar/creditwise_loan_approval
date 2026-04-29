import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 CreditWise Loan Predictor")
st.write("Fill the details below to check loan eligibility.")

# Input fields
income = st.number_input("Applicant Income", min_value=0)
score = st.number_input("Credit Score", min_value=0, max_value=900)
dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)

if st.button("Predict"):
    # Prepare data for prediction
    input_data = pd.DataFrame([[income, score, dti]], columns=['ApplicantIncome', 'CreditScore', 'DTI_Ratio'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
