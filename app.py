import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 CreditWise Full Loan Predictor")
st.write("Please fill all details for an accurate prediction.")

# Form banana taaki itne saare inputs manage ho sakein
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Applicant Income", min_value=0, value=5000)
        co_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        score = st.number_input("Credit Score", min_value=0, max_value=900, value=700)
        loan_amt = st.number_input("Loan Amount", min_value=0, value=20000)
        
    with col2:
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        existing_loans = st.number_input("Existing Loans", min_value=0, value=0)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
        savings = st.number_input("Savings", min_value=0, value=1000)
        term = st.number_input("Loan Term (Months)", value=36)

    # Submit button
    submitted = st.form_submit_button("Predict Approval")

if submitted:
    # 1. Saare 27 features ki list (Jo Colab mein thi)
    # Note: Humein wahi exact order chahiye jo train_test_split ke waqt tha
    feature_names = [
        'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 
        'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings', 
        'Collateral_Value', 'Loan_Amount', 'Loan_Term', 'Education_Level',
        'Employment_Status_Salaried', 'Employment_Status_Self-employed', 
        'Employment_Status_Unemployed', 'Marital_Status_Single', 
        'Loan_Purpose_Car', 'Loan_Purpose_Education', 'Loan_Purpose_Home', 
        'Loan_Purpose_Personal', 'Property_Area_Semiurban', 'Property_Area_Urban', 
        'Gender_Male', 'Employer_Category_Government', 'Employer_Category_MNC', 
        'Employer_Category_Private', 'Employer_Category_Unemployed'
    ]
    
    # 2. DataFrame banana zeros ke saath
    input_df = pd.DataFrame(np.zeros((1, 27)), columns=feature_names)
    
    # 3. Values fill karna
    input_df['Applicant_Income'] = income
    input_df['Coapplicant_Income'] = co_income
    input_df['Age'] = age
    input_df['Credit_Score'] = score
    input_df['DTI_Ratio'] = dti
    input_df['Loan_Amount'] = loan_amt
    input_df['Loan_Term'] = term
    input_df['Savings'] = savings
    input_df['Dependents'] = dependents
    input_df['Existing_Loans'] = existing_loans

    # 4. Predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
