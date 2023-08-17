import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import joblib





model = joblib.load('final_model.sav')

st.title('Will Customer Churn Or Not?')
gender = st.radio("Choose Sex", ['Male', 'Female']) 
SeniorCitizen = st.select_slider("Is SeniorCitizen?", [0, 1])
Partner  = st.select_slider("Have Partner?", ['Yes', 'No'])
Dependents = st.select_slider("Is Dependents?", ['Yes','No'])
tenure = st.number_input("Input tenure",0,100)
PhoneService = st.select_slider("Have PhoneService?", ["Yes", "No"])
MultipleLines = st.selectbox("Have MultipleLines?", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Type of InternetService", ['DSL', 'Fiber optic', 'No']) 
OnlineSecurity = st.selectbox("Have OnlineSecurity", ["Yes", "No", "No phone service"])
OnlineBackup = st.selectbox("Have OnlineBackup", ["Yes", "No", "No phone service"]) 
DeviceProtection = st.selectbox("Have DeviceProtection", ["Yes", "No", "No phone service"]) 
TechSupport = st.selectbox("Have TechSupport", ["Yes", "No", "No phone service"])
StreamingTV = st.selectbox("Have StreamingTV", ["Yes", "No", "No phone service"]) 
StreamingMovies = st.selectbox("Have StreamingMovies", ["Yes", "No", "No phone service"])
Contract = st.selectbox("Type of Contract", ['Month-to-month', 'One year', 'Two year']) 
PaperlessBilling = st.select_slider("Have PaperlessBilling", ["Yes", "No"]) 
PaymentMethod = st.selectbox("Type of PaymentMethod", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)']) 
MonthlyCharges = st.number_input("Input MonthlyCharges",0,150)
TotalCharges = st.number_input("Input TotalCharges",0,9000)

cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'PaperlessBilling']

columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']

def perocess(): 
    row = np.array([gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,
                    OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
                    Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges]) 
    X = pd.DataFrame([row], columns = columns)
    X = X.replace("No internet service","No")
    X = X.replace("No phone service","No")
    for col in cols:
        X[col] = X[col].replace({"No":0, "Yes":1})
    X['gender'] = X['gender'].replace({"Female":0, "Male":1})
    X['Contract'] = X['Contract'].replace({'Month-to-month':0, 'One year':1, 'Two year':2})
    X['PaymentMethod'] = X['PaymentMethod'].replace({'Electronic check':0, 'Mailed check':1, 'Bank transfer (automatic)':2,
                                                     'Credit card (automatic)':3})
    X['InternetService'] = X['InternetService'].replace({'DSL':0, 'Fiber optic':1, 'No':2})
    cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
    scaler = MinMaxScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    return X

def predict():
    X = perocess()
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Customer Churn :thumbsup:')
    else: 
        st.error('Customer Not Churn :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)
