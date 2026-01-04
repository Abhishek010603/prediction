import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction App")

glucose = st.number_input("Glucose")
bmi = st.number_input("BMI")
age = st.number_input("Age")

if st.button("Predict"):
    data = np.array([[glucose, bmi, age]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    if result[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
