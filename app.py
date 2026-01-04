import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("🩺 Diabetes Prediction App")

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("❌ Diabetic")
    else:
        st.success("✅ Not Diabetic")

