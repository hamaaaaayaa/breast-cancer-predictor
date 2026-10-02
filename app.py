import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=['target'])

st.title('Breast Cancer Diagnosis Predictor')
st.write('Enter tumor measurements to predict malignant vs. benign.')

values = []
for col in X.columns:
    values.append(st.number_input(col, value=float(X[col].mean())))

if st.button('Predict'):
    scaled = scaler.transform(np.array([values]))
    pred = model.predict(scaled)[0]
    if pred == 1:
        st.success('Prediction: Benign')
    else:
        st.error('Prediction: Malignant')
    st.caption('Educational project only, not a medical diagnosis.')
