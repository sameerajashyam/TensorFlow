import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model("insurance_model.h5")

# Define the column transformer (same one used during training)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Sample data to fit the transformer with all possible categories
sample_data = pd.DataFrame({
    'age': [25],
    'sex': ['male'],
    'bmi': [25.0],
    'children': [1],
    'smoker': ['no'],
    'region': ['southwest']
})
ct.fit(sample_data)

# Streamlit app UI
st.title("Insurance Charges Predictor")

age = st.slider("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.selectbox("Number of children", list(range(0, 6)))
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Make prediction
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Fit again with all possible categories before transforming
full_sample = pd.DataFrame({
    'age': [25] * 4,
    'sex': ['male', 'female', 'male', 'female'],
    'bmi': [25.0] * 4,
    'children': [1] * 4,
    'smoker': ['yes', 'no', 'yes', 'no'],
    'region': ['southwest', 'southeast', 'northeast', 'northwest']
})
ct.fit(full_sample)

input_transformed = ct.transform(input_df)

if st.button("Predict Insurance Charge"):
    prediction = model.predict(input_transformed)
    st.success(f"Estimated Insurance Charge: ${prediction[0][0]:,.2f}")
