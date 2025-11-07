import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder # Imports simplified

# --- Load Preprocessors and Model ---
model = tf.keras.models.load_model('model/churn_model.keras', compile=False)


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
    
with open('ohe_geography.pkl', 'rb') as f:
    ohe_geography = pickle.load(f)

# --- CRITICAL: Define Expected Feature Order ---
EXPECTED_COLUMNS = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
    'Geography_France', 'Geography_Germany', 'Geography_Spain' 
]

# --- Streamlit App UI ---
st.title('Customer Churn Prediction')

with st.sidebar:
    st.header("Customer Details")
    geography = st.selectbox('Geography', ohe_geography.categories_[0])
    gender = st.selectbox('Gender', le_gender.classes_)
    age = st.slider('Age', min_value=18, max_value=100)
    tenure = st.number_input('Tenure', min_value=0, max_value=10)
    balance = st.number_input('Balance', min_value=0.0, format='%.2f')
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
    products_number = st.slider('Products Number', min_value=0, max_value=10)
    credit_card = st.selectbox('Has Credit Card?', ['0', '1'])
    active_member = st.selectbox('Is Active Member?', ['0', '1'])
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format='%.2f')


# --- Data Preparation and Prediction ---

if st.button('Predict Churn Probability'):
    
    # Create Initial DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography], 
        'Gender': [gender],       
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products_number],
        'HasCrCard': [int(credit_card)], 
        'IsActiveMember': [int(active_member)],
        'EstimatedSalary': [estimated_salary]
    })
    
    # Label Encode 'Gender'
    input_data['Gender'] = le_gender.transform(input_data[['Gender']].values.reshape(-1, 1)).flatten()

    # One-Hot Encode 'Geography' (Fixes column naming error)
    geography_encoded = ohe_geography.transform(input_data[['Geography']])
    
    geography_df = pd.DataFrame(
        geography_encoded, 
        columns=ohe_geography.get_feature_names_out(['Geography']),
        index=input_data.index
    )

    # Combine DataFrame
    input_data = input_data.drop('Geography', axis=1)
    input_data = pd.concat([input_data, geography_df], axis=1)

    # Reorder and Scale (Fixes ValueError: Feature names mismatch)
    input_data_reordered = input_data.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    input_data_scaled = scaler.transform(input_data_reordered.values)

    # Make Prediction
    prediction = model.predict(input_data_scaled, verbose=0)
    prediction_proba = prediction[0][0]

    # Display Result
    st.subheader('Prediction Result')
    churn_probability = prediction_proba * 100
    st.metric(label="Churn Probability", value=f"{churn_probability:.2f}%")
    
    if prediction_proba > 0.5:
        st.error(f'The customer is **likely to churn** (Probability: {churn_probability:.2f}%)')
    else:
        st.success(f'The customer is **unlikely to churn** (Probability: {churn_probability:.2f}%)')