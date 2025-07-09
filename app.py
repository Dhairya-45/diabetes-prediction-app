import streamlit as st
import numpy as np
import pickle

# Load the model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data)
    return '🟢 The person is **not diabetic**.' if prediction[0] == 0 else '🔴 The person is **diabetic**.'

# Streamlit app
def main():
    # Page config
    st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

    # Custom CSS for background and box styling
    st.markdown("""
        <style>
            .main {
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 10px;
            }
            .result-box {
                padding: 20px;
                background-color: #ffffff;
                border-left: 5px solid #4B8BBE;
                border-radius: 8px;
                margin-top: 20px;
                font-size: 18px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter patient data below to predict diabetes using a trained ML model.</p>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.number_input('Pregnancies', min_value=0)
            Glucose = st.number_input('Glucose Level', min_value=0)
            BloodPressure = st.number_input('Blood Pressure', min_value=0)
            SkinThickness = st.number_input('Skin Thickness', min_value=0)

        with col2:
            Insulin = st.number_input('Insulin Level', min_value=0)
            BMI = st.number_input('BMI', min_value=0.0, format="%.2f")
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f")
            Age = st.number_input('Age', min_value=1)

        submitted = st.form_submit_button("🔍 Predict Diabetes")

    if submitted:
        input_data = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]
        result = diabetes_prediction(input_data)
        st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)

    # Optional sidebar
    st.sidebar.markdown("## 🧭 Navigation")
    st.sidebar.info("This app uses a machine learning model to predict diabetes based on user input.")

if __name__ == '__main__':
    main()
