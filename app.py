import streamlit as st
import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

# Creating a function for prediction

def diabetes_prediction(input_data):

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
       return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():

    # giving a title
    st.title('Diabetes Prediction')
   
    # getting the input data from the user       
    Pregnancies = st.text_input('Number of Pregancies: ')
    Glucose = st.text_input('Glucose Level: ')
    BloodPressure = st.text_input('BloodPressure Level: ')
    SkinThickness = st.text_input('SkinThickness Value: ')
    Insulin = st.text_input('Insulin Level: ')
    BMI = st.text_input('BMI Value: ')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value: ')
    Age = st.text_input('Age of the Person: ')

    # Code for prediction
    diagonsis = ''

    # Creating a button for Prediction 

    if st.button('Diabetes Test Result'):
        input_data = [
            float(Pregnancies), float(Glucose), float(BloodPressure),
            float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), float(Age)
        ]
        diagonsis = diabetes_prediction(input_data)

    
    st.success(diagonsis)


if __name__ == '__main__':
    main()

