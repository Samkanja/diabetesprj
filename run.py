import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/home/kskanja/stuff/models/trained_model.sav', 'rb'))

#creating a function for prediction

def diabetesprediction(input_data):
    
    modified = np.asarray(input_data)
    reshaped_data = modified.reshape(1, -1)

    prediction = loaded_model.predict(reshaped_data)
    print(prediction)
    if (prediction[0]==0):
        return 'the person is not diabetic'
    else:
        return 'the person is diabetic'

def main():

    # giving title 
    st.title('Diabetes Prediction Web App')



    # getting input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose leve')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thinkness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # code for prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Resut'):
        diagnosis = diabetesprediction([Pregnancies, Glucose, BloodPressure,SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ =='__main__':
    main()

    
    
    
