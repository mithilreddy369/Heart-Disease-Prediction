import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
with open('model_and_scaler.pkl', 'rb') as file:
    data = pickle.load(file)
    loaded_model = data['model']
    scaler = data['scaler']

# Define a function to handle user input and prediction
def predict_outcome(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Prepare input dictionary
    input_data = {
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Apply one-hot encoding to match the training data's structure
    input_df = pd.get_dummies(input_df, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
    
    # Define the expected columns
    expected_columns = [
        'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
        'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0', 'restecg_1', 'restecg_2',
        'slope_0', 'slope_1', 'slope_2', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4',
        'thal_0', 'thal_1', 'thal_2', 'thal_3'
    ]
    
    # Reindex to ensure all columns are present
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = loaded_model.predict(input_scaled)
    
    return prediction[0]

# Streamlit app
st.title("Heart Disease Prediction App")

st.write("""
### Enter the patient details to predict the likelihood of heart disease.
""")

# Create input fields with descriptive labels
age = st.number_input("Age", min_value=1, max_value=120, value=58)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x+1}")
trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=200, value=100)
chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=600, value=248)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], format_func=lambda x: f"Type {x}")
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=122)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest (Oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: f"Type {x+1}")
ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0-3)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

# Prediction button
if st.button("Predict"):
    prediction = predict_outcome(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if prediction == 1:
        st.write("The patient is likely to have heart disease.")
    else:
        st.write("The patient is not likely to have heart disease.")
