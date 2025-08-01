# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. LOAD THE PRE-TRAINED MODEL AND DATA ---

# Load the saved model
# 'rb' means 'read binary', which is the mode for reading pickle files.
try:
    with open('trained_model.sav', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please run the training script first to create 'trained_model.sav'.")
    st.stop() # Stop the app if the model isn't found

# Load the dataset for getting column information and for visualizations
@st.cache_data
def load_data():
    df = pd.read_csv('Students.csv')
    return df

df = load_data()

# Define the passing score and create the 'Pass_Fail' column for visualizations
PASSING_SCORE = 70
df['Pass_Fail'] = np.where(df['Exam_Score'] >= PASSING_SCORE, 1, 0)

# We need the list of columns from the training data for the prediction part.
# This is a key step to make sure the user's input matches the model's expectations.
features = df.drop(columns=['Student_ID', 'Name', 'Exam_Score', 'Pass_Fail'])
features_encoded = pd.get_dummies(features)
training_columns = features_encoded.columns


# --- 2. STREAMLIT USER INTERFACE ---

st.title('🎓 Student Performance Prediction')

st.write(
    "This app predicts whether a student will pass or fail based on their information."
    " Please provide the student's details below."
)

# --- VISUALIZATIONS ---
st.header("Dataset Analysis")

# Create columns for the charts
chart1, chart2 = st.columns(2)

with chart1:
    # Chart 1: Pass vs. Fail Count
    st.subheader("Pass vs. Fail Distribution")
    pass_fail_counts = df['Pass_Fail'].map({1: 'Pass', 0: 'Fail'}).value_counts()
    st.bar_chart(pass_fail_counts)

with chart2:
    # Chart 2: Study Hours vs. Exam Score
    st.subheader("Study Hours vs. Exam Score")
    st.scatter_chart(df, x='Study Hours', y='Exam_Score', color='Pass_Fail')


# --- USER INPUTS FOR PREDICTION ---
st.header("Predict Student Performance")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=15, max_value=25, value=18)
    gender = st.selectbox('Gender', options=df['Gender'].unique())
    subject = st.selectbox('Subject', options=df['Subject'].unique())
    study_hours = st.slider('Study Hours per Week', min_value=0, max_value=20, value=10)

with col2:
    attendance = st.slider('Attendance Percentage', min_value=0, max_value=100, value=90)
    parental_education = st.selectbox('Parental Education Level', options=df['Parental Education Level'].unique())
    socioeconomic_status = st.selectbox('Socioeconomic Status', options=df['Socioeconomic Status'].unique())

# Prediction button
if st.button('**Predict**'):
    # --- 3. PREDICTION LOGIC (Corrected) ---

    # Create a single-row DataFrame with the user's input.
    # The column names MUST match the original feature names before encoding.
    user_input = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Subject': subject,
        'Study Hours': study_hours,
        'Attendance_Percentage': attendance,
        'Parental Education Level': parental_education,
        'Socioeconomic Status': socioeconomic_status
    }])

    # One-hot encode the user's input.
    user_input_encoded = pd.get_dummies(user_input)

    # Align the user input columns with the training columns.
    # This is a crucial step to ensure the model receives the exact same feature structure it was trained on.
    final_input = user_input_encoded.reindex(columns=training_columns, fill_value=0)

    # Make the prediction using the loaded model
    prediction = model.predict(final_input)

    # Display the result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success('**The student is likely to PASS!** 🎉')
    else:
        st.error('**The student is likely to FAIL.** 😔')
