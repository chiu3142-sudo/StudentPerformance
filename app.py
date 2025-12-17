
import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

st.title("🎓 Student Performance Prediction")
st.write("Enter the input and hit predict to get estimated final marks")

# Inputs (same style as house project)

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0
)

past_score = st.number_input(
    "Past Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

family_support = st.number_input(
    "Family Support (1 = Yes, 0 = No)",
    min_value=0,
    max_value=1,
    value=1,
    step=1
)

extra_classes = st.number_input(
    "Extra Classes per Week",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

# Predict
if st.button("Predict Performance"):
    x = np.array([[study_hours, attendance, past_score, family_support, extra_classes]])
    pred = model.predict(x)[0]

    st.success(f"Estimated Final Score: {pred:.2f}")
