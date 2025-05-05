import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

# Expanded sample dataset
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'sleep_hours': [4, 5, 6, 7, 8, 6, 7, 8],
    'attendance': [60, 70, 80, 85, 90, 95, 100, 100],
    'participation': [2, 3, 3, 4, 4, 5, 5, 5],
    'assignments': [2, 3, 3, 4, 4, 5, 5, 5],
    'social_media': [6, 5, 4, 3, 2, 2, 1, 1],
    'health': [2, 3, 3, 4, 4, 5, 5, 5],
    'performance': ['At Risk', 'Needs Improvement', 'Needs Improvement', 'Satisfactory',
                    'Satisfactory', 'Excellent', 'Excellent', 'Excellent']
}
df = pd.DataFrame(data)

# Features and label
X = df.drop('performance', axis=1)
y = df['performance']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Enhanced Academic Performance Predictor")

with st.form("user_input"):
    st.subheader("Enter Your Details Accurately")
    study_hours = st.number_input("Daily Study Hours", min_value=0.0, max_value=24.0, step=0.5)
    sleep_hours = st.number_input("Daily Sleep Hours", min_value=0.0, max_value=24.0, step=0.5)
    attendance = st.slider("Class Attendance (%)", 0, 100, 80)
    participation = st.slider("Class Participation (1–5)", 1, 5, 3)
    assignments = st.slider("Assignment Completion (1–5)", 1, 5, 4)
    social_media = st.slider("Social Media Hours per Day", 0, 24, 3)
    health = st.slider("Self-Rated Health (1–5)", 1, 5, 3)
    submitted = st.form_submit_button("Predict")

if submitted:
    user_data = pd.DataFrame([[study_hours, sleep_hours, attendance, participation, assignments, social_media, health]],
                             columns=X.columns)
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]

    feedback = {
        'At Risk': "Your performance is at risk. Reduce distractions and focus on consistent habits.",
        'Needs Improvement': "You're improving, but need more consistency in study and class engagement.",
        'Satisfactory': "Good work. Keep refining your balance and aim higher.",
        'Excellent': "Outstanding! Your habits are well aligned with academic success."
    }

    st.subheader("Prediction Result")
    st.success(f"Predicted Performance: {prediction}")
    st.info(feedback[prediction])

    # Evaluation metrics
    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")

    # Save to history
    history_row = pd.DataFrame([[
        study_hours, sleep_hours, attendance, participation, assignments, social_media, health, prediction
    ]], columns=['study_hours', 'sleep_hours', 'attendance', 'participation',
                 'assignments', 'social_media', 'health', 'prediction'])

    if os.path.exists("prediction_history.csv"):
        history = pd.read_csv("prediction_history.csv")
        history = pd.concat([history, history_row], ignore_index=True)
    else:
        history = history_row
    history.to_csv("prediction_history.csv", index=False)

    st.subheader("Prediction History (Latest Entries)")
    st.dataframe(history.tail(5))
