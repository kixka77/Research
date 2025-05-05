import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# File to save predictions
SAVE_FILE = "prediction_history.csv"

# Load and preprocess existing data if available
def load_history():
    if os.path.exists(SAVE_FILE):
        return pd.read_csv(SAVE_FILE)
    return pd.DataFrame(columns=[
        "Study Hours", "Sleep Hours", "Class Participation", "Assignment Completion",
        "Social Media Hours", "Health Rating", "Predicted Category", "Feedback"
    ])

def save_prediction(data):
    history = load_history()
    updated = pd.concat([history, pd.DataFrame([data])], ignore_index=True)
    updated.to_csv(SAVE_FILE, index=False)

# Generate synthetic training data
np.random.seed(42)
X_synthetic = pd.DataFrame({
    "study_hours": np.random.randint(0, 24, 200),
    "sleep_hours": np.random.randint(3, 10, 200),
    "participation": np.random.randint(1, 6, 200),
    "assignments": np.random.randint(1, 6, 200),
    "social_media": np.random.randint(0, 10, 200),
    "health": np.random.randint(1, 6, 200),
})
y_synthetic = np.random.choice(["Excellent", "Satisfactory", "Needs Improvement", "At Risk"], 200)

X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

# Scale and fit model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = GradientBoostingClassifier()
model.fit(X_train_scaled, y_train)

# Evaluate on test set
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Streamlit UI
st.title("Academic Performance Predictor with Feedback & Save Function")

with st.form("predict_form"):
    study_hours = st.slider("Study hours per day", 0, 24, 2)
    sleep_hours = st.slider("Sleep hours per day", 0, 12, 6)
    participation = st.slider("Class participation (1-5)", 1, 5, 3)
    assignments = st.slider("Assignment completion (1-5)", 1, 5, 4)
    social_media = st.slider("Social media hours per day", 0, 24, 2)
    health = st.slider("Self-rated health (1-5)", 1, 5, 3)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "participation": participation,
        "assignments": assignments,
        "social_media": social_media,
        "health": health,
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    feedback = {
        "Excellent": "Great job! Keep up the consistent effort.",
        "Satisfactory": "You're doing okay, but there's room to grow.",
        "Needs Improvement": "Try adjusting your habits to improve outcomes.",
        "At Risk": "Seek support and revise your study strategies."
    }[prediction]

    st.success(f"Prediction: {prediction}")
    st.info(f"Feedback: {feedback}")

    save_prediction({
        "Study Hours": study_hours,
        "Sleep Hours": sleep_hours,
        "Class Participation": participation,
        "Assignment Completion": assignments,
        "Social Media Hours": social_media,
        "Health Rating": health,
        "Predicted Category": prediction,
        "Feedback": feedback
    })

    st.markdown("### Evaluation Metrics (for internal validation)")
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

    if st.checkbox("Show Prediction History"):
        st.dataframe(load_history())
