import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Example dataset (you should replace this with your actual structured data)
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'sleep_hours': [4, 5, 6, 7, 8, 6, 7, 8],
    'attendance': [60, 70, 80, 85, 90, 95, 100, 100],
    'performance': ['At Risk', 'Needs Improvement', 'Needs Improvement', 'Satisfactory',
                    'Satisfactory', 'Excellent', 'Excellent', 'Excellent']
}
df = pd.DataFrame(data)

# Preprocess data
X = df.drop('performance', axis=1)
y = df['performance']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split and model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# App title
st.title("Academic Performance Predictor")

# User input form
with st.form("user_input"):
    st.subheader("Enter your details")
    study_hours = st.number_input("Daily Study Hours", min_value=0.0, step=0.5)
    sleep_hours = st.number_input("Daily Sleep Hours", min_value=0.0, step=0.5)
    attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, step=1)
    submitted = st.form_submit_button("Predict")

# Make prediction
if submitted:
    user_data = pd.DataFrame([[study_hours, sleep_hours, attendance]], columns=X.columns)
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]

    # Feedback based on prediction
    feedback = {
        'At Risk': "Your performance is currently at risk. It's important to improve study habits and maintain consistent attendance.",
        'Needs Improvement': "Your performance shows room for improvement. Consider adjusting your routine and reviewing study strategies.",
        'Satisfactory': "You're doing fine, but there's always room to aim higher. Keep consistent.",
        'Excellent': "Great job! Your habits reflect excellent academic discipline. Keep it up!"
    }

    st.subheader("Prediction Result")
    st.success(f"Predicted Performance: {prediction}")
    st.info(feedback[prediction])

    # Evaluation metrics
    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

    # Save to history
    history_row = pd.DataFrame([[study_hours, sleep_hours, attendance, prediction]], columns=['study_hours', 'sleep_hours', 'attendance', 'prediction'])
    if os.path.exists("prediction_history.csv"):
        history = pd.read_csv("prediction_history.csv")
        history = pd.concat([history, history_row], ignore_index=True)
    else:
        history = history_row
    history.to_csv("prediction_history.csv", index=False)

    st.subheader("Prediction History (latest entries)")
    st.dataframe(history.tail(5))
