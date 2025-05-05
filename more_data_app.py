import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import datetime

st.set_page_config(page_title="Academic Performance Predictor", layout="centered")

# Extended dataset
data = pd.DataFrame([
    [8, 9, 10, 9, 10, 'Excellent'],
    [7, 8, 9, 8, 9, 'Excellent'],
    [6, 6, 7, 6, 7, 'Satisfactory'],
    [5, 5, 6, 5, 6, 'Satisfactory'],
    [4, 5, 5, 4, 5, 'Needs Improvement'],
    [3, 4, 4, 3, 4, 'Needs Improvement'],
    [2, 2, 3, 2, 3, 'At Risk'],
    [1, 1, 2, 1, 2, 'At Risk'],
    [7, 9, 8, 7, 8, 'Excellent'],
    [6, 7, 6, 7, 6, 'Satisfactory'],
    [5, 6, 5, 5, 5, 'Needs Improvement'],
    [2, 3, 2, 2, 2, 'At Risk'],
    [8, 8, 8, 9, 9, 'Excellent'],
    [6, 6, 6, 5, 6, 'Satisfactory'],
    [3, 3, 4, 3, 3, 'Needs Improvement'],
    [1, 2, 2, 2, 1, 'At Risk']
], columns=['study_hours', 'attendance', 'participation', 'assignments', 'quizzes', 'performance'])

# Inputs
st.title("Academic Performance Prediction")
study_hours = st.slider("Study Hours (0–10)", 0, 10)
attendance = st.slider("Attendance (0–10)", 0, 10)
participation = st.slider("Class Participation (0–10)", 0, 10)
assignments = st.slider("Assignment Score (0–10)", 0, 10)
quizzes = st.slider("Quiz Score (0–10)", 0, 10)

# Button to trigger prediction
if st.button("Predict"):
    # Encode labels
    label_map = {'Excellent': 3, 'Satisfactory': 2, 'Needs Improvement': 1, 'At Risk': 0}
    data['performance'] = data['performance'].map(label_map)

    X = data.drop('performance', axis=1)
    y = data['performance']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier()
    model.fit(X_train_scaled, y_train)

    # Prediction
    user_input = pd.DataFrame([[study_hours, attendance, participation, assignments, quizzes]],
                              columns=X.columns)
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    reverse_map = {v: k for k, v in label_map.items()}
    prediction_label = reverse_map[prediction]

    st.subheader("Predicted Performance:")
    st.success(prediction_label)

    # Recommendation
    feedback = {
        "Excellent": "Keep up the great work! Continue practicing good study habits.",
        "Satisfactory": "You're doing okay, but there's room to grow. Focus on consistent study time.",
        "Needs Improvement": "Try increasing your engagement and review areas where you're struggling.",
        "At Risk": "Consider seeking support from tutors or mentors and managing your time more effectively."
    }
    st.info("Recommendation: " + feedback[prediction_label])

    # Evaluation Metrics
    y_pred = model.predict(X_test_scaled)
    st.subheader("Model Evaluation Metrics:")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    st.write("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 2))
    st.write("Recall:", round(recall_score(y_test, y_pred, average='weighted'), 2))
    st.write("F1 Score:", round(f1_score(y_test, y_pred, average='weighted'), 2))

    # Optional: save prediction history
    log = pd.DataFrame({
        'Timestamp': [datetime.datetime.now()],
        'Study Hours': [study_hours],
        'Attendance': [attendance],
        'Participation': [participation],
        'Assignments': [assignments],
        'Quizzes': [quizzes],
        'Prediction': [prediction_label]
    })
    try:
        existing_log = pd.read_csv("prediction_log.csv")
        log = pd.concat([existing_log, log], ignore_index=True)
    except FileNotFoundError:
        pass
    log.to_csv("prediction_log.csv", index=False)
