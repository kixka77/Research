import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Dense

# Sample data
data = pd.DataFrame({
    'study_hours': [2, 4, 6, 8, 10],
    'sleep_hours': [6, 7, 5, 6, 8],
    'attendance': [60, 70, 80, 90, 95],
    'score_category': ['At Risk', 'Needs Improvement', 'Satisfactory', 'Excellent', 'Excellent']
})

# Encode target labels
label_encoder = LabelEncoder()
data['score_category_num'] = label_encoder.fit_transform(data['score_category'])

X = data[['study_hours', 'sleep_hours', 'attendance']]
y = data['score_category_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_scaled, y_train)

# Deep Neural Network Model
dnn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])
dnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

# Streamlit UI
st.title("Academic Performance Predictor")

study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=2.0)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=6.0)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)

if st.button("Predict"):
    user_input = np.array([[study_hours, sleep_hours, attendance]])
    user_scaled = scaler.transform(user_input)

    gb_pred = gb_model.predict(user_scaled)
    dnn_pred = np.argmax(dnn_model.predict(user_scaled), axis=1)

    final_pred = gb_pred[0] if gb_pred[0] == dnn_pred[0] else dnn_pred[0]
    category = label_encoder.inverse_transform([final_pred])[0]

    st.subheader(f"Prediction: **{category}**")

    # Feedback based on prediction
    feedback = {
        "At Risk": "You're currently at risk. Consider improving your study habits and attendance.",
        "Needs Improvement": "You're making some effort, but there's still room for improvement in consistency.",
        "Satisfactory": "You're doing well. Keep maintaining your current habits and strive for excellence!",
        "Excellent": "Great job! You're performing excellently. Just stay consistent and motivated!"
    }
    st.info(feedback[category])

    # Evaluation Metrics
    y_pred = gb_model.predict(X_test_scaled)
    st.write("### Model Evaluation (Gradient Boosting)")
    st.write(f"- Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"- Precision: {precision_score(y_test, y_pred, average='macro'):.2f}")
    st.write(f"- Recall: {recall_score(y_test, y_pred, average='macro'):.2f}")
    st.write(f"- F1 Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
