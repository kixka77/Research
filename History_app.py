import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = pd.read_csv("student_data.csv")
    return data

data = load_data()
X = data.drop("Performance", axis=1)
y = data["Performance"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save the model and encoder
joblib.dump(model, "gb_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# Evaluation metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Streamlit UI
st.title("Student Performance Prediction")

# Input fields
def user_input():
    inputs = {}
    for col in X.columns:
        dtype = X[col].dtype
        if np.issubdtype(dtype, np.number):
            inputs[col] = st.number_input(col, min_value=0.0, format="%.2f")
        else:
            inputs[col] = st.text_input(col)
    return pd.DataFrame([inputs])

input_df = user_input()

if st.button("Predict Performance"):
    if not input_df.empty:
        # Prediction
        model = joblib.load("gb_model.pkl")
        le = joblib.load("label_encoder.pkl")
        pred_encoded = model.predict(input_df)
        pred_label = le.inverse_transform(pred_encoded)[0]
        
        # Show prediction
        st.subheader("Prediction Result")
        st.write(f"Predicted Performance: **{pred_label}**")
        
        # Show evaluation metrics
        st.subheader("Model Evaluation Metrics")
        st.write(f"Accuracy: **{accuracy:.2f}**")
        st.write(f"Precision: **{precision:.2f}**")
        st.write(f"Recall: **{recall:.2f}**")
        st.write(f"F1 Score: **{f1:.2f}**")
        
        # Recommendation
        st.subheader("Recommendation")
        if pred_label == "At Risk":
            st.warning("Recommendation: Seek academic support and improve time management.")
        elif pred_label == "Needs Improvement":
            st.info("Recommendation: Focus more on studies and avoid distractions.")
        elif pred_label == "Satisfactory":
            st.success("Recommendation: Maintain consistent study habits.")
        else:
            st.balloons()
            st.success("Recommendation: Excellent work! Keep it up.")
        
        # Save prediction history
        history_path = "prediction_history.csv"
        input_df["Prediction"] = pred_label
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            history_df = pd.concat([history_df, input_df], ignore_index=True)
        else:
            history_df = input_df
        history_df.to_csv(history_path, index=False)
    else:
        st.error("Please fill in all input fields.")

# Display history
st.subheader("Prediction History")
if os.path.exists("prediction_history.csv"):
    hist_df = pd.read_csv("prediction_history.csv")
    st.dataframe(hist_df)

    # Download button
    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download History", data=csv, file_name="student_prediction_history.csv", mime="text/csv")

    # Clear history
    if st.button("Clear Prediction History"):
        os.remove("prediction_history.csv")
        st.success("Prediction history cleared.")
else:
    st.info("No prediction history found.")
