import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="IoT Attack Detection", layout="wide")
st.title("🔐 IoT Attack Detection Dashboard")
st.markdown("Select a machine learning model to detect IoT network attacks in real time.")

# Model selection dropdown
model_choice = st.sidebar.selectbox("🔧 Choose a model", ["Random Forest", "SVM", "KNN", "Gradient Boosting"])

# Load selected model
model_files = {
    "Random Forest": "rf_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "Gradient Boosting": "gb_model.pkl"
}
model = joblib.load(model_files[model_choice])

# Load feature names and scaler
with open("features.json", "r") as f:
    feature_names = json.load(f)

scaler = joblib.load("scaler.pkl")

# Load dataset for testing
df_full = pd.read_csv("streamlit_test_balanced_100_fakebenign.csv")
df_full = df_full.dropna()
df = df_full[feature_names]
labels = df_full["Attack_type"]

# Scale features
df_scaled = scaler.transform(df)

# Sidebar controls
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("Rows to simulate:", 10, len(df), 25)
delay = st.sidebar.slider("Delay between predictions (sec):", 0.1, 2.0, 0.5)

# UI placeholders
tab1, tab2 = st.tabs(["📋 Detection Log", "📊 Attack Summary"])
log_placeholder = tab1.empty()
chart_placeholder = tab2.empty()

if st.button("🚀 Start Detection"):
    results = []
    predictions = []

    for i in range(row_limit):
        row = df_scaled[i].reshape(1, -1)
        pred = model.predict(row)[0]
        actual = labels.iloc[i]

        results.append({"Row": i+1, "Predicted": pred, "Actual": actual})
        predictions.append(pred)

        log_df = pd.DataFrame(results)

        # Update log table
        with tab1:
            log_placeholder.subheader("📋 Real-Time Detection Log")
            log_placeholder.dataframe(log_df, use_container_width=True)
            if pred != "Benign":
                st.warning(f"⚠️ Threat detected: {pred}")

        # Update pie chart
        with tab2:
            pie_data = pd.Series(predictions).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct="%1.1f%%", startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            chart_placeholder.subheader("📊 Attack Summary")
            chart_placeholder.pyplot(fig)

        time.sleep(delay)
