import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt

# Page layout
st.set_page_config(page_title="IoT Attack Detection", layout="wide")

# App title
st.title("🔐 IoT Attack Detection Dashboard")
st.markdown("Simulated attack detection on IoT network traffic using a trained machine learning model.")

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Load feature names used during training
with open("features.json", "r") as f:
    training_features = json.load(f)

# Load dataset and prepare features and labels
df_full = pd.read_csv("RT_IOT2022_small.csv")
df = df_full[training_features]              # exact match
labels = df_full["Attack_type"]

# Sidebar controls
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("Select how many data rows to simulate:", 10, 100, 25)
delay = st.sidebar.slider("Delay between predictions (seconds):", 0.1, 2.0, 0.5)

# Tabs for layout
tab1, tab2 = st.tabs(["📋 Detection Log", "📊 Attack Summary"])

if st.button("🚀 Start Simulation"):
    result_log = []
    chart_data = []

    for i in range(row_limit):
        row = df.iloc[i].values.reshape(1, -1)
        prediction = model.predict(row)[0]
        actual = labels.iloc[i]

        result_log.append({
            "Row #": i + 1,
            "Predicted Attack": prediction,
            "Actual Attack": actual
        })
        chart_data.append(prediction)

        result_df = pd.DataFrame(result_log)

        with tab1:
            st.subheader("📋 Real-Time Prediction Log")
            st.dataframe(result_df, use_container_width=True)
            if prediction != "Benign":
                st.warning(f"⚠️ Alert: {prediction} attack detected!")

        with tab2:
            st.subheader("📊 Pie Chart of Detected Attacks")
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            st.pyplot(fig)

        time.sleep(delay)
