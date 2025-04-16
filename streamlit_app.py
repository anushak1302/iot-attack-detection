import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

# Page layout
st.set_page_config(page_title="IoT Attack Detection", layout="wide")

# App title
st.title("🔐 IoT Attack Detection (Simulated Dashboard)")
st.markdown("This app simulates real-time detection of network attacks using a trained machine learning model.")

# Load model and dataset
model = joblib.load("random_forest_model.pkl")
df = pd.read_csv("RT_IOT2022_small.csv")

# Use same features as model was trained on (fixing shape mismatch error)
df = df.drop(columns=["Attack_type", "no", "proto", "service"])
features = df.iloc[:, :76]  # Limit to first 76 columns to match training
labels = pd.read_csv("RT_IOT2022_small.csv")["Attack_type"]

# Sidebar controls (layman-friendly labels)
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("Select how many data rows to simulate:", 10, 100, 25)
delay = st.sidebar.slider("Select delay between each prediction (seconds):", 0.1, 2.0, 0.5)

# Tabs
tab1, tab2 = st.tabs(["🔍 Detection Log", "📊 Attack Summary"])

if st.button("🚀 Start Simulation"):
    result_log = []
    chart_data = []

    for i in range(row_limit):
        row = features.iloc[i].values.reshape(1, -1)
        prediction = model.predict(row)[0]
        actual = labels.iloc[i]

        result_log.append({"Index": i+1, "Predicted Attack": prediction, "Actual Attack": actual})
        chart_data.append(prediction)

        result_df = pd.DataFrame(result_log)

        # Tab 1: Live predictions
        with tab1:
            st.subheader("📋 Real-time Prediction Log")
            st.dataframe(result_df, use_container_width=True)
            if prediction != "Benign":
                st.warning(f"⚠️ Possible Threat Detected: {prediction}")

        # Tab 2: Pie chart of current detection summary
        with tab2:
            st.subheader("📊 Detected Attack Types")
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            st.pyplot(fig)

        time.sleep(delay)
