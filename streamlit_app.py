import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt

# Set up page layout
st.set_page_config(page_title="IoT Attack Detection", layout="wide")

# Title and intro
st.title("🔐 IoT Attack Detection Dashboard")
st.markdown("This dashboard simulates real-time detection of network attacks using a trained machine learning model.")

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Load feature names used during training
with open("features.json", "r") as f:
    training_features = json.load(f)

# Load dataset and ensure exact feature match
df_full = pd.read_csv("RT_IOT2022_small.csv")

# Match only the columns used in training
try:
    df = df_full[training_features]
except KeyError:
    st.error("❌ Error: Columns in dataset don't match training features.")
    st.stop()

labels = df_full["Attack_type"]

# Sidebar controls
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("Select how many rows to simulate", 10, 100, 25)
delay = st.sidebar.slider("Delay between each prediction (seconds)", 0.1, 2.0, 0.5)

# Tabs for output
tab1, tab2 = st.tabs(["📋 Detection Log", "📊 Attack Summary"])

if st.button("🚀 Start Simulation"):
    result_log = []
    chart_data = []

    for i in range(row_limit):
        try:
            row = df.iloc[i].values.reshape(1, -1)
            prediction = model.predict(row)[0]
            actual = labels.iloc[i]
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            break

        result_log.append({
            "Row": i + 1,
            "Predicted Attack": prediction,
            "Actual Attack": actual
        })
        chart_data.append(prediction)

        result_df = pd.DataFrame(result_log)

        with tab1:
            st.subheader("📋 Real-Time Prediction Log")
            st.dataframe(result_df, use_container_width=True)
            if prediction != "Benign":
                st.warning(f"⚠️ Possible Threat Detected: {prediction}")

        with tab2:
            st.subheader("📊 Pie Chart of Detected Attacks")
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            st.pyplot(fig)

        time.sleep(delay)
