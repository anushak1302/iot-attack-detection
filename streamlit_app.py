import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="IoT Intrusion Detection", layout="wide")

# Title
st.title("🔐 IoT Intrusion Detection Dashboard")
st.markdown("Simulated attack detection using a trained Gradient Boosting model on IoT data.")

# Load model
model = joblib.load("random_forest_model.pkl")

# Load dataset (use the smaller file for GitHub deployment)
df = pd.read_csv("RT_IOT2022_small.csv")

# Preprocess
features = df.drop(columns=["Attack_type", "no", "proto", "service"])
labels = df["Attack_type"]

# Sidebar controls
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("Number of simulated rows", 10, 100, 25)
delay = st.sidebar.slider("Delay per row (sec)", 0.1, 2.0, 0.5)

# Tabs
tab1, tab2 = st.tabs(["🧪 Live Detection Log", "📊 Attack Type Pie Chart"])
start_simulation = st.button("🚀 Start Simulation")

if start_simulation:
    result_log = []
    chart_data = []

    for i in range(row_limit):
        row = features.iloc[i].values.reshape(1, -1)
        prediction = model.predict(row)[0]
        actual = labels.iloc[i]

        result_log.append({"Index": i, "Predicted": prediction, "Actual": actual})
        chart_data.append(prediction)
        result_df = pd.DataFrame(result_log)

        # Tab 1: Log view
        with tab1:
            st.subheader("📋 Prediction Log")
            st.dataframe(result_df, use_container_width=True)
            if prediction != "Benign":
                st.error(f"⚠️ ALERT: {prediction} attack detected!")

        # Tab 2: Pie chart view
        with tab2:
            st.subheader("📊 Pie Chart of Detected Attack Types")
            pie_df = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_df.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            st.pyplot(fig)

        time.sleep(delay)
