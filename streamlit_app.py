import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt

# Set up Streamlit page
st.set_page_config(page_title="IoT Attack Detection", layout="wide")
st.title("🔐 IoT Attack Detection Dashboard")
st.markdown("This app simulates real-time detection of IoT network attacks using a Random Forest model.")

# Load trained model (fast version)
model = joblib.load("new_model_all_features.pkl")

# Load the features used for training
with open("features.json", "r") as f:
    training_features = json.load(f)

# Load dataset (can switch to 'RT_IOT2022.csv' if you prefer full)
df_full = pd.read_csv("RT_IOT2022_small.csv")  # Or use the full version
df = df_full[training_features]
labels = df_full["Attack_type"]

# Sidebar for simulation settings
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("How many rows to simulate?", 10, 100, 25)
delay = st.sidebar.slider("Delay between predictions (seconds)", 0.1, 2.0, 0.5)

# Tabs for log and summary
tab1, tab2 = st.tabs(["📋 Detection Log", "📊 Attack Summary"])

# Start simulation
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
            "Predicted": prediction,
            "Actual": actual
        })
        chart_data.append(prediction)

        result_df = pd.DataFrame(result_log)

        # Detection log tab
        with tab1:
            st.subheader("📋 Real-Time Detection Log")
            st.dataframe(result_df, use_container_width=True)
            if prediction != "Benign":
                st.warning(f"⚠️ Alert: {prediction} attack detected!")

        # Pie chart summary tab
        with tab2:
            st.subheader("📊 Attack Summary")
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            st.pyplot(fig)

        time.sleep(delay)
