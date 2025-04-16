import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt

# Setup Streamlit page
st.set_page_config(page_title="IoT Attack Detection", layout="wide")
st.title("🔐 IoT Attack Detection Dashboard")
st.markdown("Simulating real-time detection of IoT network attacks using a Random Forest model.")

# Load trained model and feature list
model = joblib.load("new_model_all_features.pkl")

with open("features.json", "r") as f:
    training_features = json.load(f)

# Load the balanced test dataset with fake benign traffic
df_full = pd.read_csv("streamlit_test_balanced_100_fakebenign.csv")
df_full = df_full.dropna()

# Prepare data
df = df_full[training_features]
labels = df_full["Attack_type"]

# Sidebar settings
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("How many rows to simulate?", 10, 100, 25)
delay = st.sidebar.slider("Delay between predictions (seconds)", 0.1, 2.0, 0.5)

# Tabs and placeholders
tab1, tab2 = st.tabs(["📋 Detection Log", "📊 Attack Summary"])
log_placeholder = tab1.empty()
chart_placeholder = tab2.empty()

# Simulation button
if st.button("🚀 Start Simulation"):
    result_log = []
    chart_data = []

    for i in range(min(row_limit, len(df))):
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

        # Update table
        with tab1:
            log_placeholder.subheader("📋 Real-Time Detection Log")
            log_placeholder.dataframe(result_df, use_container_width=True)
            if prediction != "Benign":
                st.warning(f"⚠️ Alert: {prediction} attack detected!")

        # Update pie chart
        with tab2:
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            chart_placeholder.subheader("📊 Attack Summary")
            chart_placeholder.pyplot(fig)

        time.sleep(delay)
