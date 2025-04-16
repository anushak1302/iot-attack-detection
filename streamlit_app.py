import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="IoT Attack Detection", layout="wide")
st.title("🔐 IoT Attack Detection Dashboard")
st.markdown("This app simulates real-time detection of IoT network attacks using a Random Forest model.")

# Load model and features
model = joblib.load("new_model_all_features.pkl")
with open("features.json", "r") as f:
    training_features = json.load(f)

# Load full dataset and filter
df_full = pd.read_csv("RT_IOT2022.csv")  # or use RT_IOT2022_small.csv
df_full = df_full.dropna()

# Sample 20% Benign, 80% Attack to make it realistic
benign_df = df_full[df_full["Attack_type"] == "Benign"].sample(n=20, random_state=42)
attack_df = df_full[df_full["Attack_type"] != "Benign"].sample(n=80, random_state=42)
df_sim = pd.concat([benign_df, attack_df]).sample(frac=1, random_state=42)  # shuffle

# Prepare features and labels
df = df_sim[training_features]
labels = df_sim["Attack_type"]

# Sidebar controls
st.sidebar.header("⚙️ Simulation Settings")
row_limit = st.sidebar.slider("How many rows to simulate?", 10, 100, 25)
delay = st.sidebar.slider("Delay between predictions (sec)", 0.1, 2.0, 0.5)

tab1, tab2 = st.tabs(["📋 Detection Log", "📊 Attack Summary"])
log_placeholder = tab1.empty()
chart_placeholder = tab2.empty()

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

        with tab1:
            log_placeholder.subheader("📋 Real-Time Detection Log")
            log_placeholder.dataframe(result_df, use_container_width=True)

            if prediction != "Benign":
                st.warning(f"⚠️ Alert: {prediction} attack detected!")

        with tab2:
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax.set_ylabel("")
            chart_placeholder.subheader("📊 Attack Summary")
            chart_placeholder.pyplot(fig)

        time.sleep(delay)
