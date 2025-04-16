import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import random

# --------------------------
# 🎨 App Styling & Config
# --------------------------
st.set_page_config(page_title="IoT Attack Detection", layout="wide")
st.markdown(
    "<h1 style='color:#009999; text-align:center;'>🔐 IoT Attack Detection Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align:center;'>Real-time simulated detection using multiple ML models</h4><hr style='border-top: 1px solid #ccc;'>",
    unsafe_allow_html=True,
)

# --------------------------
# 🔍 Model Selection
# --------------------------
model_choice = st.sidebar.selectbox(
    "🧠 Choose ML Model",
    ["Random Forest", "SVM", "KNN", "Gradient Boosting"]
)

model_files = {
    "Random Forest": "rf_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "Gradient Boosting": "gb_model.pkl"
}

model = joblib.load(model_files[model_choice])

# --------------------------
# 📂 Load Metadata
# --------------------------
with open("features.json", "r") as f:
    feature_names = json.load(f)

scaler = joblib.load("scaler.pkl")

# --------------------------
# 🧪 Load Test Data
# --------------------------
df_full = pd.read_csv("streamlit_test_attacktypes_90_correct.csv")
df_full = df_full.dropna()

# Split Benign and Attacks separately
df_benign = df_full[df_full["Attack_type"] == "Benign"]
df_attack = df_full[df_full["Attack_type"] != "Benign"]

# Prepare features
df_benign_scaled = scaler.transform(df_benign[feature_names])
df_attack_scaled = scaler.transform(df_attack[feature_names])

labels_benign = df_benign["Attack_type"].reset_index(drop=True)
labels_attack = df_attack["Attack_type"].reset_index(drop=True)

# --------------------------
# ⚙️ Simulation Settings
# --------------------------
st.sidebar.header("⚙️ Simulation Settings")
delay = st.sidebar.slider("⏳ Delay between predictions (sec)", 0.1, 2.0, 0.4)
sample_size = st.sidebar.slider("🔁 Rows to simulate (randomized each run)", 1, 15, 10)

# --------------------------
# 📋 UI Placeholders
# --------------------------
tab1, tab2 = st.tabs(["📋 Detection Log", "📊 Attack Summary"])
log_placeholder = tab1.empty()
chart_placeholder = tab2.empty()

# --------------------------
# 🚀 Simulate Button
# --------------------------
if st.button("🚀 Start Simulation"):
    result_log = []
    chart_data = []

    # Simulate 20% benign rows and 80% attack rows
    num_benign = max(1, int(sample_size * 0.2))
    num_attack = sample_size - num_benign

    benign_indices = np.random.choice(len(df_benign_scaled), size=num_benign, replace=False)
    attack_indices = np.random.choice(len(df_attack_scaled), size=num_attack, replace=False)

    combined_rows = []

    for i in benign_indices:
        combined_rows.append({
            "data": df_benign_scaled[i].reshape(1, -1),
            "actual": labels_benign[i]
        })

    for i in attack_indices:
        combined_rows.append({
            "data": df_attack_scaled[i].reshape(1, -1),
            "actual": labels_attack[i]
        })

    # Shuffle all selected rows
    random.shuffle(combined_rows)

    # Run prediction on each row
    for i, row_info in enumerate(combined_rows):
        row = row_info["data"]
        actual = row_info["actual"]
        prediction = model.predict(row)[0]

        result_log.append({
            "Row": i + 1,
            "Predicted": prediction,
            "Actual": actual
        })
        chart_data.append(prediction)

        result_df = pd.DataFrame(result_log)

        # --- Update Detection Log ---
        with tab1:
            log_placeholder.subheader("📋 Real-Time Detection Log")
            log_placeholder.dataframe(
                result_df.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x != "Benign" and x == prediction else '',
                    subset=["Predicted"]
                ),
                use_container_width=True
            )
            if prediction != "Benign":
                st.warning(f"⚠️ Alert: `{prediction}` attack detected!")

        # --- Update Pie Chart ---
        with tab2:
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct="%1.1f%%", startangle=90, colors=plt.cm.Set3.colors)
            ax.set_ylabel("")
            chart_placeholder.subheader("📊 Attack Distribution")
            chart_placeholder.pyplot(fig)

        time.sleep(delay)
