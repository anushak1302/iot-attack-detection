import streamlit as st
import pandas as pd
import joblib
import json
import time
import matplotlib.pyplot as plt
import numpy as np

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
df = df_full[feature_names]
labels = df_full["Attack_type"]
df_scaled = scaler.transform(df)

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

    indices = np.random.choice(len(df_scaled), size=sample_size, replace=False)

    for i, idx in enumerate(indices):
        row = df_scaled[idx].reshape(1, -1)
        prediction = model.predict(row)[0]
        actual = labels.iloc[idx]

        result_log.append({
            "Row": i + 1,
            "Predicted": prediction,
            "Actual": actual
        })
        chart_data.append(prediction)

        result_df = pd.DataFrame(result_log)

        # --- Detection Log ---
        with tab1:
            log_placeholder.subheader("📋 Real-Time Detection Log")
            st.dataframe(result_df.style.applymap(
                lambda x: 'background-color: #ffcccc' if x != "Benign" and x == prediction else '',
                subset=["Predicted"]
            ), use_container_width=True)
            if prediction != "Benign":
                st.warning(f"⚠️ Alert: Potential `{prediction}` detected!")

        # --- Pie Chart Summary ---
        with tab2:
            pie_data = pd.Series(chart_data).value_counts()
            fig, ax = plt.subplots()
            pie_data.plot.pie(autopct="%1.1f%%", startangle=90, colors=plt.cm.Set3.colors)
            ax.set_ylabel("")
            chart_placeholder.subheader("📊 Attack Distribution")
            chart_placeholder.pyplot(fig)

        time.sleep(delay)
