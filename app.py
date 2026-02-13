# ======================================
# FRAUD DETECTION PRO - FINAL VERSION
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, classification_report

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="Fraud Detection Pro", layout="wide")

# Custom Styling
st.markdown("""
<style>
.big-font { font-size:26px !important; font-weight:600; }
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------
# LOAD MODEL & DATA
# --------------------------------------
model = pickle.load(open("fraud_model.pkl", "rb"))
X_test = pickle.load(open("X_test.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

feature_names = X_test.columns

# --------------------------------------
# TITLE
# --------------------------------------
st.title("💳 Fraud Detection Pro")
st.markdown("### AI-Powered Credit Card Fraud Detection System")

# --------------------------------------
# SIDEBAR INPUTS
# --------------------------------------
st.sidebar.header("Transaction Details")

input_values = []
for col in feature_names:
    val = st.sidebar.number_input(col, value=0.0)
    input_values.append(val)

threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5)

features_df = pd.DataFrame([input_values], columns=feature_names)

# --------------------------------------
# PREDICTION
# --------------------------------------
probability = model.predict_proba(features_df)[0][1]
prediction = int(probability > threshold)

# --------------------------------------
# MAIN DISPLAY (2 Columns)
# --------------------------------------
col1, col2 = st.columns(2)

# ==============================
# FRAUD PROBABILITY GAUGE
# ==============================
with col1:
    st.subheader("🎯 Fraud Probability")

    fig1, ax1 = plt.subplots()
    ax1.barh(["Fraud Probability"], [probability])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Probability")
    st.pyplot(fig1)

    st.markdown(
        f"<p class='big-font'>Fraud Risk: {round(probability*100,2)}%</p>",
        unsafe_allow_html=True
    )

    if probability < 0.3:
        st.success("Low Risk Transaction ✅")
    elif probability < 0.7:
        st.warning("Medium Risk Transaction ⚠️")
    else:
        st.error("High Risk Transaction 🚨")

# ==============================
# SHAP EXPLAINABILITY
# ==============================
# ==============================
# SHAP EXPLAINABILITY (FIXED)
# ==============================
with col2:
    st.subheader("🔎 SHAP Explainability")

    try:
        explainer = shap.Explainer(model, X_test)

        shap_values = explainer(features_df)

        # For binary classification → select fraud class (index 1)
        if len(shap_values.shape) == 3:
            explanation = shap_values[:, :, 1]
        else:
            explanation = shap_values

        fig2 = plt.figure()
        shap.plots.waterfall(explanation[0], show=False)
        st.pyplot(fig2)

    except Exception as e:
        st.warning("SHAP visualization unavailable due to compatibility issue.")

# --------------------------------------
# MODEL PERFORMANCE
# --------------------------------------
st.markdown("---")
st.subheader("📊 Model Performance")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

metrics_df = pd.DataFrame(report).transpose().round(4)
st.dataframe(metrics_df)

# --------------------------------------
# CONFUSION MATRIX
# --------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax3)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

# --------------------------------------
# ROC CURVE
# --------------------------------------
st.subheader("📈 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

fig4, ax4 = plt.subplots()
ax4.plot(fpr, tpr, label="Model ROC")
ax4.plot([0, 1], [0, 1], linestyle="--")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.legend()
st.pyplot(fig4)

# --------------------------------------
# BUSINESS COST ANALYSIS
# --------------------------------------
st.markdown("---")
st.subheader("💰 Business Cost Analysis")

false_negative_cost = 5000
false_positive_cost = 500

st.write("Cost per Missed Fraud: ₹", false_negative_cost)
st.write("Cost per False Alarm: ₹", false_positive_cost)

# --------------------------------------
# DOWNLOAD REPORT
# --------------------------------------
st.markdown("---")

if st.button("Generate Prediction Report"):

    df_result = pd.DataFrame({
        "Fraud Probability": [round(probability, 4)],
        "Prediction (1=Fraud)": [prediction]
    })

    csv = df_result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="fraud_prediction_report.csv",
        mime="text/csv"
    )
