import streamlit as st
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    layout="wide"
)

# ---------------------------------------------------
# Premium Soft Dark Theme + Glass UI
# ---------------------------------------------------
st.markdown("""
<style>

/* Soft gradient background */
html, body, .main {
    background: linear-gradient(135deg, #111827, #1f2937);
    color: #f3f4f6;
}

/* Reduce default spacing */
.block-container {
    padding-top: 2rem;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 8px 24px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
}

.glass:hover {
    transform: translateY(-4px);
}

/* KPI box */
.kpi {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
}

/* Title */
.title {
    font-size: 32px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Text area styling */
textarea {
    background-color: #f9fafb !important;
    color: #111827 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    caret-color: #111827 !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    border: none;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 8px 20px rgba(37,99,235,0.4);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model & Vectorizer
# ---------------------------------------------------
with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------------------------------------------
# Text Cleaning Function
# ---------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("<div class='title'>Customer Review Sentiment Analyzer</div>", unsafe_allow_html=True)
st.write("AI-based sentiment classification of customer reviews.")

st.divider()

# ---------------------------------------------------
# Main Layout
# ---------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    review = st.text_area("Enter Customer Review")

    if st.button("Analyze Sentiment"):

        if review.strip() != "":
            cleaned = clean_text(review)
            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]
            confidence = np.max(probabilities) * 100

            st.subheader("Prediction Result")

            # Dynamic color styling
            if prediction == "Positive":
                color = "#22c55e"
            elif prediction == "Neutral":
                color = "#facc15"
            else:
                color = "#ef4444"

            st.markdown(f"""
                <div style="
                    padding: 15px;
                    border-radius: 12px;
                    background-color: rgba(255,255,255,0.05);
                    border-left: 6px solid {color};
                    font-size: 18px;
                ">
                    <strong style="color:{color};">{prediction}</strong><br>
                    Confidence: {confidence:.2f}%
                </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Please enter a review before analyzing.")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Probability Distribution")

    if 'probabilities' in locals():

        labels = model.classes_

        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(labels, probabilities,
                      color=["#ef4444", "#facc15", "#22c55e"])

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")

        for i, v in enumerate(probabilities):
            ax.text(i, v + 0.02, f"{v*100:.1f}%",
                    ha='center', fontweight='bold')

        st.pyplot(fig)

    else:
        st.write("Run analysis to view distribution.")

    st.markdown("</div>", unsafe_allow_html=True)
