
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🎫 Support Ticket Classifier")

st.write(
    "Enter a support ticket description to predict its topic group."
)

st.divider()

ticket_text = st.text_area(
    "Ticket Description",
    placeholder="Example: I cannot access my company account...",
    height=160,
)

if st.button("Classify Ticket", type="primary", use_container_width=True):

    if not ticket_text.strip():
        st.warning("Please enter a ticket description.")

    else:
        prediction = model.predict([ticket_text.strip()])[0]

        st.subheader("Prediction Result")

        st.success(f"Topic Group: {prediction}")

        