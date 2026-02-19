import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.title("🕵️‍♂️ Fake Job Detection System")
st.caption("Enter job details to assess whether a posting is legitimate or fraudulent")

# ---------------- Layout ---------------- #
col1, col2 = st.columns(2)

with col1:
    job_title = st.text_area("Job Title",height=10)
    description = st.text_area("Job Description", height=10)

with col2:
    requirements = st.text_area("Requirements", height=10)
    benefits = st.text_area("Benefits", height=10)

st.markdown("---")

# ---------------- Prediction ---------------- #
if st.button("🔍 Analyze Job Posting"):
    if not job_title or not description:
        st.warning("Please provide at least Job Title and Job Description.")
    else:
        payload = {
            "job_title": job_title,
            "description": description,
            "requirements": requirements,
            "benefits": benefits
        }

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            prob = float(result["probability_fraud"])
            prediction = result["prediction"]

            st.subheader("📊 Prediction Result")

            # ---------- Risk Level ----------
            if prob < 0.30:
                st.success("🟢 Low Risk — Job appears legitimate")
                color = "green"
            elif prob < 0.70:
                st.warning("🟡 Medium Risk — Job looks suspicious")
                color = "orange"
            else:
                st.error("🔴 High Risk — Job likely fraudulent")
                color = "red"

            # ---------- Probability Bar ----------
            st.markdown("### Fraud Probability")
            st.progress(prob)
            st.markdown(
                f"<span style='color:{color}; font-size:18px;'>"
                f"{prob*100:.2f}% chance of fraud</span>",
                unsafe_allow_html=True
            )

        except requests.exceptions.ConnectionError:
            st.error("FastAPI backend is not running. Please start the API first.")
        except Exception as e:
            st.error(f"Error: {e}")
