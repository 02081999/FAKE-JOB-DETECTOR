import streamlit as st
import requests

# 🔗 FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

# 🔧 Page config
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# 🎯 Title
st.title("🕵️‍♂️ Fake Job Detection System")
st.caption("Analyze job postings to detect fraudulent listings")

# ---------------- INPUT SECTION ---------------- #
col1, col2 = st.columns(2)

with col1:
    job_title = st.text_area("📌 Job Title", height=150)
    description = st.text_area("📝 Job Description", height=150)

with col2:
    requirements = st.text_area("📋 Requirements", height=150)
    benefits = st.text_area("🎁 Benefits", height=150)

st.markdown("---")

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Analyze Job Posting"):

    if not job_title.strip() or not description.strip():
        st.warning("⚠️ Please enter Job Title and Description")
    
    else:
        payload = {
            "job_title": job_title,
            "description": description,
            "requirements": requirements,
            "benefits": benefits
        }

        try:
            with st.spinner("Analyzing job posting..."):
                response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()

                prediction = result["prediction"]  # "Fraudulent" or "Legitimate"
                prob = float(result["probability_fraud"])

                st.subheader("📊 Prediction Result")

                # 🎯 Risk Levels
                if prediction == "Fraudulent":
                    st.error(f"🔴 Fraudulent Job ({prob*100:.2f}% confidence)")
                else:
                    st.success(f"🟢 Legitimate Job ({(1-prob)*100:.2f}% confidence)")

                # 📈 Probability Bar
                st.markdown("### 📈 Fraud Probability")
                st.progress(prob)

                st.write(f"Fraud Probability: {prob:.2f}")

            else:
                st.error("❌ API Error. Check backend.")

        except requests.exceptions.ConnectionError:
            st.error("🚫 FastAPI backend is not running.")
        
        except Exception as e:
            st.error(f"Error: {e}")