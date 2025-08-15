import streamlit as st
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import html

# ---------------------------
# Helpers
# ---------------------------
def safe_expected_value(exp):
    """Handle expected_value type differences from SHAP."""
    # SHAP may return scalar, list, np.ndarray, or None
    if exp is None:
        return 0.0
    if isinstance(exp, (list, np.ndarray)):
        return exp[1] if len(exp) > 1 else exp[0]
    return exp

def build_token_highlight_html(tokens, values, max_tokens=400):
    """Return HTML string that highlights tokens by SHAP value (green=push to FAKE, red=push to REAL)."""
    tokens = list(tokens)[:max_tokens]
    values = np.array(values[:max_tokens], dtype=float)

    # scale intensities
    max_abs = float(np.nanmax(np.abs(values))) if values.size else 0.0
    max_abs = max(max_abs, 1e-12)  # avoid div/0

    # legend
    legend = """
    <div style="font-size:14px;margin:6px 0;">
      <span style="display:inline-block;width:14px;height:14px;background:rgba(0,128,0,0.6);vertical-align:middle;border-radius:3px;margin-right:6px;"></span>
      pushes toward <b>FAKE (class 1)</b>
      &nbsp;&nbsp;&nbsp;
      <span style="display:inline-block;width:14px;height:14px;background:rgba(220,0,0,0.6);vertical-align:middle;border-radius:3px;margin-right:6px;margin-left:16px;"></span>
      pushes toward <b>REAL (class 0)</b>
    </div>
    """

    parts = [legend, '<div style="white-space:pre-wrap;line-height:1.9;font-family:ui-monospace, Menlo, Consolas, monospace;">']
    for tok, val in zip(tokens, values):
        # intensity (alpha) scaled by abs(val)
        alpha = min(abs(val) / max_abs, 1.0)
        alpha = 0.08 + 0.72 * alpha  # keep faint but visible
        bg = f"rgba(0,128,0,{alpha:.3f})" if val >= 0 else f"rgba(220,0,0,{alpha:.3f})"
        safe_tok = html.escape(str(tok))
        span = f'<span style="background:{bg};padding:2px 4px;border-radius:4px;margin:1px 1px 2px 0;display:inline-block;">{safe_tok}</span>'
        parts.append(span)
        
        if not safe_tok.endswith((" ", "\n", "\t")):         
            parts.append(" ")
    parts.append("</div>")
    return "".join(parts)

# ---------------------------
# Load artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(r"C:\Users\user\Music\FINAL_PROJECTS\tfidf_vectorizer.pkl")
    model = joblib.load(r"C:\Users\user\Music\FINAL_PROJECTS\fake_job_model.pkl")
    return vectorizer, model

vectorizer, model = load_artifacts()

# ---------------------------
# SHAP explainer (text)
# ---------------------------
@st.cache_resource
def get_shap_explainer():
    masker = shap.maskers.Text()  # works without custom tokenizer
    # Use proba so we get per-class outputs
    return shap.Explainer(lambda x: model.predict_proba(vectorizer.transform(x)), masker)

explainer = get_shap_explainer()

# ---------------------------
# Streamlit UI setup
# ---------------------------


#Page config
st.set_page_config(
    page_title="Fake Job Post Detector",
    page_icon="🚨",
    layout="centered"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        }
        /* Centering title */
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #d90429;
        }
        /* Subtitle style */
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#Title & Subtitle
st.markdown(
    "<h1 style='text-align: center; color: #FF5733; font-size: 38px; font-family: Verdana;'>🕵️‍♂️FAKE JOB POST DETECTOR🚨</h1>",
    unsafe_allow_html=True
)

#st.markdown('<h1 class="main-title">🚨FAKE JOB POST DETECTOR🚨</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect fraudulent job postings instantly which is **REAL** or **FAKE** using NLP and Machine Learning</p>', unsafe_allow_html=True)



# ---------------------------
# Inputs
# ---------------------------

# Job Title Input
st.markdown("**📝Job Title**")
job_title = st.text_input(
    label="",
    placeholder="Enter the job title (e.g., Data Analyst)",
    help="Type the official job title as listed in the posting."
)
# Job Description Input
st.markdown("**📝Job Description**")
job_description = st.text_area(
    label="",
    placeholder="Paste the complete job description here...",
    height=200,
    help="Include roles, responsibilities, requirements, and other relevant details."
)
#job_title = st.text_input("**Job Title**")
#job_description = st.text_area("**Job Description**", height=200)

# ---------------------------
# Prediction + Explanations
# ---------------------------
if st.button("🔍 Predict"):
    if job_title.strip() == "" or job_description.strip() == "":
        st.warning("Please enter both job title and job description.")
    else:
        text_input = job_title.strip() + " " + job_description.strip()

        # Prediction
        X = vectorizer.transform([text_input])
        proba = model.predict_proba(X)[0]
        
        proba_fake = float(proba[1])

        if proba_fake >= 0.5:
            st.error(f"🚨 This job posting is likely **FAKE** (confidence: {proba_fake:.2%})")
        elif proba_fake <= 0.3:
            st.success(f"✅ This job posting is likely **REAL** (confidence: {(1 - proba_fake):.2%})")
        else:
            st.warning(f"⚠️ **SUSPICIOUS** — borderline classification "
                       f"(Fake: {proba_fake:.2%}, Real: {(1 - proba_fake):.2%})")
            
        # ---------------------------
        # SHAP explanations
        # ---------------------------
        st.subheader("📊 Model Explanation (SHAP)")

        try:
            sv = explainer([text_input])  # Explanation object for one sample

            # Choose class-1 (FAKE) column if multi-output
            # sv.values shapes:
            #  - (samples, tokens, classes)  -> pick class index 1
            #  - (samples, tokens)           -> already single output
            if hasattr(sv, "values") and sv.values.ndim == 3:
                vals = sv.values[..., 1]
                base_value = safe_expected_value(explainer.expected_value)
            else:
                vals = sv.values
                base_value = safe_expected_value(explainer.expected_value)

            # tokens for display; if missing, fall back to analyzer
            tokens = sv.data[0] if (hasattr(sv, "data") and sv.data is not None) else None
            if tokens is None or (isinstance(tokens, (list, np.ndarray)) and len(tokens) == 0):
                tokens = vectorizer.build_analyzer()(text_input)

            # Build an Explanation for the waterfall plot
            explanation = shap.Explanation(
                values=vals[0],
                base_values=base_value,
                data=tokens         # used for labels
            )

            # Waterfall (top contributions)
            fig, _ = plt.subplots(figsize=(6, 3))
            shap.plots.waterfall(explanation, show=False, max_display=15)
            st.pyplot(fig)


            # Custom token-level heatmap (HTML) — robust across SHAP versions
            st.subheader("📝 Token-level Highlights")
            html_block = build_token_highlight_html(tokens, vals[0])
            st.components.v1.html(html_block, height=150, scrolling=True)

            # Optional: Top words table
            st.markdown("**Top influential tokens (toward FAKE):**")
            top_idx = np.argsort(-np.abs(vals[0]))[:15]
            st.table({
                "token": [str(tokens[i]) for i in top_idx],
                "shap_value": [float(vals[0][i]) for i in top_idx]
            })
        except Exception as e:
            st.warning("SHAP explanation failed. Reason: {e}")
