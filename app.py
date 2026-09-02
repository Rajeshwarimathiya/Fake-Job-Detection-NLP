import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json
import hashlib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
VISUALS_DIR = BASE_DIR / "visuals"
MAX_FIELD_LENGTH = 10_000

st.set_page_config(page_title="Fake Job Detector", page_icon="favicon.png", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f7f6;
        background-image: radial-gradient(circle at top right, #d9eee8 0, transparent 34%), linear-gradient(135deg, #f4f7f6, #e8f0ef);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load only the model artifacts shipped with this application."""
    manifest_path = MODEL_DIR / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for filename, expected_hash in manifest.get("sha256", {}).items():
            artifact_path = MODEL_DIR / filename
            if not artifact_path.is_file():
                raise RuntimeError(f"Required model artifact is missing: {filename}")
            actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_hash.lower() != str(expected_hash).lower():
                raise RuntimeError(f"Model artifact integrity check failed: {filename}")
    model = joblib.load(MODEL_DIR / "xgb_final_model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    metadata_path = MODEL_DIR / "metadata.json"
    threshold = 0.5
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        threshold = float(metadata.get("threshold", threshold))
        if not 0.0 < threshold < 1.0:
            threshold = 0.5
    return model, vectorizer, shap.TreeExplainer(model), threshold


model, vectorizer, explainer, decision_threshold = load_artifacts()
if 1 not in model.classes_:
    raise RuntimeError("The model artifact does not contain the expected fraudulent class.")
fake_class_index = list(model.classes_).index(1)


def normalize_text(text: str) -> str:
    """Match the training-time normalization without external downloads."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

st.sidebar.title("Fake job detector")
st.sidebar.info("""
Screen a job posting with a machine-learning model, then review the evidence before taking action.
""")
st.sidebar.markdown("---")
st.sidebar.write(f"Model: XGBoost | threshold {decision_threshold:.2f}")
st.sidebar.write("Text: TF-IDF with normalized input")
st.sidebar.write("Explainability: SHAP local + global")
st.sidebar.caption("Predictions support review. Verify identity, compensation, and contact details independently.")

# Main title
st.title("Fake job posting detector", icon=":material/security:")
st.markdown("Assess a listing, inspect the signals, and make a better-informed decision.")

# Divider
st.markdown("---")
st.subheader("Analyze a listing", anchor=False)

    
    
if "example" not in st.session_state:
    st.session_state.example = False

def set_example_inputs():
    st.session_state.title = "Data Scientist"
    st.session_state.description = "We are looking for a Data Scientist to analyze large amounts of raw information to find patterns that will help improve our company."
    st.session_state.requirements = "Experience in Python, SQL, machine learning, data visualization. Familiarity with cloud platforms is a plus."

with st.container(horizontal=True, horizontal_alignment="distribute"):
    st.button("Try example listing", on_click=set_example_inputs, icon=":material/auto_awesome:")

    def clear_inputs():
        st.session_state.title = ""
        st.session_state.description = ""
        st.session_state.requirements = ""

    st.button("Clear fields", on_click=clear_inputs, icon=":material/delete_sweep:")

with st.form("job_analysis", border=True):
    title = st.text_input("Job title", key="title", max_chars=MAX_FIELD_LENGTH, placeholder="e.g., Data Scientist")
    description = st.text_area("Job description", key="description", max_chars=MAX_FIELD_LENGTH, height=180, placeholder="Describe the role, employer, location, and compensation...")
    requirements = st.text_area("Job requirements", key="requirements", max_chars=MAX_FIELD_LENGTH, height=150, placeholder="List experience, skills, and application instructions...")
    sensitivity = st.select_slider("Decision sensitivity", options=["Conservative", "Balanced", "High recall"], value="Balanced", help="Conservative reduces false alarms; High recall flags more potentially fraudulent listings for review.")
    submitted = st.form_submit_button("Analyze listing", type="primary", icon=":material/search:")

if submitted:
        fields = [title.strip(), description.strip(), requirements.strip()]
        if not all(fields):
            st.warning("Complete all three fields before analyzing the listing.", icon=":material/warning:")
        else:
            text = " ".join(fields)
            text_vector = vectorizer.transform([normalize_text(text)])
            fake_probability = float(model.predict_proba(text_vector)[0][fake_class_index])
            thresholds = {"Conservative": min(decision_threshold + 0.10, 0.90), "Balanced": decision_threshold, "High recall": max(decision_threshold - 0.10, 0.10)}
            active_threshold = thresholds[sensitivity]
            prediction = int(fake_probability >= active_threshold)
            confidence = fake_probability if prediction == 1 else 1 - fake_probability

            with st.container(border=True):
                st.subheader("Analysis result", anchor=False)
                predicted_label = "Fake" if prediction == 1 else "Real"
                result_label = predicted_label if confidence >= 0.65 else "Needs manual review"
                if result_label == "Needs manual review":
                    st.warning(f"This listing needs **manual review**. The model confidence is only {confidence:.1%}.", icon=":material/manage_search:")
                elif prediction == 1:
                    st.error(f"This listing is flagged as **Fake** with {confidence:.1%} confidence.", icon=":material/error:")
                else:
                    st.success(f"This listing is classified as **Real** with {confidence:.1%} confidence.", icon=":material/check_circle:")
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Decision", result_label)
                metric_b.metric("Confidence", f"{confidence:.1%}")
                metric_c.metric("Fake probability", f"{fake_probability:.1%}")
                st.progress(int(confidence * 100), text=f"Confidence: {confidence:.1%} | sensitivity: {sensitivity.lower()}")

            suspicious_terms = ["pay", "fee", "telegram", "whatsapp", "crypto", "bitcoin", "gift card", "urgent"]
            matched_terms = [term for term in suspicious_terms if term in normalize_text(text)]
            words = len(text.split())
            result_text = json.dumps({"prediction": result_label, "confidence": round(confidence, 4), "fake_probability": round(fake_probability, 4), "threshold": active_threshold, "sensitivity": sensitivity, "word_count": words, "review_terms": matched_terms}, indent=2)
            st.download_button("Download analysis JSON", result_text, file_name="job_analysis.json", mime="application/json", icon=":material/download:")

            review_tab, explanation_tab, checklist_tab = st.tabs(["Review signals", "Why this result", "Verification checklist"])
            with review_tab:
                if matched_terms:
                    st.warning(f"Review language detected: {', '.join(matched_terms)}. This is a screening signal, not proof of fraud.", icon=":material/flag:")
                else:
                    st.success("No common high-risk phrases were detected by the review helper.", icon=":material/search_check:")
                st.caption(f"Input quality: {words} words across all fields. Short or generic listings should be reviewed manually.")
            with explanation_tab:
                st.caption("SHAP shows which learned text features influenced this individual prediction.")
                # SHAP waterfall plot
                shap_values = explainer(text_vector)
                dense_vec = text_vector.toarray()[0]
                feature_names = vectorizer.get_feature_names_out()

                input_features = {
                    name: dense_vec[i]
                    for i, name in enumerate(feature_names)
                    if dense_vec[i] > 0
                }

                shap_exp = shap.Explanation(
                    values=shap_values.values[0][dense_vec > 0],
                    base_values=shap_values.base_values[0],
                    data=dense_vec[dense_vec > 0],
                    feature_names=list(input_features.keys())
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_exp, max_display=10, show=False)
                st.pyplot(fig)
                plt.close(fig)

            with checklist_tab:
                st.checkbox("Employer identity and domain verified", key="verify_employer")
                st.checkbox("No upfront payment or sensitive document request", key="verify_payment")
                st.checkbox("Compensation and location are realistic", key="verify_compensation")
                st.caption("Complete these checks independently before contacting an employer.")

with st.expander("📊 SHAP Global Explanation (Beeswarm Plot)"):
    st.write("The plot below shows how each feature impacts the model output globally.")
    st.image(
        str(VISUALS_DIR / "shap_beeswarm_plot.png"),
        caption="SHAP Summary Plot (Beeswarm)",
        width="stretch"
    )

with st.expander("Batch analyze a CSV", icon=":material/table_view:"):
    st.caption("Upload a CSV containing `title`, `description`, and `requirements` columns. Files are processed in memory and are not stored.")
    batch_file = st.file_uploader("Upload job postings", type="csv", key="batch_file")
    if batch_file is not None:
        if batch_file.size > 5_000_000:
            st.error("The CSV is larger than the 5 MB upload limit.", icon=":material/upload_file:")
        else:
            try:
                batch_frame = pd.read_csv(batch_file)
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
                st.error("The uploaded file could not be read as a valid UTF-8 CSV.", icon=":material/error:")
            else:
                required_columns = {"title", "description", "requirements"}
                missing_columns = required_columns - set(batch_frame.columns)
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(sorted(missing_columns))}", icon=":material/error:")
                elif len(batch_frame) > 2_000:
                    st.error("The batch is limited to 2,000 listings per upload.", icon=":material/format_list_numbered:")
                else:
                    batch_text = (
                        batch_frame["title"].fillna("").astype(str) + " "
                        + batch_frame["description"].fillna("").astype(str) + " "
                        + batch_frame["requirements"].fillna("").astype(str)
                    ).map(normalize_text)
                    batch_vectors = vectorizer.transform(batch_text)
                    batch_fake_probability = model.predict_proba(batch_vectors)[:, fake_class_index]
                    batch_confidence = np.maximum(batch_fake_probability, 1 - batch_fake_probability)
                    batch_prediction = (batch_fake_probability >= decision_threshold).astype(int)
                    batch_result = batch_frame.copy()
                    batch_result["prediction"] = np.where(batch_confidence < 0.65, "Needs manual review", np.where(batch_prediction == 1, "Fake", "Real"))
                    batch_result["confidence"] = np.round(batch_confidence, 4)
                    batch_result["fake_probability"] = np.round(batch_fake_probability, 4)
                    st.dataframe(batch_result[["prediction", "confidence", "fake_probability"]], width="stretch")
                    st.download_button("Download batch results", batch_result.to_csv(index=False), file_name="job_batch_analysis.csv", mime="text/csv", icon=":material/download:")

            
# How it works
with st.expander("ℹ️ How this works"):
    st.write("""
    This tool uses a trained XGBoost model that analyzes job title, description, and requirements using TF-IDF features.
    It was trained on a real-world job postings dataset and predicts whether a job post is fake or real.
    """)

st.markdown("---")
st.markdown("Made with ❤️ by **Rajeshwari Mathiya** | [LinkedIn](https://www.linkedin.com/in/rajeshwari-mathiya-77a4a3321) | [GitHub Profile](https://github.com/Rajeshwarimathiya) | [Project repository](https://github.com/Rajeshwarimathiya/Fake-Job-Detection-NLP)")