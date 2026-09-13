import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("fake_job_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 AI-Based Fake Job Posting & Scam Risk Detection System")

st.write(
    "Enter the job posting details below to check whether "
    "the job is Real or Fake."
)

# Input fields
title = st.text_input("Job Title")

department = st.text_input("Department")

company_profile = st.text_area("Company Profile")

description = st.text_area("Job Description")

requirements = st.text_area("Job Requirements")

benefits = st.text_area("Benefits")

industry = st.text_input("Industry")

function = st.text_input("Job Function")


# Check button
if st.button("🔍 Check Job"):

    if title.strip() == "" and description.strip() == "":
        st.warning("Please enter Job Title or Job Description.")

    else:

        # SAME text columns used during training
        combined_text = " ".join([
            title,
            company_profile,
            description,
            requirements,
            benefits,
            department,
            industry,
            function
        ])

        # Convert text into TF-IDF
        text_vector = tfidf.transform([combined_text])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Probability
        probability = model.predict_proba(text_vector)[0]

        fake_probability = probability[1] * 100
        real_probability = probability[0] * 100

        # Display result
        if prediction == 1:

            st.error("🚨 FAKE JOB / SCAM")

            st.write(
                f"Fake Job Probability: **{fake_probability:.2f}%**"
            )

        else:

            st.success("✅ REAL JOB")

            st.write(
                f"Real Job Probability: **{real_probability:.2f}%**"
            )