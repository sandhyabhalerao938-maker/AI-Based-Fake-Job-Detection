import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Load model and vectorizer
model = joblib.load("fake_job_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Take one actual job from dataset
row = df[df["fraudulent"] == 1].iloc[0]

text_cols = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "department",
    "industry",
    "function"
]

combined_text = " ".join(
    str(row[col]) if pd.notna(row[col]) else ""
    for col in text_cols
)

text_vector = tfidf.transform([combined_text])

prediction = model.predict(text_vector)[0]

print("Actual Label:", row["fraudulent"])
print("Predicted Label:", prediction)

if prediction == 1:
    print("Prediction: FAKE JOB / SCAM 🚨")
else:
    print("Prediction: REAL JOB ✅")