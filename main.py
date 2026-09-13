import pandas as pd

df = pd.read_csv("fake_job_postings.csv")

print("Dataset Loaded Successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nReal vs Fake Jobs:")
print(df["fraudulent"].value_counts())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

text_columns = [
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

for col in text_columns:
    df[col] = df[col].fillna("")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

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

df["combined_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)

print("\nCombined Text:")
print(df["combined_text"].head())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(df["combined_text"])
y = df["fraudulent"]

print("\nTF-IDF Shape:")
print(X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

import joblib

joblib.dump(model, "fake_job_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel and TF-IDF Vectorizer Saved Successfully!")