# AI-Based Fake Job Posting & Scam Risk Detection System

## Project Description

This project uses Machine Learning to predict whether a job posting is Real or Fake/Scam.

## Objectives

- Detect fake and fraudulent job postings.
- Use Natural Language Processing for job posting text.
- Convert text into numerical features using TF-IDF.
- Classify job postings using Logistic Regression.
- Provide prediction through a Streamlit web application.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLP
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit

## Dataset

The project uses a public Fake Job Postings dataset containing:

- 17,880 job postings
- 18 columns
- Target column: `fraudulent`
- `0` = Real Job
- `1` = Fake Job

The original dataset is not included in this repository because the file size is larger than GitHub's web upload limit.

## Machine Learning

### Text Feature Extraction
TF-IDF is used to convert job posting text into numerical features.

### Classification Algorithm
Logistic Regression is used to classify job postings into:

- Real Job
- Fake Job / Scam

## Model Performance

- Test Accuracy: **97.01%**
- Fake Job Precision: **64%**
- Fake Job Recall: **90%**
- Fake Job F1-Score: **74%**

## Project Flow

Dataset → Data Cleaning → Text Preprocessing → TF-IDF → Train/Test Split → Logistic Regression → Model Evaluation → Streamlit Prediction

## Streamlit Application

The application allows the user to enter job posting details and predicts whether the job is Real or Fake/Scam.

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
