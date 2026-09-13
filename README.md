# AI-Based Fake Job Posting & Scam Risk Detection System

## Project Description
This project uses Machine Learning to predict whether a job posting is Real or Fake/Scam.

## Technologies Used
- Python
- Machine Learning
- NLP
- TF-IDF
- Logistic Regression
- Pandas
- Scikit-learn
- Streamlit

## Dataset
The project uses a Fake Job Postings dataset containing 17,880 job postings.

## Model Performance
- Test Accuracy: 97.01%
- Fake Job Recall: 90%

## Project Flow
Data Cleaning → Text Preprocessing → TF-IDF → Logistic Regression → Prediction → Streamlit

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
