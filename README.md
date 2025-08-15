# 🚨 ML-Based Fake Job Postings Detector Using NLP and Machine Learning

## 📌 Overview

This project builds a machine learning model to detect fake job postings using Natural Language Processing (NLP) techniques and classification algorithms.

It processes job descriptions, requirements, and company details to identify fraudulent listings, helping protect job seekers and maintain trust in online job platforms.

# 🎯 Skills Gained

Natural Language Processing (NLP)

Text preprocessing and cleaning

TF-IDF Vectorization and Word Embeddings (Word2Vec)

Machine Learning models: Logistic Regression, SVM, Gradient Boosting

Model evaluation using Precision, Recall, F1-Score, ROC-AUC

Handling imbalanced datasets (SMOTE, class weighting)

## 🏢 Domain

Human Resources

Job Portals & Search Engines

Online Security

Text Classification (NLP)


## 📍 Problem Statement

Fake job postings are a growing problem on online platforms, posing risks such as identity theft and scams. 
This project detects fraudulent postings based on textual and metadata features, allowing:

1.Job portals to automatically flag suspicious listings

2.Job seekers to avoid scams

3.HR teams to protect company brand reputation

## 💼 Business Use Cases

Job Portals: Screen and block scams before they go live

Job Search Engines: Improve trust by labeling suspicious posts

Corporate HR: Identify fraudulent postings impersonating their brand

Candidate Safety: Protect from phishing and identity theft

## 📊 Dataset

Source: Kaggle – Fake Job Postings Prediction

Size: ~17,000 job postings

Features: Job title, description, location, requirements, benefits, label (real or fake)

## 🛠 Approach

1. Data Preprocessing

Handle missing values, drop irrelevant columns

Clean text: lowercase, remove HTML tags, punctuation, numbers

Tokenization and lemmatization

Combine multiple text fields into a single feature

2. Feature Extraction

TF-IDF vectorization (max_features=5000)

(Optional) Word2Vec embeddings

3. Model Training

Algorithms: Logistic Regression, SVM, XGBoost

Stratified train-test split

Hyperparameter tuning with GridSearchCV

4. Model Evaluation

Precision, Recall, F1-Score

Confusion Matrix

ROC-AUC Curve

Precision–Recall Curve

SHAP feature importance for explainability

5. Deployment

Export model & vectorizer with joblib

## 📈 Results

ROC-AUC: 0.98

Recall (Fake class): ~85% (most fake jobs detected)

Precision (Fake class): ~68% (low false alarm rate)

High accuracy with strong balance between catching scams and avoiding false positives.

## 📜 Evaluation Metrics

Precision: Accuracy of scam predictions

Recall: Ability to catch all scams

F1-Score: Balance of precision and recall

Confusion Matrix: Class-wise performance visualization




