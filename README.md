# 🚨 ML-Based Fake Job Postings Detector Using NLP and Machine Learning

## 📌 Overview

This project builds a machine learning model to detect fake job postings using Natural Language Processing (NLP) techniques and classification algorithms.

It processes job descriptions, requirements, and company details to identify fraudulent listings, helping protect job seekers and maintain trust in online job platforms.

The system provides real-time fraud probability scoring through a REST API and a Streamlit frontend.

# 🎯 Skills Gained

Natural Language Processing (NLP)

Text preprocessing and cleaning

TF-IDF Vectorization and Word Embeddings (Word2Vec)

Machine Learning models: Logistic Regression, SVM, Gradient Boosting

Model evaluation using Precision, Recall, F1-Score, ROC-AUC

Handling imbalanced datasets (SMOTE, class weighting)

## 🧠 Features

Text preprocessing and feature engineering

TF-IDF vectorization

Multiple classification models (Logistic Regression, SVM, XGBoost)

Model evaluation using:

    Precision

    Recall

    F1-Score

    Confusion Matrix

    ROC-AUC

FastAPI-based REST API for real-time inference

Streamlit frontend with fraud risk visualization

Modular architecture (Model → API → Frontend)


## 🏗️ Project Architecture

Streamlit Frontend
⬇
FastAPI Backend
⬇
TF-IDF Vectorizer + Trained ML Model


## 📊 Dataset

Source: Kaggle – Fake Job Postings Prediction

## 📊 Model Evaluation

    The models were evaluated using:

    Precision, Recall, F1-score

    Confusion Matrix

    ROC-AUC score

The final model provides fraud probability scoring for risk categorization:

    Low Risk

    Medium Risk

    High Risk

## Run FastAPI Backend

    uvicorn app:app --reload

   Open API docs at:

    http://127.0.0.1:8000/docs

## Run Streamlit Frontend

    streamlit run streamlit_app.py

## 🎯 Learning Outcomes

End-to-end ML pipeline development

Model serialization and deployment

REST API development with FastAPI

Frontend integration using Streamlit

Real-time ML inference architecture







