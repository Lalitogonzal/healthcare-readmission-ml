# Hospital Readmission Prediction API

An end-to-end machine learning system that predicts 30-day hospital readmission
risk using clinical data and serves predictions through a REST API.

This project demonstrates the full ML workflow: data preprocessing, model
training, evaluation, and deployment using FastAPI.

---

## Project Overview

Hospital readmissions increase healthcare costs and negatively impact patient
outcomes. This system analyzes patient and hospital visit data to estimate the
probability of readmission within 30 days.

The trained model is exposed via a web API that can be integrated into clinical
decision-support systems.

---

## Tech Stack

- Python 3
- Pandas / NumPy
- scikit-learn
- FastAPI
- Uvicorn
- Joblib

---

## Dataset

- UCI Diabetes 130-US Hospitals Dataset
- Publicly available healthcare dataset containing patient admissions data

---

## Features Used

- Age  
- Time in Hospital  
- Number of Lab Procedures  
- Number of Medications  
- Gender  
- Race  

---

## Machine Learning Model

- Algorithm: Random Forest Classifier
- Preprocessing:
  - Missing value imputation
  - Feature scaling
  - One-hot encoding
- Class imbalance handling
- Model serialization using Joblib

---

## Performance

Example evaluation metrics:

- ROC-AUC: ~0.51  
- PR-AUC: ~0.11  

(Note: Readmission prediction is a highly imbalanced and challenging problem.
Future work will focus on improving model performance.)

---

## Project Structure

