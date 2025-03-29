# Tweet Sentiment Analysis

A simple machine learning model that analyzes the sentiment of text inputs, specifically trained on tweet data.

### Overview
- Analyzes whether text inputs have positive or negative sentiment
- Uses machine learning to classify text based on sentiment
- Trained on a dataset of tweets with labeled sentiments

### Technologies
- Python
- scikit-learn for TF-IDF vectorization and classification
- Pandas for data handling
- joblib for model persistence

### Features
- Text sentiment classification (positive/negative)
- TF-IDF vectorization with n-gram support
- Trained using Logistic Regression with OneVsRest classifier
- Model persistence using joblib

### Installation
- Clone the repository
- Install required packages: `pip install pandas scikit-learn joblib`

### Dataset
- Cleaned dataset of tweets stored in parquet format
- Label 4 represents positive sentiment
- Other labels represent negative sentiment
- Dataset: https://drive.google.com/drive/folders/1l0GR5NjO1CbiQ0B6HUNoYyqoUxaGkQTR?usp=sharing

### Usage
- Run `python sentiment_analysis.py`
- If no model exists, it will train one using the dataset
- Enter text when prompted
- Receive sentiment classification (positive or negative)

### Model Details
- Vectorization: TF-IDF with max 10,000 features and 1-2 word n-grams
- Classifier: Logistic Regression with OneVsRest
- Train-Test Split: 80% training, 20% testing with random_state=50