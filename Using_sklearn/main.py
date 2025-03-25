import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


MODEL_FILENAME = "sentiment_model.pkl"

# Check if model exists
if os.path.exists(MODEL_FILENAME):
    print("Loading the saved model...")
    model = joblib.load(MODEL_FILENAME)

else:

    #Read data from the parquet file access using this link ("https://drive.google.com/drive/folders/1l0GR5NjO1CbiQ0B6HUNoYyqoUxaGkQTR?usp=sharing")
    data = pd.read_parquet("clean_dataset.parquet")
    print(data.head())

    #Get values
    X = data['tweet'].values
    y = data['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    log_reg = LogisticRegression(solver='lbfgs')
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ("classifier", OneVsRestClassifier(log_reg))
    ])

    #train the model
    model.fit(X_train,y_train)

    #save model
    joblib.dump(model, MODEL_FILENAME)
    print("Model trained and saved!")


X_test = [input("Enter your thoughts: ")]

prediction = model.predict(X_test)


if prediction[0] == 4:
    print("Your thoughts seems positive!!")
else:
    print("You are giving negative vibes!!")