import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("imdb.csv")

x = data["review"]
y = data["sentiment"]

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.20, random_state=40
)

# Vectorization
vc = CountVectorizer(stop_words="english")
xtrain = vc.fit_transform(xtrain)
xtest = vc.transform(xtest)

# Model
mb = MultinomialNB()
mb.fit(xtrain, ytrain)

# Accuracy
ypred = mb.predict(xtest)
print("Accuracy:", accuracy_score(ytest, ypred) * 100)

# 🔥 Save model & vectorizer
pickle.dump(mb, open("sentiment_model.pkl", "wb"))
pickle.dump(vc, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully")
