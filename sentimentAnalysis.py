import pandas as pd
data=pd.read_csv("imdb.csv")

x = data["review"]  
y = data["sentiment"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=40)

from sklearn.feature_extraction.text import CountVectorizer
vc=CountVectorizer(stop_words="english")
xtrain=vc.fit_transform(xtrain)
xtest=vc.transform(xtest)

from sklearn.naive_bayes import MultinomialNB
mb=MultinomialNB()
mb.fit(xtrain,ytrain)
ypred=mb.predict(xtest)

from sklearn.metrics import accuracy_score
ac=accuracy_score(ypred,ytest)
print(ac*100)

user_input = input("Enter a review to predict sentiment: ")
user_input_transformed = vc.transform([user_input])
prediction = mb.predict(user_input_transformed)

if prediction == "positive":
    print("Positive")
else:
    print("Negative")
