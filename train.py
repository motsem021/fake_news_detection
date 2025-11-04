import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib

# LOAD DATA
news_dataset = pd.read_csv('C:\\Users\\sesra\\OneDrive\\Desktop\\PS\\project\\fake_news_detection\\15_fake_news_detection.csv', engine='python', on_bad_lines='skip')

# label encoding
label_encoder = LabelEncoder()
news_dataset['label'] = label_encoder.fit_transform(news_dataset['label'])

# data split
X = news_dataset['text'].values
Y = news_dataset['label'].values

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

# MODEL
model = LogisticRegression(max_iter=2000, class_weight='balanced', C=2.0)
model.fit(X_train, Y_train)

# scores
print("train accuracy:", accuracy_score(model.predict(X_train), Y_train))
print("test accuracy:", accuracy_score(model.predict(X_test), Y_test))
print("cross val:", cross_val_score(model, X, Y, cv=5).mean())

# SAVE
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("MODEL SAVED ")
