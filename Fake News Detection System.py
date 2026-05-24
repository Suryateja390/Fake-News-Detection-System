#Fake News Detection System

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#dataset
data = {
    "news": [
        "Government launches new education policy",
        "Aliens landed in New York yesterday",
        "Stock market reaches all time high",
        "Fake miracle cure discovered online"
    ],
    "label": [1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Features and labels
x = df["news"]
y = df["label"]

# Convert text into numerical data
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Prediction
prediction = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, prediction))

# Test custom news
sample_news = ["Breaking news: Scientists discover water on Mars"]
sample_vector = vectorizer.transform(sample_news)

result = model.predict(sample_vector)

if result[0] == 1:
    print("Real News")
else:
    print("Fake News")