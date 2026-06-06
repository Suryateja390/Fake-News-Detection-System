# Fake News Detection System using Kaggle Dataset (Direct URL)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset from Direct URLs
fake_url = "https://raw.githubusercontent.com/laxmimerit/Fake-Real-News-Dataset/main/data/Fake.csv"
true_url = "https://raw.githubusercontent.com/laxmimerit/Fake-Real-News-Dataset/main/data/True.csv"

print("Loading dataset...")

fake_df = pd.read_csv(fake_url)
true_df = pd.read_csv(true_url)

# Add labels
fake_df["label"] = 0  # Fake News
true_df["label"] = 1  # Real News

# Merge datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Keep only required columns
df = df[["text", "label"]]

print("Dataset Loaded Successfully!")
print("Total Records:", len(df))

# Features and Labels
X = df["text"]
y = df["label"]

# Convert text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
print("\nTraining Model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Training Complete!")

# Evaluate Model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Custom News Prediction
while True:
    print("\nEnter a news headline/article:")
    news = input(">> ")

    if news.lower() == "exit":
        break

    news_vector = vectorizer.transform([news])

    result = model.predict(news_vector)

    if result[0] == 1:
        print(" Real News")
    else:
        print(" Fake News")
