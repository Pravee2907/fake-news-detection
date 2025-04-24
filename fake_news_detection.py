from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Dataset (Mini version)
news = [
    "The government has announced a new scheme for farmers",
    "NASA discovers new planet in solar system",
    "Click here to win a million dollars now!",
    "Elections postponed due to weather conditions",
    "Scientists create vaccine for deadly virus",
    "You won a free iPhone! Claim now!",
    "Fake news spreading quickly on WhatsApp",
]

labels = [
    "REAL",  # Real news
    "REAL",
    "FAKE",  # Fake news
    "REAL",
    "REAL",
    "FAKE",
    "FAKE"
]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Test it manually
sample = ["Breaking: Celebrity spotted in secret alien base!"]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)
print("Prediction for sample news:", prediction[0])
