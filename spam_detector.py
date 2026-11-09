
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from prepare_data import extract_features

# Load the email dataset
data_path = Path("data/emails.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "Run prepare_data.py first to create data/emails.csv"
    )

df = pd.read_csv(data_path)

# Use the six email features
features = [
    "word_count",
    "exclamations",
    "money_words",
    "all_caps",
    "urgent_words",
    "question_marks",
]

X = df[features]
y = df["label"]

# Split the emails into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train the spam detector
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, pos_label=1))
print("Recall:", recall_score(y_test, predictions, pos_label=1))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Show which features influence the predictions
print("\nFeature weights:")
for feature, weight in zip(features, model.coef_[0]):
    print(f"{feature}: {weight:.4f}")
# Part 3.5 - Test emails designed to fool the spam detector

test_emails = [
    "Hello, your meeting is scheduled for tomorrow at 10 AM.",
    "Congratulations!!! You won a FREE cash prize!!!",
    "Hi, please review the attached invoice for your records."
]

print("\nTesting three new emails:")

for email in test_emails:
    print(email)
    email_features = extract_features(email)
    email_df = pd.DataFrame([email_features])[features]
    prediction = model.predict(email_df)[0]
    print("Prediction:", "Spam" if prediction == 1 else "Legitimate")       