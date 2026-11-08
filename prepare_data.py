
import pandas as pd
from pathlib import Path

# Sample dataset: 8 spam and 8 legitimate emails
emails = [
    ("WIN FREE CASH NOW!!! Claim your prize!", 1),
    ("Congratulations! You won a money prize!", 1),
    ("URGENT!!! Send money to claim your reward!", 1),
    ("Get rich quickly with this special offer!", 1),
    ("FREE gift card! Click here to win!", 1),
    ("Earn extra cash from home today!", 1),
    ("LIMITED OFFER!!! Win a cash prize!", 1),
    ("Claim your FREE reward before midnight!", 1),
    ("Hi, can we meet for lunch tomorrow?", 0),
    ("Your appointment is confirmed for Monday.", 0),
    ("Please review the meeting notes.", 0),
    ("Thank you for your help with the project.", 0),
    ("The class assignment is due on Friday.", 0),
    ("Can you send me the document?", 0),
    ("Your package has been delivered.", 0),
    ("Let's discuss the schedule tomorrow.", 0),
]

money_terms = {"cash", "money", "prize", "reward", "rich"}
urgent_terms = {"urgent", "now", "today", "midnight"}

def extract_features(message):
    words = message.split()
    cleaned = [w.strip("!.,?").lower() for w in words]
    return {
        "word_count": len(words),
        "exclamations": message.count("!"),
        "money_words": sum(w in money_terms for w in cleaned),
        "all_caps": sum(w.isupper() and w.isalpha() for w in words),
        "urgent_words": sum(w in urgent_terms for w in cleaned),
        "question_marks": message.count("?"),
    }

rows = []
for message, label in emails:
    row = extract_features(message)
    row["label"] = label
    rows.append(row)

Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("data/emails.csv", index=False)

print("Created data/emails.csv with 16 emails.")
