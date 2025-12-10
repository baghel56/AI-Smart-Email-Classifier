# src/add_urgency.py
import pandas as pd

# Load dataset
df = pd.read_csv("synthetic_emails.csv")

# Function to assign urgency
def assign_urgency(row):
    text = (row['subject'] + " " + row['body']).lower()
    # High urgency keywords
    if any(word in text for word in ["not received", "damaged", "failed", "immediately"]):
        return "High"
    # Medium urgency keywords
    elif any(word in text for word in ["invoice", "update", "change"]):
        return "Medium"
    # Low urgency by default
    else:
        return "Low"

# Apply function
df['urgency'] = df.apply(assign_urgency, axis=1)

# Save new CSV
df.to_csv("synthetic_emails_with_urgency.csv", index=False)
print("Urgency column added and saved at synthetic_emails_with_urgency.csv")
