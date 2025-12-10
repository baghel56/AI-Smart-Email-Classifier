# main.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('dataset/synthetic_emails.csv')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Combine subject + body
df['processed_text'] = (df['subject'] + ' ' + df['body']).apply(preprocess)

# Encode labels
cat_encoder = LabelEncoder()
df['category_label'] = cat_encoder.fit_transform(df['category'])

urg_encoder = LabelEncoder()
df['urgency_label'] = urg_encoder.fit_transform(df['urgency'])

# Display processed dataset
print(df[['email_id','processed_text','category','category_label','urgency','urgency_label']].head())

# Save processed dataset
df.to_csv('dataset/processed_emails.csv', index=False)
print("Processed dataset saved as processed_emails.csv")
