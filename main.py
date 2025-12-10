
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('synthetic_emails.csv')  

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = str(text).lower()  # convert to lowercase
    text = re.sub(r'\S+@\S+', '', text)       # remove email addresses
    text = re.sub(r'http\S+|www\S+', '', text) # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)       # remove punctuation and numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Combine subject + body and apply preprocessing
df['processed_text'] = (df['subject'] + ' ' + df['body']).apply(preprocess)


print(df[['subject', 'body', 'processed_text']].head())

# Save processed data to new CSV
df.to_csv('processed_emails.csv', index=False)
print("Preprocessing complete. Saved to 'processed_emails.csv'.")



