# Text preprocessing for the Flight Ticket Booking Complaint Classification project:
#1. Load raw dataset
#2. Combine Title and Reviews
#3. Clean text
#4. Remove stopwords
#5. Apply lemmatization
#6. Save processed dataset


import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources (only needed the first time)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize stopword list and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


#Perform all preprocessing steps on a single text.
#Parameters
#    ----text : str
#Returns
#    ----str
#    Cleaned and normalized text.

def preprocess(text):
    # Handle missing values
    if pd.isna(text):
        return ""
    # Convert to string
    text = str(text)
    # Convert all characters to lowercase
    text = text.lower()
    # Fix common encoding issues found in the dataset
    text = text.replace("Â", " ")
    text = text.replace("â€œ", '"')
    text = text.replace("â€", '"')
    text = text.replace("â€™", "'")
    text = text.replace("â€˜", "'")
    text = text.replace("â€", '"')
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize text
    words = text.split()
    # Remove English stopwords
    words = [word
        for word in words
        if word not in stop_words]
    # Remove single-character tokens
    words = [
        word
        for word in words
        if len(word) > 1]
    # Apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words]
    # Convert tokens back to sentence
    text = " ".join(words)
    return text


# Load Dataset
df = pd.read_csv("data/labeled/airlines_reviews_labeled.csv")
# Create input text by combining Title and Reviews
df["Text"] = (
    df["Title"].fillna("")
    + ". "
    + df["Reviews"].fillna(""))
# Apply preprocessing
df["Processed_Text"] = df["Text"].apply(preprocess)
# Save processed dataset
df.to_csv("data/processed/complaints_preprocessed.csv",index=False)
print("Preprocessing completed.")