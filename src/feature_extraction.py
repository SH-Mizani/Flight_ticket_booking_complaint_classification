# Converts preprocessed complaint text into numerical feature vectors using TF-IDF.
#1. Load preprocessed dataset and extract text
#2. Convert text into TF-IDF features
#3. Save feature matrix


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Load dataset
df = pd.read_csv("data/processed/complaints_preprocessed.csv")
# Extract text and labels
x_text = df["Processed_Text"]
y = df["Category"]
# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    lowercase=False,
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90)

# Convert text into vectors
x = vectorizer.fit_transform(x_text)
# Convert sparse matrix to DataFrame
feature_names = vectorizer.get_feature_names_out()
X_df = pd.DataFrame(
    x.toarray(),
    columns=feature_names)
# Append labels
X_df["Category"] = y.values
# Save extracted features
X_df.to_csv("data/processed/features.csv",index=False)
print("Feature extraction completed.")