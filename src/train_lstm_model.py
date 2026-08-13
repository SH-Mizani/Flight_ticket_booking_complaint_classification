# Trains a Bidirectional LSTM model on the preprocessed complaint text
#1. Load preprocessed text (not TF-IDF features -> raw/cleaned text this time)
#2. Train test split
#3. Tokenize + pad sequences, encode labels
#4. Build and train a Bidirectional LSTM
#5. Evaluate with accuracy, weighted F1, macro F1
#6. Save comparison row + model + tokenizer

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

#Paths

processed_path = Path("data/processed/complaints_preprocessed.csv")
results_path = Path("data/processed")
models_path = Path("models")
models_path.mkdir(parents = True, exist_ok = True)

#Load preprocessed text
df = pd.read_csv(processed_path)
x_text = df["Processed_Text"].astype(str)
y_raw = df["Category"]

print(f"Loaded data: {x_text.shape[0]} rows")
print("\nClass distribution:\n", y_raw.value_counts())

#Encode labels to integers (needed for the model + class_weight)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

#Train test split
x_train_text, x_test_text, y_train, y_test = train_test_split(
    x_text, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y,)

#Tokenize
vocab_size = 10000
max_len = 100

tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
tokenizer.fit_on_texts(x_train_text)

x_train_seq = tokenizer.texts_to_sequences(x_train_text)
x_test_seq = tokenizer.texts_to_sequences(x_test_text)

x_train_pad = pad_sequences(x_train_seq, maxlen = max_len, padding = "post", truncating = "post")
x_test_pad = pad_sequences(x_test_seq, maxlen = max_len, padding = "post", truncating = "post")

#Class weights (imbalance)
class_weights_array = compute_class_weight(
    class_weight = "balanced",
    classes = np.unique(y_train),
    y = y_train,
)
class_weight_dict = dict(enumerate(class_weights_array))

#Build model
embedding_dim = 128

model = Sequential([
    Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_len),
    Bidirectional(LSTM(64, return_sequences = False)),
    Dropout(0.5),
    Dense(64, activation = "relu"),
    Dropout(0.3),
    Dense(num_classes, activation = "softmax"),
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"],
)

model.summary()

#Train
early_stop = EarlyStopping(
    monitor = "val_loss",
    patience = 3,
    restore_best_weights = True,
)

history = model.fit(
    x_train_pad, y_train,
    validation_split = 0.1,
    epochs = 15,
    batch_size = 32,
    class_weight = class_weight_dict,
    callbacks = [early_stop],
)

#Evaluate
y_pred_probs = model.predict(x_test_pad)
y_pred = np.argmax(y_pred_probs, axis = 1)

acc = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average = "weighted")
f1_macro = f1_score(y_test, y_pred, average = "macro")

print(f"\nAccuracy:      {acc:.3f}")
print(f"Weighted F1:   {f1_weighted:.3f}")
print(f"Macro F1:      {f1_macro:.3f}")
print("\nClassification report:\n", classification_report(
    y_test, y_pred, target_names = label_encoder.classes_, zero_division = 0
))

#Append this result to the comparison table (created earlier for the classical models)
comparison_file = results_path / "model_comparison_all.csv"
new_row = pd.DataFrame([{
    "Model": "Bidirectional LSTM",
    "Accuracy": round(acc, 3),
    "Weighted_F1": round(f1_weighted, 3),
    "Macro_F1": round(f1_macro, 3),
}])

if comparison_file.exists():
    existing = pd.read_csv(comparison_file)
    combined = pd.concat([existing, new_row], ignore_index = True)
else:
    tfidf_file = results_path / "tfidf_model_comparison.csv"
    existing = pd.read_csv(tfidf_file) if tfidf_file.exists() else pd.DataFrame()
    combined = pd.concat([existing, new_row], ignore_index = True)

combined = combined.sort_values("Weighted_F1", ascending = False)
combined.to_csv(comparison_file, index = False)

print("\nUpdated comparison table:")
print(combined.to_string(index = False))

#Save model + tokenizer + label encoder (needed together to run inference later)
model.save(models_path / "lstm_model.keras")

with open(models_path / "lstm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

joblib.dump(label_encoder, models_path / "lstm_label_encoder.joblib")

print(f"\nModel saved to {models_path / 'lstm_model.keras'}")
print(f"Tokenizer saved to {models_path / 'lstm_tokenizer.pkl'}")
print(f"Label encoder saved to {models_path / 'lstm_label_encoder.joblib'}")
