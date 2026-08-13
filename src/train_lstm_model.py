#Trains a Bidirectional LSTM model on the preprocessed complaint text
#1. Load preprocessed text
#2. Encode labels
#3. Split data into training and testing sets
#4. Build a TextVectorization layer using only training data
#5. Convert text into padded integer sequences
#6. Build and train a Bidirectional LSTM
#7. Evaluate with Accuracy, Weighted F1 and Macro F1
#8. Save comparison results, model and label encoder


import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report,

from keras.models import Sequential
from keras.layers import (
    TextVectorization,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
from keras.callbacks import EarlyStopping


#Paths
processed_path = Path("data/processed/complaints_preprocessed.csv")
results_path = Path("data/processed")
models_path = Path("models")
models_path.mkdir(parents=True, exist_ok=True)

#Load preprocessed text
df = pd.read_csv(processed_path)
x_text = df["Processed_Text"].astype(str)
y_raw = df["Category"]
print(f"Loaded data: {x_text.shape[0]} rows")
print("\nClass distribution:\n",y_raw.value_counts())

#Encode labels
#LSTM outputs integer class IDs.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
print("\nClasses:",list(label_encoder.classes_))

#Train Test Split
x_train_text, x_test_text, y_train, y_test = train_test_split(
    x_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"\nTraining samples: {len(x_train_text)}")
print(f"Testing samples:  {len(x_test_text)}")

# Text Vectorization
# pad_sequences pipeline.
vocab_size = 10000
max_len = 100
vectorizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_len,
    standardize=None,
)
vectorizer.adapt(x_train_text.to_numpy())
x_train_seq = vectorizer(x_train_text.to_numpy())
x_test_seq = vectorizer(x_test_text.to_numpy())
print(f"\nVocabulary size: {len(vectorizer.get_vocabulary())}")
print(f"Sequence length: {x_train_seq.shape[1]}")


# Class Weights
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weight_dict = dict(enumerate(class_weights_array))
print("\nClass weights:",class_weight_dict)

# Build Bidirectional LSTM Model
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=vocab_size,output_dim=embedding_dim,),
    Bidirectional(LSTM(64,return_sequences=False)),
    Dropout(0.5),
    Dense(64,activation="relu"),
    Dropout(0.3),
    Dense(num_classes,activation="softmax"),
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Display model architecture.
model.summary()
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

# Train Model
history = model.fit(
    x_train_seq,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
)

# Evaluate Model
y_pred_probs = model.predict(x_test_seq,verbose=0)
y_pred = np.argmax(y_pred_probs,axis=1)

# Accuracy
acc = accuracy_score(y_test,y_pred)
f1_weighted = f1_score(y_test,y_pred,average="weighted")
f1_macro = f1_score(y_test,y_pred,average="macro")
print(f"\nAccuracy:      {acc:.3f}")
print(f"Weighted F1:   {f1_weighted:.3f}")
print(f"Macro F1:      {f1_macro:.3f}")

# Classification Report
print("\nClassification report:\n",
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

# Append Result to Model Comparison Table
comparison_file = (results_path / "model_comparison_all.csv")
new_row = pd.DataFrame([
    {   "Model": "Bidirectional LSTM",
        "Accuracy": round(acc, 3),
        "Weighted_F1": round(f1_weighted, 3),
        "Macro_F1": round(f1_macro, 3),
    }
])
if comparison_file.exists():
    existing = pd.read_csv(comparison_file)
    combined = pd.concat([existing, new_row],ignore_index=True)
else:
    tfidf_file = (results_path / "tfidf_model_comparison.csv")
    if tfidf_file.exists():
        existing = pd.read_csv(tfidf_file)
    else:
        existing = pd.DataFrame()
    combined = pd.concat([existing, new_row],ignore_index=True)

# Sort models by Weighted F1.
combined = combined.sort_values("Weighted_F1",ascending=False)

# Save comparison table.
combined.to_csv(comparison_file,index=False)
print("\nUpdated comparison table:")
print(combined.to_string(index=False))
model_path = (models_path / "lstm_model.keras")
model.save(model_path)
vocabulary_path = (models_path / "lstm_vocabulary.txt")
with open(
    vocabulary_path,
    "w",
    encoding="utf-8"
) as f:
    for word in vectorizer.get_vocabulary():
        f.write(word + "\n")
label_encoder_path = (models_path / "lstm_label_encoder.joblib")
joblib.dump(label_encoder,label_encoder_path)

#Output
print(f"\nModel saved to: {model_path}")
print(f"Vocabulary saved to: {vocabulary_path}")
print(f"Label encoder saved to: {label_encoder_path}")
print("\nLSTM training completed successfully.")