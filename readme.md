# Flight Ticket Booking Complaint Classification

A Machine Learning project for automatically classifying airline customer complaints into predefined complaint categories using traditional Natural Language Processing (NLP) and Machine Learning techniques.

>  This project is currently under active development.

---

# Project Overview

Customer complaints contain valuable information that helps airlines identify recurring service issues and improve customer satisfaction. The goal of this project is to develop a multi-class text classification model capable of predicting the primary category of an airline customer's complaint. Unlike sentiment analysis, this project focuses on identifying **what the customer is complaining about**, rather than whether the complaint is positive or negative.

---

# Dataset

The dataset consists of real airline customer reviews. Only reviews containing actual complaints are included in the final dataset. Each complaint belongs to **exactly one** complaint category. The raw dataset intentionally keeps the original **Title** and **Reviews** fields unchanged. During preprocessing, these fields are merged into a single input text using the following format:

```
Title + ". " + Reviews
```
This preprocessing step is performed programmatically to ensure a clean and reproducible machine learning pipeline.

---

# Complaint Categories

The dataset contains the following complaint categories:
- Booking
- Payment
- Refund
- Flight Delay
- Flight Cancellation
- Baggage
- Check-in
- Seat & Cabin
- Customer Service
- Other

Detailed annotation rules are available in:

**labeling_guidelines.md**

---

# Labeling Strategy

The dataset was annotated using a hybrid approach.

### Phase 1

Manual annotation was performed on an initial subset of complaints in order to:

- define annotation rules
- validate category definitions
- reduce ambiguity
- establish labeling consistency

### Phase 2

The remaining complaints were automatically labeled using a rule-based weak supervision approach based on weighted keywords and handcrafted rules. Low-confidence samples were flagged for manual review.

---

# Project Structure

```
Flight_ticket_booking_complaint_classification/

│
├── data/
│   ├── raw/
│   ├── labeled/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│
├── models/
│
├── labeling_guidelines.md
├── requirements.txt
├── README.md
└── main.py
```

---

# Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Data Preprocessing
      │
      ├── Missing Value Handling
      ├── Merge Title + Reviews
      ├── Text Cleaning
      ├── Tokenization
      └── TF-IDF Vectorization
      │
      ▼
Train / Validation Split
      │
      ▼
Model Training
      │
      ▼
Evaluation
```

---

# Current Progress

✅ Dataset collection

✅ Complaint filtering

✅ Manual annotation

✅ Rule-based automatic labeling

✅ Annotation guideline creation

✅ Text preprocessing

✅ Feature engineering

⬜ Model training

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

Additional libraries may be introduced as the project evolves.

---

# Future Work

- Text preprocessing
- Data cleaning
- TF-IDF vectorization
- Baseline machine learning models
- Hyperparameter optimization
- Performance comparison
- Error analysis
- Model persistence
- Prediction interface

---

# License

This project is intended for educational and research purposes.

The original review data belongs to its respective source.

---

# Contributions

Suggestions, improvements and pull requests are always welcome.