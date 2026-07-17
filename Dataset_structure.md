# Dataset Structure

Each row in the dataset represents **one customer complaint**.

The final dataset contains the following columns.

| Column | Description |
|---------|-------------|
| Title | Short title or summary of the complaint. |
| Airline | Airline name. |
| Reviews | Original customer review text. |
| Overall Rating | Customer rating (1–10). |
| Recommended | Whether the customer recommends the airline (Yes/No). |
| Category | Final complaint category assigned to the sample. |
| Label_Confidence | Confidence score (0–1) for automatically generated labels. Manually labeled samples receive 1.0. |
| Needs_Review | Indicates whether manual verification is recommended. |

---

## Model Input

The raw dataset intentionally keeps the **Title** and **Reviews** fields separate.

During the preprocessing stage, these two fields are merged to create the input text used for model training.

The merged text follows the format:

Title + ". " + Reviews

This preprocessing step is performed programmatically and is **not stored in the original dataset**, keeping the raw data unchanged and reproducible.

---

## Category Column

The **Category** column stores the final complaint label.

Each complaint belongs to exactly one of the predefined categories.

---

## Confidence Columns

Two additional columns are included to improve annotation quality.

### Label_Confidence

Represents the confidence of the assigned label.

- **1.0** → manually verified label
- **0–1** → automatically generated confidence score

### Needs_Review

Indicates whether the automatically assigned label should be manually inspected.

Possible values:

- False
- True