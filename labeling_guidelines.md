# Complaint Labeling Guidelines

## Overview

This document describes the annotation methodology used in the **Flight Ticket Booking Complaint Classification** project. The objective is to assign **one and only one complaint category** to each airline customer complaint based on the customer's primary issue. Although a complaint may mention multiple problems, only the **main complaint** should determine the final label.

The annotation process combines **manual labeling** with **rule-based weak supervision** to efficiently construct a large-scale labeled dataset while maintaining annotation consistency.

---

# Labeling Strategy

The dataset was annotated using a hybrid annotation strategy consisting of two complementary phases.

## Phase 1 – Manual Annotation

An initial subset of complaints was manually reviewed and labeled. This stage was performed to:

- define clear annotation rules;
- validate complaint categories;
- reduce ambiguity between similar complaint types;
- establish consistent labeling decisions;
- create high-quality reference examples.

These manually annotated samples serve as the reference for the automatic annotation stage.

---

## Phase 2 – Rule-Based Weak Supervision

Weak supervision refers to automatically assigning labels using handcrafted rules instead of manual human annotation. After defining the annotation policy, the remaining unlabeled complaints were automatically labeled using a **rule-based weak supervision** approach. The automatic labeling system follows exactly the same complaint definitions and labeling rules described in this document. Low-confidence predictions are flagged for manual review. Automatically generated labels should be considered **silver labels** rather than manually verified ground truth.

---

# Annotation Principles

The following principles were applied throughout the annotation process.

- Each complaint receives exactly **one** category.
- The assigned label must represent the customer's primary complaint.
- Similar complaints should receive consistent labels.
- Operational issues take precedence over general dissatisfaction whenever appropriate.
- Refund-related requests receive higher priority when the customer's main objective is obtaining reimbursement.
- Low-confidence annotations should always be reviewed manually.

---

# Complaint Categories

## 1. Booking

### Description

Problems related to booking creation, reservation changes, ticket modifications, booking confirmation or passenger information.

### Examples

- Unable to change my booking.
- Wrong passenger information.
- Booking confirmation never arrived.

---

## 2. Payment

### Description

Issues involving payment failures, duplicate charges, billing problems or transaction errors.

### Examples

- My credit card was charged twice.
- Payment failed but money was deducted.

---

## 3. Refund

### Description

Complaints regarding refund requests, delayed refunds or rejected refunds.

### Examples

- I still haven't received my refund.
- My refund request has been pending for weeks.

---

## 4. Flight Delay

### Description

Complaints caused by delayed departures or arrivals.

### Examples

- My flight was delayed for five hours.
- The aircraft departed much later than scheduled.

---

## 5. Flight Cancellation

### Description

Complaints caused by cancelled flights.

### Examples

- My flight was cancelled without notice.
- Airline cancelled my booking.

---

## 6. Baggage

### Description

Complaints related to lost, delayed or damaged baggage.

### Examples

- My suitcase never arrived.
- My baggage was damaged during transport.

---

## 7. Check-in

### Description

Problems encountered during online or airport check-in.

### Examples

- Online check-in failed.
- I couldn't print my boarding pass.

---

## 8. Seat & Cabin

### Description

Complaints regarding seat comfort, seat assignment, cabin cleanliness or onboard seating conditions.

### Examples

- My seat was broken.
- Very uncomfortable seats.
- Dirty cabin.
- I paid for extra legroom but didn't receive it.

---

## 9. Customer Service

### Description

Complaints involving airline staff, cabin crew, call centers or customer support interactions.

### Examples

- Customer support never answered.
- Staff behaved rudely.
- Cabin crew ignored passengers.

---

## 10. Other

### Description

Complaints that do not clearly belong to any predefined complaint category.

---

# Labeling Rules

## Rule 1 — Single Label

Each complaint must receive **exactly one** category.
Multi-label annotation is not used in this project.

---

## Rule 2 — Primary Complaint

When multiple issues appear in the same complaint, assign the label corresponding to the customer's primary concern.

### Example

Complaint:

> My flight was delayed and my baggage arrived late.

Assigned label:

**Flight Delay**

---

## Rule 3 — Refund Priority

If the customer's main objective is obtaining reimbursement, assign the **Refund** category even when another operational issue is also mentioned.

### Example

Complaint:

> My flight was cancelled and I still haven't received my refund.

Assigned label:

**Refund**

---

## Rule 4 — Manual Review

If the assigned label has low confidence or the complaint is ambiguous, it should be flagged for manual review.

---

# Automatic Labeling Method

The automatic annotation system uses a **rule-based weak supervision** approach.
For each unlabeled complaint:

1. Generate the model input by combining the complaint title and review text.
2. Search for category-specific keywords and phrases.
3. Assign different weights to keywords based on their importance.
4. Compute a score for every complaint category.
5. Reduce the influence of repeated keyword matches.
6. Assign the category with the highest final score.

This process enables scalable annotation while preserving consistency with the manual labeling policy.

---

## Refund Priority Rule

Refund-related complaints receive additional weighting whenever refund expressions appear together with unresolved reimbursement requests.
Typical indicators include:

- refund
- reimburse
- reimbursement
- money back
- still waiting
- pending
- denied
- haven't received
This rule follows **Rule 3** of the annotation policy.

---

## Tie-Breaking Strategy

If multiple categories receive similar scores, the following priority order is applied:

1. Refund
2. Flight Cancellation
3. Flight Delay
4. Baggage
5. Check-in
6. Booking
7. Payment
8. Seat & Cabin
9. Customer Service
10. Other

This priority order helps maintain annotation consistency across ambiguous complaints.

---

## No Matching Rule

If no meaningful keywords or complaint patterns are detected, the complaint is assigned to:

**Other**

---

# Confidence Estimation

Each automatically generated label includes an estimated confidence level.

## Label_Confidence

A confidence score ranging from **0.0** to **1.0**. The score is calculated using the normalized difference between the highest-scoring category and the second-highest category. Higher values indicate stronger confidence in the assigned label. Manually annotated complaints receive a confidence score of **1.0**.

---

## Needs_Review

A Boolean flag indicating whether manual verification is recommended.
A complaint is marked as **Needs_Review = True** when:

- the confidence score is low;
- no strong category-specific keywords are detected;
- multiple categories receive similar scores.

This follows **Rule 4** of the annotation policy.

---

# Dataset Annotation Summary

The final dataset combines manually verified annotations with automatically generated labels.

| Annotation Phase | Method |
|------------------|--------|
| Phase 1 | Manual Annotation |
| Phase 2 | Rule-Based Weak Supervision |

The manually labeled complaints remain unchanged throughout the annotation process. Automatically generated labels should be interpreted as **silver labels**, not manually verified ground truth. Before training or evaluating machine learning models, manually reviewing a representative subset of low-confidence samples is strongly recommended.

---

# Annotation Workflow

```
                    Raw Complaint
                          │
                          ▼
               Manual Annotation (Initial Set)
                          │
                          ▼
             Annotation Rules Validation
                          │
                          ▼
           Rule-Based Weak Supervision
                          │
                          ▼
               Confidence Estimation
                          │
                          ▼
                 Needs_Review ?
                    │          │
                  Yes         No
                   │           │
                   ▼           ▼
            Manual Review   Final Label
                   │
                   ▼
             Final Labeled Dataset
```

---

# Notes

These guidelines were developed to ensure a consistent, transparent and reproducible annotation process for airline customer complaints. The combination of manual annotation and rule-based weak supervision enables efficient large-scale dataset creation while preserving annotation quality. Automatically generated labels should be regarded as **silver labels**, whereas manually reviewed annotations represent the highest-quality reference labels available in this project. Future versions of the dataset may incorporate additional manual verification or active learning strategies to further improve annotation quality.