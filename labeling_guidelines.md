# Complaint Labeling Guidelines

## Overview

This document describes the labeling strategy used for the Flight Ticket Booking Complaint Classification project. The objective is to classify each customer complaint into one and only one predefined category based on its primary issue. If a complaint contains multiple issues, the label should represent the customer's main concern.

# Label Categories

## 1. Booking

Description:
Problems related to booking creation, modification, cancellation, ticket information, reservation errors or booking confirmation.

Examples:

- Unable to change my booking.
- Wrong passenger information.
- Booking confirmation never arrived.

---

## 2. Payment

Description:
Any payment-related issue including failed payment, duplicate payment, billing problems or transaction errors.

Examples:

- My credit card was charged twice.
- Payment failed but money was deducted.

---

## 3. Refund

Description:
Complaints regarding refund requests, delayed refunds or refund rejection.

Examples:

- I still haven't received my refund.
- Refund request has been pending for weeks.

---

## 4. Flight Delay

Description:
Complaints caused by delayed departures or arrivals.

Examples:

- My flight was delayed for five hours.
- The plane departed much later than scheduled.

---

## 5. Flight Cancellation

Description:
Complaints related to flight cancellations.

Examples:

- My flight was cancelled without notice.
- Airline cancelled my booking.

---

## 6. Baggage

Description:
Lost baggage, delayed baggage or damaged luggage.

Examples:

- My suitcase never arrived.
- My luggage was damaged.

---

## 7. Check-in

Description:
Problems during online check-in or airport check-in.

Examples:

- Online check-in did not work.
- I couldn't print my boarding pass.

---

## 8. Seat & Cabin

Description:
Seat assignment, cabin cleanliness, seat comfort, upgrades or onboard seating issues.

Examples:

- I paid for extra legroom.
- Dirty seats.

---

## 9. Customer Service

Description:
Complaints about airline staff, call center or customer support.

Examples:

- Customer support never answered.
- Staff behaved rudely.

---

## 10. Other

Description:
Complaints that do not clearly belong to any previous category.

---

# Labeling Rules

## Rule 1

Each complaint receives only one label.

## Rule 2

Choose the label representing the primary issue.

Example
Flight delayed and baggage lost.
Primary complaint:
Flight Delay

## Rule 3

If the complaint mainly requests money back, use Refund even if other problems exist.
Example
My flight was cancelled and I still haven't received my refund.
Label: Refund

## Rule 4

If confidence is low, leave a note for manual review.

---

# Labeling Workflow

Review Text

↓

Identify Main Issue

↓

Assign One Category

↓

Human Review

↓

Final Label