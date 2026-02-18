Online Purchase Orders Classification
Project Overview

This project aims to predict the risk of default payment for online purchase orders using machine learning techniques.
The dataset contains 30,000 online purchase orders, each described by 44 attributes, covering customer behavior, payment methods, order history, and risk indicators.

The business goal is to help an online trader identify high-risk customers before order fulfillment, enabling better risk management and reduced financial loss.

Problem Statement

Each order is classified into one of two categories:

yes → High risk of default payment

no → Low risk of default payment

This is a binary classification problem, where the objective is to learn patterns that distinguish risky customers from reliable ones.

Dataset Description

Dataset: risk-dataset.txt

Records: ~30,000

Features: 44

Target variable: CLASS

Attribute definitions: risk-attributes.txt

Each row represents one online purchase order.

Target Variable

CLASS

yes → High risk

no → Low risk

This column is used as the label for supervised learning.

Initial Data Cleaning
Identifier & Non-informative Columns Removed

ORDER_ID
A unique identifier with no predictive value.

Redundant columns removed:

ANUMMER_01, ANUMMER_02, ANUMMER_03, ANUMMER_04, ANUMMER_05,
ANUMMER_06, ANUMMER_07, ANUMMER_08, ANUMMER_09, ANUMMER_10

Handling Missing Values

Missing values in the dataset were represented using the symbol ?.
The following columns contained missing values:

Z_CARD_ART, DATE_LORDER, MAHN_AKT, MAHN_HOECHST,
Z_LAST_NAME, B_BIRTHDATE, TIME_ORDER


Instead of applying a single imputation strategy, each column was analyzed and handled based on business logic and data context.

1. Z_CARD_VALID – Card Expiry Date

Represents the expiration date of the card

In raw form, not directly useful for prediction

Feature Engineering Applied:

Identified the maximum DATE_LORDER as a reference point

Computed a new numerical feature:

CARD_MONTHS_TO_EXPIRY
→ Number of months remaining until card expiration at the time of last order

This transformed a raw date field into a meaningful numerical risk indicator.

2. Z_CARD_ART – Type of Card

Missing values were investigated rather than blindly imputed

Analysis showed:

Missing values occur only when payment was not made by card

These orders were paid via check or debit card

Imputation Strategy:

Missing values replaced with:
not_a_card_payment

This column is treated as a categorical feature.

3. Z_LAST_NAME – Customer Last Name

Missing values initially appeared suspicious

Investigation showed:

All rows with missing last name used Z_METHODE = check (14,808 rows)

Interpretation:

Customers paying by check may not be required to provide a name

Some records likely originate from legacy or offline workflows

Imputation Strategy:

Missing values replaced with:
not_applicable

This column is treated as a categorical feature.

4. MAHN_AKT & MAHN_HOECHST – Reminder Stages

These columns track reminder history for unpaid orders

Root Cause Analysis:

15,032 missing values correspond to new customers

Remaining missing values occur when:

Customers paid via debit card

Reminder system was not applicable (no online credit risk)

Imputation Strategy:

Missing values replaced with -1

Indicates “reminder not applicable”

Preserves numeric meaning without fabricating risk signals

5. DATE_LORDER – Last Order Date

Missing values analyzed

Found to strongly correlate with new customers

Decision:

No imputation applied

Missing values retained as NULL

Feature Engineering Applied:

Created a new numerical feature:

DAYS_SINCE_LAST_ORDER
→ Number of days since the last order relative to the reference date

6. B_BIRTHDATE – Date of Birth

Not a mandatory field

Missing values expected and non-random

Feature Engineering Applied:

Extracted Age (numerical)

Derived Age Group (categorical)

This allowed retention of demographic signal without relying on raw dates.

7. TIME_ORDER – Order Time

Feature Engineering Applied:

Extracted Order Hour (order_hour) → Numerical

Derived Order Part of Day (ORDER_PART_OF_DAY) → Categorical
(e.g., morning, afternoon, evening, night)

This captures customer behavior patterns related to purchase timing.
