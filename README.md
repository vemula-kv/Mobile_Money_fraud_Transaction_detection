# Mobile Money Fraud Transaction Detection

A Python-based machine learning project that explores synthetic
mobile-money transaction data and builds a Logistic Regression
classifier to identify potentially fraudulent transactions.

## Project Overview

The project analyzes transaction attributes and account-balance
movements to understand patterns associated with fraudulent activity. It
includes data inspection, exploratory data analysis (EDA),
preprocessing, feature engineering, visual analysis, model training, and
evaluation.

**Important:** The notebook uses a **synthetic dataset**. The results
are an educational/modeling exercise and should not be interpreted as
evidence of performance on real bank transactions or a production fraud
system.

## Dataset

-   **Dataset:** [Synthetic Mobile Money Transaction
    Dataset](https://www.kaggle.com/datasets/denishazamuke/synthetic-mobile-money-transaction-dataset)
-   **Source used in notebook:**
    `denishazamuke/synthetic-mobile-money-transaction-dataset` via
    `kagglehub`
-   **Records loaded:** 4,225,958
-   **Columns:** 10
-   **Target:** `isFraud` (1 = fraud, 0 = not fraud)

The notebook loads the CSV file
`MoMTSim_20240722202413_1000_dataset.csv`.

### Main fields

  -----------------------------------------------------------------------
  Field                               Description
  ----------------------------------- -----------------------------------
  `step`                              Transaction time-step field
                                      supplied by the dataset

  `transactionType`                   Transaction category (for example,
                                      payment or transfer)

  `amount`                            Transaction amount

  `initiator`                         Initiating account identifier

  `oldBalInitiator`                   Initiator balance before the
                                      transaction

  `newBalInitiator`                   Initiator balance after the
                                      transaction

  `recipient`                         Recipient account identifier

  `oldBalRecipient`                   Recipient balance before the
                                      transaction

  `newBalRecipient`                   Recipient balance after the
                                      transaction

  `isFraud`                           Fraud target label
  -----------------------------------------------------------------------

## Workflow

### 1. Data loading and inspection

The notebook downloads the dataset with `kagglehub`, loads it into a
Pandas DataFrame, and inspects its structure and descriptive statistics.

### 2. Data quality checks

-   Checks for missing values.
-   Checks for duplicate rows and applies `drop_duplicates()`.
-   Reviews transaction-type and fraud-label distributions.

The notebook reports no null values and no duplicate rows in the
initially loaded dataset.

### 3. Exploratory data analysis

The analysis includes: - Transaction-type distribution. - Fraud versus
non-fraud distribution. - Descriptive statistics. - Correlation matrix
and heatmap. - Transaction amount comparison by fraud label.

### 4. Preprocessing and feature engineering

The notebook label-encodes `transactionType` and `recipient`, then
creates additional transaction features:

-   `balanceChangeInitiator`: initiator's new balance minus old balance.
-   `balanceChangeRecipient`: recipient's new balance minus old balance.
-   `amountToBalanceRatioInitiator`: transaction amount divided by the
    initiator's old balance (with a small epsilon to avoid division by
    zero).
-   `transactionSizeRelativeToBalance`: calculated using the same
    formula as `amountToBalanceRatioInitiator` in the current notebook.

### 5. Model development

-   Separates `isFraud` as the target variable.
-   Uses a 70% training / 30% test split with `random_state=42`.
-   Scales features with `StandardScaler`, fitting the scaler on the
    training data and applying it to the test data.
-   Trains a Scikit-learn `LogisticRegression` classifier.
-   Evaluates predictions on the test set.
-   Fits a second Logistic Regression model with
    `class_weight='balanced'` to inspect the absolute model coefficients
    as feature-importance indicators.

## Results

The notebook reports the following results for its Logistic Regression
evaluation:

  Metric        Non-fraud (class 0)   Fraud (class 1)
  ----------- --------------------- -----------------
  Precision                    0.97              0.93
  Recall                       0.91              0.98
  F1-score                     0.94              0.95

**Overall test accuracy: 94.69%**

Confusion matrix (actual rows, predicted columns):

                       Predicted non-fraud   Predicted fraud
  ------------------ --------------------- -----------------
  Actual non-fraud                 545,820            51,756
  Actual fraud                      15,515           654,691

The fraud-class recall reported by the notebook is 0.98, meaning the
model identified a high proportion of the fraud-labeled records in this
test split. Precision and recall should be considered together: false
positives can create unnecessary reviews, while false negatives can
allow fraudulent transactions to go undetected.

## Technologies

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   KaggleHub
-   Jupyter Notebook

## How to Run

1.  Clone this repository:

    ``` bash
    git clone https://github.com/vemula-kv/Mobile_Money_fraud_Transaction_detection.git
    cd Mobile_Money_fraud_Transaction_detection
    ```

2.  Create and activate a Python virtual environment (recommended).

3.  Install the packages used by the notebook:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
    ```

4.  Launch Jupyter:

    ``` bash
    jupyter notebook
    ```

5.  Open `Synthetic_Mobile_Money_Transaction.ipynb` and run the cells in
    order. The notebook uses `kagglehub` to download the dataset.

Depending on your Kaggle access and environment configuration, you may
need to configure Kaggle authentication before downloading the dataset.

## Limitations and Potential Improvements

This repository demonstrates an analytical workflow on synthetic data;
it does not document a production deployment or live
transaction-monitoring integration.

Potential next steps include:

-   Use a stratified or time-based train/test split, as appropriate to
    the intended use case.
-   Compare Logistic Regression with tree-based models using
    implemented, reproducible experiments.
-   Review categorical encoding for nominal transaction and account
    identifiers.
-   Remove or redesign the duplicate ratio feature, since the two ratio
    features currently use the same calculation.
-   Evaluate precision-recall trade-offs and threshold tuning in light
    of the cost of false positives and false negatives.
-   Add cross-validation, calibration, and additional model diagnostics.
-   Validate performance on representative real-world data before
    considering operational use.

## Repository Contents

``` text
Mobile_Money_fraud_Transaction_detection/
└── Synthetic_Mobile_Money_Transaction.ipynb
```

## Author

**Vinod Kumar Vemula**

GitHub: [@vemula-kv](https://github.com/vemula-kv)
