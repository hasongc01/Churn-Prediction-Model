
# Digital Marketplace Fraudulent Customer Classification Detection Model

## 1. Business Problem

Digital marketplaces face fraud from fake accounts, stolen credentials, or chargebacks. Early detection reduces financial loss and improves platform trust. Manual review does not scale with volume.

## 2. Data Understanding

Data is merged on `consumer_id` from three folders:

- **User** ([`datasets/user/`](datasets/user/)): `user.csv`, `delivery_address.csv`, `email.csv` — password history, email traits, delivery addresses, tenure
- **Item** ([`datasets/item/`](datasets/item/)): `latest_item.csv` — order amount, category, product title, quantity, tag count
- **Activity** ([`datasets/activity/`](datasets/activity/)): `per_day.csv`, `per_week.csv`, `per_month.csv` — transactions, add-to-cart, purchase totals, payment changes, devices per user, etc.

**Key stats** (from EDA):

- ~16,485 consumers, ~6.5% fraudulent (class imbalance)
- 29 features after merge
- High missingness in activity features (many users have no per_day/per_week/per_month data)

## 3. Project Objective

Build a binary classifier to predict `is_fraudulent` for consumers and rank them by fraud risk for review or automated action.

## 4. Methods

### 4A. Exploratory Data Analysis

Univariate stats, distributions, and box plots by fraud status. Variables with clearer separation between fraud and non-fraud: transaction history, email traits, delivery behavior, spending patterns, account activity, device usage, and payment behavior.

### 4B. Data Cleaning and Preparation

- Categorical missing → `"Missing"`
- Numerical missing → `-999`
- Cast categorical columns to `category`
- Merge user, item, and activity on `consumer_id`

### 4C. Feature Engineering

All 29 features are used; no additional feature creation in the current pipeline. Preprocessing via `ColumnTransformer`: `OneHotEncoder(handle_unknown="ignore")` for categorical, `StandardScaler` for numerical.

### 4D. Feature Selection

Not implemented; all features are used.

### 4E. Model Selection and Hyperparameter Tuning

Baseline XGBClassifier in a `Pipeline` with the preprocessor. Hyperparameter tuning in the notebooks.

### 4F. Evaluation

ROC-AUC on training set (current baseline). Validation/test metrics can be added.

## 5. Incrementality Test

Placeholder — no implementation yet. Typical use: compare model predictions vs. random or business rules to measure lift.

## Project Structure

```
Digital-Marketplace-Fraud-Prediction-Model/
├── 01_eda_cleaning.ipynb    # EDA, cleaning, train/val/test split, save CSVs
├── 02_feature_eng.ipynb     # Load splits, preprocess, XGB pipeline, evaluation
├── datasets/
│   ├── user/                # user.csv, delivery_address.csv, email.csv
│   ├── item/                # latest_item.csv
│   ├── activity/            # per_day.csv, per_week.csv, per_month.csv
│   ├── train/               # X_train.csv, y_train.csv, ids_train.csv
│   ├── val/                 # X_val.csv, y_val.csv, ids_val.csv
│   └── test/                # X_test.csv, y_test.csv, id_test.csv
└── README.md
```

## Setup and Usage

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`

**Execution order:**

1. Run `01_eda_cleaning.ipynb` (produces train/val/test CSVs)
2. Run `02_feature_eng.ipynb` (loads splits, trains model)

## Data Split

- Stratified split: 80% test, 20% temp (train+val)
- Of temp: 80% train, 20% validation
- Approximate sizes: Train ~2,637, Val ~660, Test ~13,188
