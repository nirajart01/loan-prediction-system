import os
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------
# Create folders
# ----------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")

# ----------------------------
# Missing Values
# ----------------------------

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

df["Credit_History"].fillna(
    df["Credit_History"].mode()[0],
    inplace=True
)

df["LoanAmount"].fillna(
    df["LoanAmount"].median(),
    inplace=True
)

df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].median(),
    inplace=True
)

# ----------------------------
# Feature Engineering
# ----------------------------

df["TotalIncome"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)

df["TotalIncome_log"] = np.log(
    df["TotalIncome"]
)

df["LoanAmount_log"] = np.log(
    df["LoanAmount"]
)

df["EMI"] = (
    df["LoanAmount"] /
    df["Loan_Amount_Term"]
)

df["Income_Loan_Ratio"] = (
    df["TotalIncome"] /
    df["LoanAmount"]
)

# ----------------------------
# Label Encoding
# ----------------------------

encoders = {}

columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    encoders[col] = encoder

# ----------------------------
# Save Clean Dataset
# ----------------------------

df.to_csv(
    "data/clean_dataset.csv",
    index=False
)

# ----------------------------
# Features
# ----------------------------

X = df.drop(
    ["Loan_ID", "Loan_Status"],
    axis=1
)

y = df["Loan_Status"]

# ----------------------------
# Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------
# Models
# ----------------------------

models = {

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

    "Decision Tree":
    DecisionTreeClassifier(random_state=42),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(),

    "XGBoost":
    XGBClassifier(
        random_state=42,
        learning_rate=0.05,
        max_depth=4,
        n_estimators=200,
        eval_metric="logloss"
    )
}

results = []

best_model = None
best_accuracy = 0

print("\nTraining Models...\n")

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    results.append(
        [name, accuracy]
    )

    print(
        f"{name:<25} : {accuracy:.4f}"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model

# ----------------------------
# Cross Validation
# ----------------------------

score = cross_val_score(
    best_model,
    X,
    y,
    cv=5
)

print("\nAverage CV Accuracy :",
      score.mean())

# ----------------------------
# Save Model
# ----------------------------

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/features.pkl"
)

joblib.dump(
    encoders,
    "models/encoders.pkl"
)

pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
).to_csv(
    "models/model_results.csv",
    index=False
)

print("\nFiles Saved Successfully")

print("""
models/
    best_model.pkl
    features.pkl
    encoders.pkl
    model_results.csv

data/
    clean_dataset.csv
""")