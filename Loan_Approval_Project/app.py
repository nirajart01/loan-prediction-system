import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Main App */
.stApp{
    background-color:#f4f7fc;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f4c81,#1565c0);
    color:white;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p{
    color:white;
}

/* Title */
h1{
    color:#003366;
    font-weight:700;
    text-align:center;
}

h2{
    color:#004080;
}

h3{
    color:#0059b3;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:48px;
    border:none;
    border-radius:10px;
    background:#1565c0;
    color:white;
    font-size:17px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    background:#0d47a1;
    transform:scale(1.02);
}

/* Input Boxes */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{
    border-radius:10px;
}

/* Metric Cards */
.metric-card{
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0 6px 18px rgba(0,0,0,.08);
    transition:.3s;
}

.metric-card:hover{
    transform:translateY(-5px);
    box-shadow:0 12px 25px rgba(0,0,0,.15);
}

/* DataFrames */
[data-testid="stDataFrame"]{
    border-radius:12px;
    overflow:hidden;
}

/* Success Message */
.stSuccess{
    border-radius:10px;
}

/* Warning */
.stWarning{
    border-radius:10px;
}

/* Error */
.stError{
    border-radius:10px;
}

/* Expander */
.streamlit-expanderHeader{
    font-size:18px;
    font-weight:bold;
}

/* Divider */
hr{
    border:1px solid #d6d6d6;
}

/* Cards */
.card{
    background:white;
    border-radius:18px;
    padding:25px;
    box-shadow:0 8px 20px rgba(0,0,0,.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("models/best_model.pkl")

    features = joblib.load("models/features.pkl")

    encoders = joblib.load("models/encoders.pkl")

    return model,features,encoders

model,feature_columns,encoders = load_model()

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/clean_dataset.csv")

    return df

df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🏦 Loan Prediction System")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 Dataset",

        "📈 EDA Dashboard",

        "🤖 Prediction",

        "📉 Model Comparison",

        "⭐ Feature Importance",

        "📄 Classification Report",

        "🔲 Confusion Matrix"

    ]

)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if page=="🏠 Home":

    st.title("🏦 Loan Approval Prediction System")

    st.write("---")

    if os.path.exists("assets/bank.png"):

        st.image(
            "assets/bank.png",
            use_container_width=True
        )

    st.write("""

This project predicts whether a customer's loan will be approved
using Machine Learning.

The application was developed using

- Python
- Streamlit
- XGBoost
- Scikit-Learn
- Plotly
- Pandas

""")

    st.write("---")

    total_records=len(df)

    total_features=df.shape[1]

    approved=df[df["Loan_Status"]==1].shape[0]

    rejected=df[df["Loan_Status"]==0].shape[0]

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Total Records",
        total_records
    )

    c2.metric(
        "Features",
        total_features
    )

    c3.metric(
        "Approved",
        approved
    )

    c4.metric(
        "Rejected",
        rejected
    )

    st.write("---")

    st.subheader("Project Workflow")

    st.markdown("""

1. Load Dataset

2. Data Cleaning

3. Feature Engineering

4. Model Training

5. Model Evaluation

6. Loan Prediction

7. Dashboard Visualization

""")

    st.success("Use the sidebar to navigate through the project.")


# ==========================================================
# DATASET PAGE
# ==========================================================

elif page == "📊 Dataset":

    st.title("📊 Loan Dataset")

    st.write("### Dataset Shape")
    rows, cols = df.shape

    c1, c2 = st.columns(2)

    c1.metric("Rows", rows)
    c2.metric("Columns", cols)

    st.write("---")

    st.subheader("First 10 Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write("---")

    st.subheader("Dataset Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.write("---")

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    st.dataframe(
        missing,
        use_container_width=True
    )

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Clean Dataset",
        data=csv,
        file_name="clean_dataset.csv",
        mime="text/csv"
    )

# ==========================================================
# EDA DASHBOARD
# ==========================================================

elif page == "📈 EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Loan_Status",
            title="Loan Status Distribution",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="Gender",
            color="Loan_Status",
            barmode="group",
            title="Gender vs Loan Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Education",
            color="Loan_Status",
            title="Education vs Loan Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="Property_Area",
            color="Loan_Status",
            title="Property Area"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.write("---")

    fig = px.histogram(
        df,
        x="ApplicantIncome",
        nbins=40,
        title="Applicant Income Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.histogram(
        df,
        x="LoanAmount",
        nbins=40,
        title="Loan Amount Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Applicant Income vs Loan Amount")

    fig = px.scatter(
        df,
        x="ApplicantIncome",
        y="LoanAmount",
        color="Loan_Status",
        hover_data=df.columns
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Loan Amount by Property Area")

    fig = px.box(
        df,
        x="Property_Area",
        y="LoanAmount",
        color="Property_Area"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Credit History Distribution")

    fig = px.pie(
        df,
        names="Credit_History",
        title="Credit History"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================================
# LOAN PREDICTION PAGE
# ==========================================================

elif page == "🤖 Prediction":

    st.title("🤖 Loan Approval Prediction")

    st.write("Enter the applicant details below.")

    st.write("---")

    col1, col2 = st.columns(2)

    # =============================
    # LEFT COLUMN
    # =============================

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

    # =============================
    # RIGHT COLUMN
    # =============================

    with col2:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1,
            value=120
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [12,36,60,84,120,180,240,300,360]
        )

        credit_history = st.selectbox(
            "Credit History",
            [1,0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural","Semiurban","Urban"]
        )

    st.write("---")

    if st.button("Predict Loan Approval"):

        # ---------------------------------------
        # Encoding
        # ---------------------------------------

        gender = 1 if gender=="Male" else 0

        married = 1 if married=="Yes" else 0

        education = 0 if education=="Graduate" else 1

        self_employed = 1 if self_employed=="Yes" else 0

        dependents_map = {
            "0":0,
            "1":1,
            "2":2,
            "3+":3
        }

        property_map = {
            "Rural":0,
            "Semiurban":1,
            "Urban":2
        }

        dependents = dependents_map[dependents]

        property_area = property_map[property_area]

        # ---------------------------------------
        # Feature Engineering
        # ---------------------------------------

        total_income = applicant_income + coapplicant_income

        total_income_log = np.log(max(total_income,1))

        loan_amount_log = np.log(max(loan_amount,1))

        emi = loan_amount / loan_term

        income_loan_ratio = total_income / loan_amount

        input_data = pd.DataFrame({

            "Gender":[gender],

            "Married":[married],

            "Dependents":[dependents],

            "Education":[education],

            "Self_Employed":[self_employed],

            "ApplicantIncome":[applicant_income],

            "CoapplicantIncome":[coapplicant_income],

            "LoanAmount":[loan_amount],

            "Loan_Amount_Term":[loan_term],

            "Credit_History":[credit_history],

            "Property_Area":[property_area],

            "TotalIncome":[total_income],

            "TotalIncome_log":[total_income_log],

            "LoanAmount_log":[loan_amount_log],

            "EMI":[emi],

            "Income_Loan_Ratio":[income_loan_ratio]

        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        confidence = probability.max()*100

        st.write("---")

        if prediction == 1:

            st.success(
                f"✅ Loan Approved\n\nConfidence : {confidence:.2f}%"
            )

        else:

            st.error(
                f"❌ Loan Rejected\n\nConfidence : {confidence:.2f}%"
            )

        st.write("---")

        col1,col2 = st.columns(2)

        with col1:

            gauge = go.Figure(go.Indicator(

                mode="gauge+number",

                value=confidence,

                title={"text":"Prediction Confidence"},

                gauge={

                    "axis":{"range":[0,100]},

                    "bar":{"color":"green"}

                }

            ))

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        with col2:

            st.subheader("Applicant Summary")

            summary = pd.DataFrame({

                "Feature":[
                    "Applicant Income",
                    "Coapplicant Income",
                    "Loan Amount",
                    "Loan Term",
                    "Credit History",
                    "Property Area"
                ],

                "Value":[
                    applicant_income,
                    coapplicant_income,
                    loan_amount,
                    loan_term,
                    credit_history,
                    property_area
                ]

            })

            st.table(summary)

        st.write("---")

        st.subheader("Prediction Probability")

        chart = pd.DataFrame({

            "Class":["Rejected","Approved"],

            "Probability":probability

        })

        fig = px.bar(

            chart,

            x="Class",

            y="Probability",

            text="Probability",

            color="Class"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )


        # ==========================================================
# MODEL COMPARISON
# ==========================================================

elif page == "📉 Model Comparison":

    st.title("📉 Model Comparison")

    try:

        results = pd.read_csv("models/model_results.csv")

        st.dataframe(results, use_container_width=True)

        fig = px.bar(
            results,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Accuracy",
            title="Model Accuracy Comparison"
        )

        fig.update_layout(xaxis_title="", yaxis_title="Accuracy")

        st.plotly_chart(fig, use_container_width=True)

        best = results.loc[results["Accuracy"].idxmax()]

        st.success(
            f"Best Model : {best['Model']} ({best['Accuracy']:.4f})"
        )

    except Exception as e:

        st.error("Unable to load model_results.csv")
        st.exception(e)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

elif page == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": feature_columns,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        st.dataframe(
            importance,
            use_container_width=True
        )

        fig = px.bar(

            importance,

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance",

            title="Top Important Features"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Current model does not support Feature Importance."
        )


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

elif page == "📄 Classification Report":

    st.title("📄 Classification Report")

    X = df.drop(
        ["Loan_ID", "Loan_Status"],
        axis=1,
        errors="ignore"
    )

    y = df["Loan_Status"]

    pred = model.predict(X)

    report = classification_report(
        y,
        pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.success("Classification report generated successfully.")


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

elif page == "🔲 Confusion Matrix":

    st.title("🔲 Confusion Matrix")

    X = df.drop(
        ["Loan_ID", "Loan_Status"],
        axis=1,
        errors="ignore"
    )

    y = df["Loan_Status"]

    pred = model.predict(X)

    cm = confusion_matrix(y, pred)

    fig = px.imshow(

        cm,

        text_auto=True,

        color_continuous_scale="Blues",

        labels=dict(
            x="Predicted",
            y="Actual",
            color="Count"
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("### Matrix Values")

    st.dataframe(
        pd.DataFrame(

            cm,

            index=["Actual Reject","Actual Approve"],

            columns=["Pred Reject","Pred Approve"]

        ),

        use_container_width=True

    )

    accuracy = (cm[0,0]+cm[1,1]) / cm.sum()

    st.metric(
        "Training Accuracy",
        f"{accuracy*100:.2f}%"
    )

