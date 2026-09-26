import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


DATA_URL = "https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write("Analyze customer data and predict the likelihood of customer churn.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


@st.cache_resource
def train_model(data):

    df = data.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    df.drop("customerID", axis=1, inplace=True)

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    probability = model.predict(
        X_test,
        verbose=0
    ).ravel()

    prediction = (probability >= 0.5).astype(int)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    report = classification_report(
        y_test,
        prediction,
        output_dict=True
    )

    return model, accuracy, auc, report


try:

    data = load_data()

    st.success("Customer dataset loaded successfully.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customers",
            f"{len(data):,}"
        )

    with col2:
        churn_rate = (
            data["Churn"]
            .value_counts(normalize=True)
            .get("Yes", 0) * 100
        )

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with col3:
        st.metric(
            "Features",
            data.shape[1]
        )


    st.subheader("Customer Data")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )


    st.subheader("Churn Distribution")

    churn_count = data["Churn"].value_counts()

    st.bar_chart(churn_count)


    st.subheader("Model Training")

    if st.button("Train Churn Model"):

        with st.spinner("Training model..."):

            model, accuracy, auc, report = train_model(data)

        st.success("Model trained successfully.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Test Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with col2:
            st.metric(
                "ROC-AUC",
                f"{auc:.3f}"
            )

        st.subheader("Classification Report")

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )


except Exception as e:

    st.error(
        "Unable to load or process the dataset."
    )

    st.write(str(e))
