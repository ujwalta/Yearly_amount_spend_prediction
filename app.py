import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import scipy.stats as stats
import pylab

# Title
st.title("E-Commerce Spending Prediction App")

# Load model
model = joblib.load('linear_regression_model.pkl')

# Upload CSV to run full analysis
uploaded_file = st.file_uploader("Upload the ecommerce.csv file for analysis", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Features and target
    X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
    y = df['Yearly Amount Spent']

    # Split
    from sklearn.model_selection import train_test_split
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    # Prediction
    prediction = model.predict(X_test)

    # Show Coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    st.write(coef_df)

    # Evaluation
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)

    st.subheader("Evaluation Metrics")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**Root Mean Squared Error:** {rmse:.2f}")
    st.write(f"**R-squared:** {r2:.4f}")

    # Scatter plot: Prediction vs Actual
    st.subheader("Prediction vs Actual")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=prediction, y=y_test, ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Predicted vs Actual")
    st.pyplot(fig1)

    # Residuals Distribution
    residuals = y_test - prediction
    st.subheader("Residuals Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(residuals, bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    # QQ Plot
    st.subheader("QQ Plot of Residuals")
    fig3 = plt.figure()
    stats.probplot(residuals, dist="norm", plot=pylab)
    st.pyplot(fig3)

# Sidebar for manual prediction
st.sidebar.header("Enter User Data for Prediction")
avg_session_length = st.sidebar.number_input("Avg. Session Length", value=33.0)
time_on_app = st.sidebar.number_input("Time on App", value=12.0)
time_on_website = st.sidebar.number_input("Time on Website", value=37.0)
length_of_membership = st.sidebar.number_input("Length of Membership", value=3.0)

if st.sidebar.button("Predict Yearly Amount Spent"):
    user_input = [[avg_session_length, time_on_app, time_on_website, length_of_membership]]
    result = model.predict(user_input)
    st.sidebar.success(f"Predicted Yearly Amount Spent: ${result[0]:.2f}")
