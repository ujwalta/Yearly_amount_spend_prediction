📊 E-Commerce Spending Prediction App

This project builds a machine learning model to predict yearly e-commerce spending based on user behavior metrics such as time spent on the website, time spent on the app, session length, and membership duration. The trained model is visualized and deployed using Streamlit.

🚀 Features
Train a Linear Regression model on e-commerce data

Visualize important relationships using Seaborn and Matplotlib

Evaluate the model using MAE, MSE, RMSE, and R² metrics

Predict Yearly Amount Spent based on custom user inputs

Display residual distribution and QQ plot

Interactive web app interface using Streamlit

🧠 Technologies Used
Python

Pandas

Seaborn & Matplotlib

Scikit-learn

Joblib

Streamlit

SciPy (for QQ plot)

📂 Files and Structure

├── ecommerce.csv                # Dataset file (upload in app)
├── model(1).pkl                 # Trained Linear Regression model
├── app.py                      # Streamlit web application
├── README.md                   # Project documentation
📈 Dataset Overview

The dataset ecommerce.csv contains user interaction data with the e-commerce platform including:

Avg. Session Length

Time on App

Time on Website

Length of Membership

Yearly Amount Spent (target)

💻 How to Run the App

Install dependencies:


pip install streamlit pandas matplotlib seaborn scikit-learn joblib scipy

Place the following in the same folder:

app.py

model(1).pkl (your trained model)

ecommerce.csv (dataset for analysis)

Run the app:

streamlit run app.py

🧪 Model Training Summary

The model was trained using the following features:

Avg. Session Length

Time on App

Time on Website

Length of Membership

Model performance:

MAE, MSE, and RMSE are displayed

R² Score indicates how well the model explains variance in spending

🔮 Prediction

You can input your own values in the sidebar to predict the Yearly Amount Spent by a customer.

📷 Visuals Included

Jointplots of feature vs. target

Pairplot of the dataset

Scatter plot of predictions vs actual values

Residuals distribution plot

QQ Plot for residuals

✅ Credits

Developed by Ujwalta Khanal

Data Source: Provided ecommerce.csv dataset
 
