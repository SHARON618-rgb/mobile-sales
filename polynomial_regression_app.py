import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st

st.title("📈 Mobile Sales Revenue Prediction for August 2024")

uploaded_file = st.file_uploader("Upload your mobile-sales.csv file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    
    df['date'] = pd.to_datetime(df['Dates'], dayfirst=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df = df.drop(columns=['CustomerID', 'Dates', 'date'], errors='ignore')
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns='TotalRevenue')
    y = df['TotalRevenue']


    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    future_dates = pd.date_range(start='2025-08-01', periods=30)
    future_df = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month,
        'day': future_dates.day
    })

    for col in X.columns:
        if col in future_df.columns:
            continue
        future_df[col] = df[col].mode()[0] if df[col].dtype == 'uint8' else df[col].mean()

    future_df = future_df[X.columns]
    future_scaled = scaler.transform(future_df)
    future_predictions = model.predict(future_scaled)

    prediction_output = pd.DataFrame({
        'Date': future_dates,
        'PredictedRevenue': future_predictions
    })

    st.subheader("📊 Predicted Revenue for August 2024")
    st.dataframe(prediction_output)

    st.line_chart(prediction_output.set_index('Date'))