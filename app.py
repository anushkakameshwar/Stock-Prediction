import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.saving import register_keras_serializable
import tensorflow as tf
import matplotlib.pyplot as plt

@register_keras_serializable(package="Custom", name="rmse")
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.losses.MeanSquaredError()(y_true, y_pred))

model = load_model("C://Users//hp//Desktop//stck_prdctn//Stock_Prediction_Models.keras", custom_objects={"rmse": rmse})  

st.header('Stock Market Predictor')

stock = st.text_input('Enter stock symbol (e.g., GOOG)', 'GOOG').upper()
start, end = '2015-01-01', '2024-12-31'

try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error("No data found. Please check the stock symbol and try again.")
    else:
        st.subheader('Stock Data')
        st.write(data)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

train_size = int(len(data) * 0.80)
data_train, data_test = data.Close[:train_size], data.Close[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train.values.reshape(-1, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test.values.reshape(-1, 1))

x, y = [], []
for i in range(100, len(data_test_scaled)):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

if x.shape[0] > 0:
    y_pred_scaled = model.predict(x)
    y_pred = y_pred_scaled * (1 / scaler.scale_[0])  
    y_actual = y * (1 / scaler.scale_[0])

    st.subheader("Predicted vs. Actual Stock Prices")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, 'g', label="Actual Price")
    plt.plot(y_pred, 'r', label="Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)
else:
    st.warning("Not enough data points for prediction.")