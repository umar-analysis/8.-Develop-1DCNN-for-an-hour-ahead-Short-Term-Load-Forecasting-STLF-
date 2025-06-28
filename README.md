## 🧪 Lab 8: 1D CNN for Hour-Ahead Short-Term Load Forecasting (STLF)

This lab demonstrates how to use a **1D Convolutional Neural Network (1D-CNN)** to perform **Short-Term Load Forecasting (STLF)**, specifically predicting the electrical load one hour ahead based on historical time-series data.

### ⚙️ Key Components

- **📊 Time-Series Data Preparation**  
  Sliding window approach to generate sequences of past load data as input features and the next hour as the prediction target.

- **🧠 1D-CNN Model Development**  
  Constructed using Keras/TensorFlow with convolutional and dense layers optimized for temporal pattern recognition.

- **🧪 Training & Evaluation**  
  The model is trained and evaluated using metrics like **MAE**, **RMSE**, and **MAPE** to assess forecasting accuracy.

- **⏱️ Hour-Ahead Forecasting**  
  The trained model is used to make one-hour-ahead predictions, enabling smart grid demand planning and energy optimization.


