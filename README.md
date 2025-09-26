# Energy Consumption Forecasting

A machine learning project that predicts residential energy consumption using smart meter data and advanced time series modeling techniques.

## Project Overview

This project develops predictive models to forecast household energy consumption using data from 5,567 London households collected between November 2011 and February 2014. By leveraging LSTM neural networks and ensemble methods, the system achieves accurate consumption predictions to enable better energy grid management and demand optimization.

## Key Features

- **Time Series Forecasting**: LSTM and Random Forest models for hourly/daily consumption prediction
- **Feature Engineering**: Seasonal patterns, weather integration, and temporal feature extraction
- **Interactive Dashboard**: Real-time visualization of consumption patterns and predictions
- **Automated Pipeline**: End-to-end ML pipeline from data ingestion to model deployment
- **Performance Metrics**: RMSE of 850 kWh, R² score of 0.87, MAPE of 8.2%

## Dataset

- **Source**: London Smart Meter Data (Kaggle)
- **Size**: 52,000+ hourly observations from 5,567 households
- **Features**: Timestamp, household ID, energy consumption, weather data
- **Duration**: 3+ years of continuous smart meter readings

## Technology Stack

- **Languages**: Python, SQL
- **ML Libraries**: TensorFlow/Keras, scikit-learn, pandas, numpy
- **Visualization**: Plotly, Dash, Matplotlib, Seaborn
- **Database**: MongoDB
- **Deployment**: Flask, AWS
- **Tools**: Jupyter, Git

## Quick Start
```bash
