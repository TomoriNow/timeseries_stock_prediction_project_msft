# Microsoft Stock Price Prediction using LSTM

A deep learning project that uses Long Short-Term Memory (LSTM) neural networks to predict Microsoft stock prices based on historical data.

## Project Overview

This project implements a time series forecasting model to predict Microsoft stock prices using TensorFlow/Keras. The model uses historical stock price data to learn patterns and make future predictions.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis of Microsoft stock data
- **Visualization**: Multiple plots showing stock price trends, volume analysis, and correlation heatmaps
- **LSTM Model**: Deep learning model with two LSTM layers and dropout for regularization
- **Performance Metrics**: Evaluation using MSE, MAE, RMSE, R², and MAPE
- **Prediction Visualization**: Comparison plots between actual and predicted stock prices

## Dataset

The project uses `MicrosoftStock.csv` which contains historical Microsoft stock data with the following columns:
- `date`: Date of the trading day
- `open`: Opening price
- `close`: Closing price
- `volume`: Trading volume
- `Name`: Stock name (Microsoft)

## Model Architecture

The LSTM model consists of:
- **Input Layer**: Sequences of 60 previous days' closing prices
- **LSTM Layer 1**: 64 units with return_sequences=True
- **Dropout Layer 1**: 20% dropout rate
- **LSTM Layer 2**: 64 units 
- **Dense Layer**: 128 units with ReLU activation
- **Dropout Layer 2**: 50% dropout rate
- **Output Layer**: 1 unit for price prediction

## Key Results

The model provides:
- Time series predictions for Microsoft stock prices
- Performance evaluation metrics
- Visual comparisons between actual and predicted prices
- Full timeline visualization showing training/test split

## Installation

1. Clone this repository or download the project files
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure `MicrosoftStock.csv` is in the project directory
2. Open `stock_prediction.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially to:
   - Load and analyze the data
   - Train the LSTM model
   - Generate predictions
   - Visualize results
   - Calculate performance metrics

## Project Structure

```
time_series_forecast_project/
│
├── stock_prediction.ipynb    # Main Jupyter notebook
├── MicrosoftStock.csv       # Historical stock data
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Model Performance

The model is evaluated using multiple metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**
- **Mean Absolute Percentage Error (MAPE)**

## Visualizations

The notebook includes several visualization types:
1. **Stock Price Trends**: Open vs Close prices over time
2. **Volume Analysis**: Trading volume patterns
3. **Correlation Heatmap**: Relationships between numeric features
4. **Prediction Comparison**: Actual vs predicted prices
5. **Full Timeline View**: Complete dataset with train/test split
6. **Zoomed Test Period**: Detailed view of prediction accuracy

## Technical Details

- **Sequence Length**: 60 days of historical data for each prediction
- **Train/Test Split**: 80/20 split with temporal ordering preserved
- **Data Scaling**: StandardScaler for both features and targets
- **Optimizer**: Adam optimizer
- **Loss Function**: Mean Absolute Error (MAE)
- **Training**: 50 epochs with batch size of 32

## Requirements

See `requirements.txt` for a complete list of dependencies.
