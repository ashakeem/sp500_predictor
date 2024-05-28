# S&P 500 Prediction

Predicting the S&P 500 index using historical stock data.

## Key Areas

- **Data Collection**: Historical S&P 500 stock prices including features like Date, Open, High, Low, Close, and Volume.
- **Data Preprocessing**: Cleaning data by removing irrelevant columns and handling missing values.
- **Exploratory Data Analysis (EDA)**: Visualizing trends and analyzing feature correlations.
- **Feature Engineering**: Creating new features such as moving averages and volatility measures.
- **Model Training**: Training models (e.g., Linear Regression, Decision Trees, LSTM) and tuning hyperparameters.
- **Model Evaluation**: Assessing models using metrics like MAE, MSE, and R².

## Dependencies

Google Collab

Required libraries: `pandas`, `numpy`, `scikit-learn`, `yfinance`.

## Data Collection

```python
import yfinance as yf

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
sp500 = sp500.loc["1950-01-01":].copy()
```

## Data Preprocessing

### Removing Irrelevant Columns

```python
sp500 = sp500.drop(columns=["Dividends", "Stock Splits"])
```

### Handling Missing Values

```python
sp500 = sp500.dropna()
```

## Feature Engineering

### Adding Moving Averages and Volatility Measures

```python
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500['Close'] / rolling_averages['Close']
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]
```

## Model Training

### Training a Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
```

## Model Evaluation

### Predictions and Precision Score

```python
from sklearn.metrics import precision_score
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index, name="Predictions")
combined = pd.concat([test["Target"], preds], axis=1)
precision_score(test["Target"], preds)
```

### Plotting Predictions vs Actual

```python
combined.plot()
```

### Precision Score

```python
0.75
```

## Complete Function for Prediction

```python
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
```

## Backtesting

### Function for Backtesting

```python
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
```

### Value Counts and Precision Score

```python
predictions["Predictions"].value_counts()
precision_score(predictions["Target"], predictions["Predictions"])
```
