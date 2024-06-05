# Time Series Decomposition (Cycles)

## Introduction

Time series decomposition is a technique that breaks down a time series into its constituent components: trend, seasonality, and residuals (noise). This process helps in understanding the underlying patterns in the data and is essential for accurate forecasting and analysis.

### Components of Time Series

1. **Trend**: The long-term progression of the series. It represents the underlying direction in which the data is moving over time.
2. **Seasonality**: The repeating short-term cycle in the series. It captures regular patterns or cycles that occur within a specific period (e.g., daily, monthly, yearly).
3. **Residuals (Noise)**: The random variation in the series. It includes any irregular fluctuations that cannot be attributed to the trend or seasonality.

## Mathematical Formulation

The decomposition can be represented mathematically as:

\[ y(t) = T(t) + S(t) + R(t) \]

where:
- \( y(t) \) is the observed time series at time \( t \)
- \( T(t) \) is the trend component
- \( S(t) \) is the seasonal component
- \( R(t) \) is the residual component

Alternatively, for multiplicative decomposition, the relationship is:

\[ y(t) = T(t) \times S(t) \times R(t) \]

## Process of Decomposition

### Using Python

#### 1. Load Data

First, load your time series data into a pandas DataFrame. Ensure your data has a datetime index.

```python
import pandas as pd

# Load your data
data = pd.read_csv('your_time_series_data.csv', parse_dates=['date'], index_col='date')
```

#### 2. Decompose the Time Series

Using a time series decomposition tool (e.g., seasonal_decompose from statsmodels), decompose the series into its components.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
result = seasonal_decompose(data['value'], model='additive')

# Extract the components
trend = result.trend
seasonal = result.seasonal
residual = result.resid
```

#### 3. Visualize the Components

Visualizing the components helps in understanding the underlying patterns in the data.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Original Series
plt.subplot(411)
plt.plot(data['value'], label='Original')
plt.legend(loc='upper left')

# Trend
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')

# Seasonality
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')

# Residuals
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
```

#### 4. Using Prophet for Decomposition (Optional)

While the focus here is on the theory and process, tools like Prophet can also be used for time series decomposition. Here's a brief example using Prophet:

```python
from prophet import Prophet

# Prepare the data
df = data.reset_index().rename(columns={'date': 'ds', 'value': 'y'})

# Fit the model
model = Prophet()
model.fit(df)

# Make predictions
forecast = model.predict(df)

# Plot components
fig = model.plot_components(forecast)
plt.show()
```

### Using R

#### 1. Load Data

First, load your time series data into an R dataframe. Ensure your data has a datetime index.

```r
library(readr)
library(dplyr)

# Load your data
data <- read_csv('your_time_series_data.csv') %>%
  mutate(date = as.Date(date))
```

#### 2. Decompose the Time Series

Using a time series decomposition tool (e.g., `stl` function), decompose the series into its components.

```r
library(forecast)

# Convert to time series object
ts_data <- ts(data$value, frequency = 12) # Adjust frequency as needed

# Decompose the time series
decomposed <- stl(ts_data, s.window = "periodic")

# Extract the components
trend <- decomposed$time.series[, "trend"]
seasonal <- decomposed$time.series[, "seasonal"]
residual <- decomposed$time.series[, "remainder"]
```

#### 3. Visualize the Components

Visualizing the components helps in understanding the underlying patterns in the data.

```r
library(ggplot2)

# Plot the components
autoplot(decomposed) +
  ggtitle("Time Series Decomposition") +
  xlab("Time") +
  ylab("Values")
```

#### 4. Using Prophet for Decomposition (Optional)

While the focus here is on the theory and process, tools like Prophet can also be used for time series decomposition. Here's a brief example using Prophet:

```r
library(prophet)

# Prepare the data
df <- data %>%
  rename(ds = date, y = value)

# Fit the model
model <- prophet(df)

# Make predictions
future <- make_future_dataframe(model, periods = 365) # Adjust periods as needed
forecast <- predict(model, future)

# Plot components
prophet_plot_components(model, forecast)
```

## Conclusion

Time series decomposition is a powerful tool for analyzing the components of a time series. By breaking down the series into trend, seasonality, and residuals, you can better understand the underlying patterns and make more accurate forecasts.
