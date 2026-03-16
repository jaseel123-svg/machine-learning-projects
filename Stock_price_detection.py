import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def prepare_data(df, forecast_col, forecast_out, test_size):

    # create label column (future price)
    df['label'] = df[forecast_col].shift(-forecast_out)

    # use multiple features
    X = np.array(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    X = preprocessing.scale(X)

    # data for future prediction
    X_lately = X[-forecast_out:]

    # remove last rows used for forecast
    X = X[:-forecast_out]

    df.dropna(inplace=True)
    y = np.array(df['label'])

    # split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    return X_train, X_test, Y_train, Y_test, X_lately


# load dataset
df = pd.read_csv(r"C:\Users\jasee\Downloads\apple stocks price\Apple.csv")

forecast_col = "Close"
forecast_out = 5
test_size = 0.2

# prepare data
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(
    df, forecast_col, forecast_out, test_size
)

# train model
model = LinearRegression()
model.fit(X_train, Y_train)

# predictions
score = model.score(X_test, Y_test)
predicted = model.predict(X_test)
forecast = model.predict(X_lately)

# evaluation
mse = mean_squared_error(Y_test, predicted)

print("Model accuracy (R2 Score):", score)
print("Mean Squared Error:", mse)
print("Next 5 day predicted prices:", forecast)


# plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(Y_test, label="Actual Prices")
plt.plot(predicted, label="Predicted Prices")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Test Data Points")
plt.ylabel("Price")
plt.legend()
plt.show()


# Plot Future Forecast
plt.figure(figsize=(6,4))
plt.plot(forecast, marker='o')
plt.title("Next 5 Day Stock Price Forecast")
plt.xlabel("Future Days")
plt.ylabel("Predicted Price")
plt.show()