import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np

# loadin the data
df = pd.read_csv('car data.csv')


# --- data prep ---

# convrt text cols to numbers for the model
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# X is features, y is what we want to predict
# car name is no good for predictshun, so drop it
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

# split data, 80 train 20 test
# random_state so its the same each time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


# --- model traning ---

# RndomForest is a solid choice for this
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# train on the traning data
regressor.fit(X_train, y_train)


# --- check how good it is ---

# see how it does on data it hasnt seen
predictions = regressor.predict(X_test)

# print the scors
print('--- Perfomance ---')
print(f'R2 Score: {metrics.r2_score(y_test, predictions):.4f}')
print(f'MAE: {metrics.mean_absolute_error(y_test, predictions):.4f}')
print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions)):.4f}')
print('------------------')


# --- vizualize ---

# scatter plot to see how close we are
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted")

# red line is a perfect predict
# closer the dots, the beter
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.show()