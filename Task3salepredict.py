import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# get the data
df = pd.read_csv('Advertising.csv')
# drop that weird extra colum
df = df.drop(columns=['Unnamed: 0'])

# X is the stuff we use to predict
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales'] # y is what we want to gues

# split data for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- traning the model ---
model = LinearRegression()
model.fit(X_train, y_train)

# now make predictshuns on the test data
predictions = model.predict(X_test)

# --- see how good it is ---
print('--- Perfomance ---')
r2 = metrics.r2_score(y_test, predictions) # r-squared
print(f'R-squared: {r2:.4f}')

mae = metrics.mean_absolute_error(y_test, predictions)
print(f'MAE: {mae:.4f}') # mean abs error

rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse:.4f}')
print('------------------')


# plot the results to vizualize it
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions) # acutal vs predicted
plt.xlabel("Acutal Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted")

# red line is perfect score
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.grid(True)
plt.show() # show the thing