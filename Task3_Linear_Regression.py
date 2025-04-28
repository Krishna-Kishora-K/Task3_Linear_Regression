# Task 3: Linear Regression - House Price Prediction

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the Dataset
df = pd.read_csv('Housing.csv')

# Step 3: Explore the Dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nInformation about dataset:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Step 4: Preprocessing - Convert Categorical Columns to Numeric

# Replace 'yes' -> 1, 'no' -> 0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# For 'furnishingstatus' column (more than two categories)
df['furnishingstatus'] = df['furnishingstatus'].map({
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
})

# Check after encoding
print("\nAfter Encoding:")
print(df.head())

# Step 5: Select Features and Target
X = df.drop('price', axis=1)   # All features except 'price'
y = df['price']                # Target variable 'price'

# Step 6: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of training features:", X_train.shape)
print("Shape of testing features:", X_test.shape)

# Step 7: Build Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Model Coefficients and Intercept
print("\nModel Coefficients:")
print(model.coef_)

print("\nModel Intercept:")
print(model.intercept_)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Step 11: Visualize the Regression Line (only for 'area' feature)

plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price')
plt.scatter(X_test['area'], y_pred, color='red', label='Predicted Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price based on Area')
plt.legend()
plt.show()
