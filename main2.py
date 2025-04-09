import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load the dataset
df = pd.read_csv("C:/Users/vinod/Downloads/advertising.csv")

# 2. View basic info
print("Dataset Head:\n", df.head())
print("\nDataset Info:")
print(df.info())

# 3. Define features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]  # Advertising budgets
y = df['Sales']                       # Target: Product sales

# 4. Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluate using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")

# 8. Example predictions
pred_df = pd.DataFrame({'TV': X_test['TV'], 'Actual Sales': y_test, 'Predicted Sales': y_pred})
print("\nSample Predictions:\n", pred_df.head())
