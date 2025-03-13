import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Generate sample housing data (in a real project, you would load actual data)
np.random.seed(42)
n_samples = 1000

# Features
sqft = np.random.normal(1500, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)
location_score = np.random.uniform(1, 10, n_samples)

# Target: house prices with some noise
prices = 100000 + 100 * sqft + 20000 * bedrooms + 25000 * bathrooms - 1000 * age + 15000 * location_score
prices = prices + np.random.normal(0, 50000, n_samples)

# Create DataFrame
housing_data = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'location_score': location_score,
    'price': prices
})

# Split the data
X = housing_data.drop('price', axis=1)
y = housing_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Feature importance
coefficients = pd.DataFrame(model.coef_, X.columns)
coefficients.columns = ['Coefficient']
print("\nFeature Importance:")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Housing Prices')
plt.tight_layout()
plt.show()

# Sample prediction for a new house
new_house = pd.DataFrame({
    'sqft': [2000],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [10],
    'location_score': [7.5]
})
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)[0]
print(f"\nPredicted price for the sample house: ${predicted_price:.2f}")
