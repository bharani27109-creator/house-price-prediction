import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load the dataset
housing = fetch_california_housing()

# 2. Create DataFrame
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["PRICE"] = housing.target

# 3. Select features and target
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Save fresh model
joblib.dump(
    model,
    "house_price_model.pkl",
    compress=3,
    protocol=4
)

print("Fresh model saved successfully!")
