import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import sys

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Features (remove G3 from features)
X = numeric_df.drop("G3", axis=1)

# Target
y = numeric_df["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))
