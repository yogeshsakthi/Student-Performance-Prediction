
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("data/student_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# Correlation Heatmap (Numeric only)
# -------------------------------
plt.figure(figsize=(15, 8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -------------------------------
# Scatter Plot: G1 vs G3
# -------------------------------
plt.figure()
sns.scatterplot(x="G1", y="G3", data=df)
plt.title("G1 vs Final Grade (G3)")
plt.show()

# -------------------------------
# Scatter Plot: G2 vs G3
# -------------------------------
plt.figure()
sns.scatterplot(x="G2", y="G3", data=df)
plt.title("G2 vs Final Grade (G3)")
plt.show()

# -------------------------------
# Distribution of Final Grade
# -------------------------------
plt.figure()
sns.histplot(data=df, x="G3", kde=True)
plt.title("Distribution of Final Grade (G3)")
plt.show()

print("\nAnalysis Completed Successfully ")
