import pandas as pd

# Load dataset
data = pd.read_csv('data/sample.csv')

# Display first rows
print("Dataset preview:")
print(data.head())

# Basic statistics
print("\nStatistics:")
print(data.describe())

# Count labels
print("\nLabel distribution:")
print(data['label'].value_counts())
