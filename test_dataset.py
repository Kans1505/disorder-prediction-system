import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/disorder_data.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

print("\n" + "=" * 60)
print("DISORDER DISTRIBUTION")
print("=" * 60)
print(df['disorder'].value_counts())
print(f"\nPercentage:\n{df['disorder'].value_counts(normalize=True) * 100}")

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("STATISTICS (numeric columns)")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("GENDER DISTRIBUTION")
print("=" * 60)
print(df['gender'].value_counts())

print("\n✅ Dataset test complete!")