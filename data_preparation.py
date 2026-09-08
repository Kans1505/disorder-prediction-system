import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load raw data
df = pd.read_csv('data/raw/disorder_data.csv')
print(f"✅ Loaded {len(df)} records")

# Encode categorical columns
le_gender = LabelEncoder()
le_disorder = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['disorder'] = le_disorder.fit_transform(df['disorder'])

print(f"✅ Encoded gender: {le_gender.classes_}")
print(f"✅ Encoded disorders: {le_disorder.classes_}")

# Separate features and target
X = df.drop('disorder', axis=1)
y = df['disorder']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"✅ Scaled features: mean={X_scaled.mean().mean():.2f}, std={X_scaled.std().mean():.2f}")

# Train/val/test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\n✅ SPLIT SIZES:")
print(f"   Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Val:   {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
print(f"   Test:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

# Save processed data
X_train.to_csv('data/processed/X_train.csv', index=False)
X_val.to_csv('data/processed/X_val.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_val.to_csv('data/processed/y_val.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print(f"\n✅ Processed data saved to data/processed/")