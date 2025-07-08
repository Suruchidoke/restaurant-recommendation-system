import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('restaurant_data.csv')

print("First 5 rows of the dataset:\n")
print(df.head())

print("\nColumn names:\n")
print(df.columns)

print("\nMissing values in each column:\n")
print(df.isnull().sum())

df['Cuisines'] = df['Cuisines'].fillna('Unknown')

label_encoder = LabelEncoder()
df['Cuisine_Encoded'] = label_encoder.fit_transform(df['Cuisines'])
df['Price_Encoded'] = label_encoder.fit_transform(df['Price range'])

df.to_csv('processed_restaurant_data.csv', index=False)

print("\n✅ Preprocessing complete. Encoded file saved as 'processed_restaurant_data.csv'")
