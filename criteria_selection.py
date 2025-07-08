import pandas as pd

# Load the preprocessed data
df = pd.read_csv('processed_restaurant_data.csv')

# --- Display Available Options ---
print("Available Cuisines:\n", df['Cuisines'].unique())
print("\nAvailable Price Ranges:\n", df['Price range'].unique())

# --- Simulated User Input ---
user_cuisine = input("\nEnter preferred cuisine (as shown above): ")
user_price = int(input("Enter preferred price range (1 to 4): "))

# --- Filter Based on User Input ---
filtered_df = df[
    (df['Cuisines'].str.contains(user_cuisine, case=False)) &
    (df['Price range'] == user_price)
]

# --- Display Results ---
if not filtered_df.empty:
    print(f"\nRestaurants matching '{user_cuisine}' with price range {user_price}:\n")
    print(filtered_df[['Restaurant Name', 'Cuisines', 'Price range']].head(10))
else:
    print("\nNo matching restaurants found.")
